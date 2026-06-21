package com.nightroadvision.app.inference

import android.content.Context
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.Closeable
import java.nio.ByteBuffer

/**
 * ONNX Runtime engine for the openpilot supercombo driving model.
 * Parses lead vehicle detections from the flat 6409-dim output vector.
 */
class SupercomboEngine(private val context: Context) : Closeable {

    companion object {
        private const val TAG = "SupercomboEngine"
    }

    private val env = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    private var isModelLoaded = false

    fun loadModel(assetPath: String): Boolean {
        try {
            closeSession()
            val modelBytes = context.assets.open(assetPath).use { it.readBytes() }
            val opts = OrtSession.SessionOptions()
            session = env.createSession(modelBytes, opts)
            isModelLoaded = true

            session?.let { s ->
                Log.i(TAG, "Model loaded: $assetPath")
                Log.i(TAG, "Input names: ${s.inputNames}")
                Log.i(TAG, "Output names: ${s.outputNames}")
            }
            return true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model: $assetPath", e)
            isModelLoaded = false
            return false
        }
    }

    fun runInference(inputs: SupercomboPreprocessor.ModelInputs): SupercomboOutput? {
        val s = session ?: return null
        if (!isModelLoaded) return null

        val startTime = System.nanoTime()

        try {
            val inputTensors = mutableMapOf<String, OnnxTensor>()

            // Image: (1, 12, 128, 256)
            inputTensors[SupercomboConstants.IN_IMGS] = OnnxTensor.createTensor(
                env, inputs.img, longArrayOf(
                    1,
                    SupercomboConstants.STACKED_CHANNELS.toLong(),
                    SupercomboConstants.MODEL_HEIGHT.toLong(),
                    SupercomboConstants.MODEL_WIDTH.toLong()
                )
            )

            // Desire: (1, 8)
            inputTensors[SupercomboConstants.IN_DESIRE] = OnnxTensor.createTensor(
                env, inputs.desire, longArrayOf(1, SupercomboConstants.DESIRE_LEN.toLong())
            )

            // Traffic convention: (1, 2)
            inputTensors[SupercomboConstants.IN_TRAFFIC_CONVENTION] = OnnxTensor.createTensor(
                env, inputs.trafficConvention, longArrayOf(1, SupercomboConstants.TRAFFIC_CONVENTION_LEN.toLong())
            )

            // Initial state (hidden state): (1, 512)
            inputTensors[SupercomboConstants.IN_INITIAL_STATE] = OnnxTensor.createTensor(
                env, inputs.prevFeatures, longArrayOf(1, SupercomboConstants.FEATURE_LEN.toLong())
            )

            // Run inference
            val results = s.run(inputTensors)
            val inferenceTimeMs = (System.nanoTime() - startTime) / 1_000_000

            // Extract the single flat output
            val outputTensor = results.get(SupercomboConstants.OUT_OUTPUTS).orElse(null)
                as? OnnxTensor
            if (outputTensor == null) {
                Log.e(TAG, "Output tensor '${SupercomboConstants.OUT_OUTPUTS}' not found")
                inputTensors.values.forEach { it.close() }
                results.close()
                return null
            }

            val outputArray = getFloatArray(outputTensor)
            Log.d(TAG, "Output size: ${outputArray.size}, first 10: ${outputArray.take(10).joinToString { "%.4f".format(it) }}")
            val leads = parseLeadOutput(outputArray)

            // Note: this ONNX model does NOT expose GRU hidden state as a separate output.
            // The recurrent state flows internally through the ONNX graph.
            // We pass zeros as initial_state; the model's internal GRU handles temporal context.

            inputTensors.values.forEach { it.close() }
            results.close()

            return SupercomboOutput(
                leads = leads,
                inferenceTimeMs = inferenceTimeMs
            )

        } catch (e: Exception) {
            Log.e(TAG, "Inference failed", e)
            return null
        }
    }

    /**
     * Parse lead vehicles from the flat 6409-dim output vector.
     * Lead data uses MHP (mixture of hypotheses): 3 leads × 3 MHP × 6 timesteps × 4 values.
     * We use the first MHP hypothesis (index 0) for each lead.
     * Lead probs at LEAD_PROB_OFFSET, shape (3) = 3 values.
     */
    private fun parseLeadOutput(output: FloatArray): List<LeadVehicle> {
        val leads = mutableListOf<LeadVehicle>()

        if (output.size < SupercomboConstants.LEAD_PROB_OFFSET + SupercomboConstants.LEAD_PROB_SIZE) {
            Log.w(TAG, "Output too small for lead parsing: ${output.size} < ${SupercomboConstants.LEAD_PROB_OFFSET + SupercomboConstants.LEAD_PROB_SIZE}")
            return leads
        }

        // Read lead probabilities (sigmoid applied)
        val probs = FloatArray(SupercomboConstants.LEAD_MAX)
        for (i in 0 until SupercomboConstants.LEAD_MAX) {
            val raw = output[SupercomboConstants.LEAD_PROB_OFFSET + i]
            probs[i] = sigmoid(raw)
            Log.d(TAG, "Lead $i raw prob=$raw sigmoid=${probs[i]}")
        }

        // MHP structure: each lead has 3 hypotheses × 6 timesteps × 4 values = 72 floats
        // Layout: [lead0_hyp0(24), lead0_hyp1(24), lead0_hyp2(24),
        //          lead1_hyp0(24), lead1_hyp1(24), lead1_hyp2(24), ...]
        val hypStride = SupercomboConstants.LEAD_TRAJ_LEN * SupercomboConstants.LEAD_WIDTH  // 24
        val leadStride = hypStride * 3  // 72 (3 MHP hypotheses per lead)

        for (leadIdx in 0 until SupercomboConstants.LEAD_MAX) {
            val prob = probs[leadIdx]
            if (prob < SupercomboConstants.LEAD_PROB_THRESHOLD) {
                Log.d(TAG, "Lead $leadIdx prob=$prob below threshold ${SupercomboConstants.LEAD_PROB_THRESHOLD}")
                continue
            }

            // Use first MHP hypothesis (index 0) for each lead
            val baseOffset = SupercomboConstants.LEAD_OFFSET + leadIdx * leadStride

            if (baseOffset + 3 >= output.size) {
                Log.w(TAG, "Lead $leadIdx offset $baseOffset out of bounds")
                continue
            }

            // First time step of first hypothesis: x, y, velocity, acceleration
            val x = output[baseOffset + 0]  // longitudinal distance (m)
            val y = output[baseOffset + 1]  // lateral offset (m)
            val v = output[baseOffset + 2]  // velocity (m/s)
            val a = output[baseOffset + 3]  // acceleration (m/s²)

            val distance = Math.sqrt((x * x + y * y).toDouble()).toFloat()

            // Map vehicle-frame coords to normalized camera coords
            // x = forward distance, y = lateral offset
            // Camera center = (0.5, 0.5), forward maps to upper part of frame
            val scale = 0.01f
            val cameraX = (0.5f + y * scale).coerceIn(0f, 1f)
            val cameraY = (0.5f - x * scale * 0.5f).coerceIn(0f, 1f)

            // Rough bbox estimate
            val approxBoxHeight = (0.12f / maxOf(distance, 1f)).coerceIn(0.01f, 0.5f)
            val approxBoxWidth = approxBoxHeight * 0.5f

            Log.i(TAG, "Lead $leadIdx: x=$x y=$y v=$v dist=$distance prob=$prob")

            leads.add(
                LeadVehicle(
                    x = x, y = y,
                    velocity = v, acceleration = a,
                    probability = prob,
                    distance = distance,
                    timeOffset = SupercomboConstants.LEAD_T_OFFSETS.getOrElse(leadIdx) { 0f },
                    normalizedCameraX = cameraX,
                    normalizedCameraY = cameraY,
                    approxBoxWidth = approxBoxWidth,
                    approxBoxHeight = approxBoxHeight
                )
            )
        }

        return leads
    }

    private fun sigmoid(x: Float): Float {
        return (1.0f / (1.0f + Math.exp(-x.toDouble()))).toFloat()
    }

    private fun getFloatArray(tensor: OnnxTensor): FloatArray {
        val shape = tensor.info.shape
        val totalElements = shape.fold(1L) { acc, dim -> acc * dim }.toInt()
        val arr = FloatArray(totalElements)
        val buffer = tensor.floatBuffer
        buffer.get(arr)
        buffer.rewind()
        return arr
    }

    val isReady: Boolean get() = isModelLoaded && session != null

    private fun closeSession() {
        session?.close()
        session = null
        isModelLoaded = false
    }

    override fun close() {
        closeSession()
    }

    data class LeadVehicle(
        val x: Float, val y: Float,
        val velocity: Float, val acceleration: Float,
        val probability: Float, val distance: Float,
        val timeOffset: Float,
        val normalizedCameraX: Float, val normalizedCameraY: Float,
        val approxBoxWidth: Float, val approxBoxHeight: Float
    )

    data class SupercomboOutput(
        val leads: List<LeadVehicle>,
        val inferenceTimeMs: Long
    )
}
