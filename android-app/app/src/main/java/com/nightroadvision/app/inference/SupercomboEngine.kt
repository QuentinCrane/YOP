package com.nightroadvision.app.inference

import android.content.Context
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.Closeable

/**
 * ONNX Runtime engine for the openpilot supercombo driving model.
 * Parses lead vehicle detections from the flat 6409-dim output vector.
 */
class SupercomboEngine(private val context: Context) : Closeable {

    companion object {
        private const val TAG = "SupercomboEngine"
    }

    private val env = OrtEnvironment.getEnvironment()
    private val lock = Any()
    private var session: OrtSession? = null
    @Volatile private var isModelLoaded = false

    fun loadModel(assetPath: String): Boolean = synchronized(lock) {
        try {
            closeSession()
            val modelBytes = context.assets.open(assetPath).use { it.readBytes() }
            val opts = OrtSession.SessionOptions()
            session = try {
                env.createSession(modelBytes, opts)
            } finally {
                opts.close()
            }
            isModelLoaded = true

            session?.let { s ->
                val expectedInputs = setOf(
                    SupercomboConstants.IN_IMGS,
                    SupercomboConstants.IN_DESIRE,
                    SupercomboConstants.IN_TRAFFIC_CONVENTION,
                    SupercomboConstants.IN_INITIAL_STATE,
                )
                check(s.inputNames == expectedInputs) {
                    "Unsupported supercombo inputs: ${s.inputNames}"
                }
                check(SupercomboConstants.OUT_OUTPUTS in s.outputNames) {
                    "Missing supercombo output: ${SupercomboConstants.OUT_OUTPUTS}"
                }
                Log.i(TAG, "Model loaded: $assetPath")
                Log.i(TAG, "Input names: ${s.inputNames}")
                Log.i(TAG, "Output names: ${s.outputNames}")
            }
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model: $assetPath", e)
            runCatching { closeSession() }
            false
        }
    }

    fun runInference(inputs: SupercomboPreprocessor.ModelInputs): SupercomboOutput? = synchronized(lock) {
        val s = session ?: return@synchronized null
        if (!isModelLoaded) return@synchronized null

        val startTime = System.nanoTime()
        val inputTensors = mutableMapOf<String, OnnxTensor>()
        var results: OrtSession.Result? = null

        try {
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
            val inferenceResults = s.run(inputTensors)
            results = inferenceResults
            val inferenceTimeMs = (System.nanoTime() - startTime) / 1_000_000

            // Extract the single flat output
            val outputTensor = inferenceResults.get(SupercomboConstants.OUT_OUTPUTS).orElse(null)
                as? OnnxTensor
            if (outputTensor == null) {
                Log.e(TAG, "Output tensor '${SupercomboConstants.OUT_OUTPUTS}' not found")
                return@synchronized null
            }

            val outputArray = getFloatArray(outputTensor)
            Log.d(TAG, "Output size: ${outputArray.size}, first 10: ${outputArray.take(10).joinToString { "%.4f".format(it) }}")
            val leads = parseLeadOutput(outputArray)
            val pathPoints = parsePlanOutput(outputArray)

            // Note: this ONNX model does NOT expose GRU hidden state as a separate output.
            // The recurrent state flows internally through the ONNX graph.
            // We pass zeros as initial_state; the model's internal GRU handles temporal context.

            SupercomboOutput(
                leads = leads,
                pathPoints = pathPoints,
                inferenceTimeMs = inferenceTimeMs
            )

        } catch (e: Exception) {
            Log.e(TAG, "Inference failed", e)
            null
        } finally {
            runCatching { results?.close() }
            inputTensors.values.forEach { tensor -> runCatching { tensor.close() } }
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

    /**
     * Parse predicted driving path from the plan section of the output vector.
     *
     * Plan layout: 33 timesteps × 15 values × 5 MHP hypotheses (mean)
     *            + same for std + 5 weights = 4955 floats.
     *
     * Each hypothesis is a candidate future trajectory. We pick the one with
     * the highest weight and extract (x, y, z) positions at each timestep.
     */
    private fun parsePlanOutput(output: FloatArray): List<PathPoint> {
        if (output.size < SupercomboConstants.PLAN_OFFSET + SupercomboConstants.PLAN_SIZE) {
            return emptyList()
        }

        val numTimesteps = 33
        val numValues = 15
        val numHyp = 5

        // Mean block: numTimesteps * numValues * numHyp floats
        val meanSize = numTimesteps * numValues * numHyp  // 2475
        // Std block: same size
        // Weights: numHyp floats at end of plan section
        val weightsOffset = SupercomboConstants.PLAN_OFFSET + meanSize * 2

        // Find best hypothesis by weight
        var bestHyp = 0
        var bestWeight = Float.NEGATIVE_INFINITY
        for (h in 0 until numHyp) {
            val w = output[weightsOffset + h]
            if (w > bestWeight) {
                bestWeight = w
                bestHyp = h
            }
        }

        // Extract path points from best hypothesis
        val points = mutableListOf<PathPoint>()
        val hypOffset = SupercomboConstants.PLAN_OFFSET + bestHyp * numValues

        for (t in 0 until numTimesteps) {
            val base = hypOffset + t * numValues * numHyp
            if (base + 2 >= output.size) break

            val x = output[base + 0]  // forward distance (m)
            val y = output[base + 1]  // lateral offset (m)
            val z = output[base + 2]  // vertical offset (m)

            // Skip points that are essentially zero (padding)
            if (x == 0f && y == 0f) continue

            points.add(PathPoint(
                x = x, y = y, z = z,
                timestepsFromNow = t * 0.5f,  // ~20Hz model → 0.5s per step
            ))
        }

        Log.d(TAG, "Parsed ${points.size} path points from plan (best hyp=$bestHyp, weight=${"%.3f".format(bestWeight)})")
        return points
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

    val isReady: Boolean get() = synchronized(lock) { isModelLoaded && session != null }

    private fun closeSession() {
        session?.close()
        session = null
        isModelLoaded = false
    }

    override fun close() {
        synchronized(lock) { closeSession() }
    }

    data class LeadVehicle(
        val x: Float, val y: Float,
        val velocity: Float, val acceleration: Float,
        val probability: Float, val distance: Float,
        val timeOffset: Float,
        val normalizedCameraX: Float, val normalizedCameraY: Float,
        val approxBoxWidth: Float, val approxBoxHeight: Float
    )

    /**
     * A single point on the predicted driving path (vehicle frame).
     * x = forward distance (m), y = lateral offset (m).
     */
    data class PathPoint(
        val x: Float, val y: Float, val z: Float,
        val timestepsFromNow: Float,
    )

    data class SupercomboOutput(
        val leads: List<LeadVehicle>,
        val pathPoints: List<PathPoint>,
        val inferenceTimeMs: Long
    )
}
