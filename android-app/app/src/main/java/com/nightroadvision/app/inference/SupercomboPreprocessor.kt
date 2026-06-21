package com.nightroadvision.app.inference

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Converts camera YUV frames into the openpilot supercombo model input format.
 *
 * Model input "input_imgs" shape: (1, 12, 128, 256)
 * - 2 consecutive frames, each with 6 channels (Y, U, V × 2)
 * - Normalized to [0, 1]
 *
 * Auxiliary inputs:
 * - desire: (1, 8) zeros (no lane change intent)
 * - traffic_convention: (1, 2) [0, 1] for right-hand traffic
 * - initial_state: (1, 512) GRU hidden state (zeros — model handles internally)
 */
class SupercomboPreprocessor {

    // Ring buffer for 2 frames of YUV data
    private var frame0: ByteArray? = null
    private var frame1: ByteArray? = null
    private var frameWidth = 0
    private var frameHeight = 0

    // Hidden state from previous inference
    private var hiddenState = FloatArray(SupercomboConstants.FEATURE_LEN)

    // Pre-allocated buffers
    private var imgBuffer: ByteBuffer? = null
    private val desireBuffer = createDesireBuffer()
    private val trafficBuffer = createTrafficBuffer()

    /**
     * Feed a raw NV21 frame from the camera.
     * Call this for every camera frame before [prepareInputs].
     */
    @Synchronized
    fun feedFrame(nv21Data: ByteArray, width: Int, height: Int) {
        if (frameWidth != width || frameHeight != height) {
            frameWidth = width
            frameHeight = height
            frame0 = null
            frame1 = null
            imgBuffer = null
        }

        // Shift frames: old frame0 becomes frame1, new data becomes frame0
        frame1 = frame0
        frame0 = nv21Data.copyOf()

        // Allocate img buffer once dimensions are known
        if (imgBuffer == null) {
            val size = 1 * SupercomboConstants.STACKED_CHANNELS *
                    SupercomboConstants.MODEL_HEIGHT * SupercomboConstants.MODEL_WIDTH * 4
            imgBuffer = ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder())
        }
    }

    /**
     * Prepare all model inputs from the buffered frames.
     * Returns null if not enough frames have been buffered yet.
     */
    @Synchronized
    fun prepareInputs(): ModelInputs? {
        val f0 = frame0 ?: return null
        // If only one frame available, use it for both
        val f1 = frame1 ?: f0

        val img = prepareImgBuffer(f0, f1) ?: return null

        return ModelInputs(
            img = img,
            desire = desireBuffer,
            trafficConvention = trafficBuffer,
            prevFeatures = createHiddenStateBuffer()
        )
    }

    /**
     * Update the hidden state from model output after inference.
     */
    @Synchronized
    fun updateHiddenState(newState: FloatArray) {
        if (newState.size == SupercomboConstants.FEATURE_LEN) {
            System.arraycopy(newState, 0, hiddenState, 0, SupercomboConstants.FEATURE_LEN)
        }
    }

    /**
     * Reset all internal state (e.g., when camera restarts).
     */
    @Synchronized
    fun reset() {
        frame0 = null
        frame1 = null
        hiddenState = FloatArray(SupercomboConstants.FEATURE_LEN)
    }

    private fun prepareImgBuffer(f0: ByteArray, f1: ByteArray): ByteBuffer? {
        val buf = imgBuffer ?: return null
        buf.rewind()

        // Process both frames: each produces 6 channels (Y full-res, U half-res, V half-res)
        writeFrameChannels(buf, f0)
        writeFrameChannels(buf, f1)

        buf.rewind()
        return buf
    }

    /**
     * Write 6 channels for one NV21 frame into the buffer.
     *
     * The model input shape is (1, 12, 128, 256) — 2 frames × 6 channels.
     * Each channel is a full 128×256 plane (32768 floats).
     *
     * The 6 channels per frame are Y, U, V each written twice:
     *   Ch 0: Y  (full res 128×256, nearest-neighbor from source)
     *   Ch 1: U  (half res 64×128, upsampled to 128×256)
     *   Ch 2: V  (half res 64×128, upsampled to 128×256)
     *   Ch 3: Y  (duplicate of ch 0)
     *   Ch 4: U  (duplicate of ch 1)
     *   Ch 5: V  (duplicate of ch 2)
     */
    private fun writeFrameChannels(buf: ByteBuffer, nv21: ByteArray) {
        val w = frameWidth
        val h = frameHeight
        val targetW = SupercomboConstants.MODEL_WIDTH   // 256
        val targetH = SupercomboConstants.MODEL_HEIGHT  // 128
        val ySize = w * h

        // NV21 layout: Y plane (w*h bytes), then interleaved VU (w*h/2 bytes)

        // Channel 0: Y plane (full res)
        writeResizedPlane(buf, nv21, 0, w, h, targetW, targetH)

        // Channel 1: U plane (from NV21 VU interleaved, upsampled to full res)
        writeResizedChromaPlane(buf, nv21, ySize, w, h, targetW, targetH, takeV = false)

        // Channel 2: V plane (from NV21 VU interleaved, upsampled to full res)
        writeResizedChromaPlane(buf, nv21, ySize, w, h, targetW, targetH, takeV = true)

        // Channels 3-5: duplicate Y, U, V
        writeResizedPlane(buf, nv21, 0, w, h, targetW, targetH)
        writeResizedChromaPlane(buf, nv21, ySize, w, h, targetW, targetH, takeV = false)
        writeResizedChromaPlane(buf, nv21, ySize, w, h, targetW, targetH, takeV = true)
    }

    /**
     * Write Y plane: resize from (srcW x srcH) to (targetW x targetH), normalize to [0,1].
     * Uses nearest-neighbor for speed.
     */
    private fun writeResizedPlane(
        buf: ByteBuffer, data: ByteArray, offset: Int,
        srcW: Int, srcH: Int, targetW: Int, targetH: Int
    ) {
        val scaleX = srcW.toFloat() / targetW
        val scaleY = srcH.toFloat() / targetH

        for (ty in 0 until targetH) {
            val sy = (ty * scaleY).toInt().coerceIn(0, srcH - 1)
            for (tx in 0 until targetW) {
                val sx = (tx * scaleX).toInt().coerceIn(0, srcW - 1)
                val value = (data[offset + sy * srcW + sx].toInt() and 0xFF) / 255.0f
                buf.putFloat(value)
            }
        }
    }

    /**
     * Write chroma plane: resize from half-res NV21 interleaved VU to (targetW x targetH).
     * NV21 after Y plane: V0 U0 V1 U1 V2 U2 ...
     * takeV=true: extract V; takeV=false: extract U
     */
    private fun writeResizedChromaPlane(
        buf: ByteBuffer, data: ByteArray, vuOffset: Int,
        srcW: Int, srcH: Int, targetW: Int, targetH: Int,
        takeV: Boolean
    ) {
        val chromaW = srcW / 2
        val chromaH = srcH / 2
        val scaleX = chromaW.toFloat() / targetW
        val scaleY = chromaH.toFloat() / targetH
        val byteOffset = if (takeV) 0 else 1

        for (ty in 0 until targetH) {
            val sy = (ty * scaleY).toInt().coerceIn(0, chromaH - 1)
            for (tx in 0 until targetW) {
                val sx = (tx * scaleX).toInt().coerceIn(0, chromaW - 1)
                val idx = vuOffset + (sy * chromaW + sx) * 2 + byteOffset
                val value = (data[idx].toInt() and 0xFF) / 255.0f
                buf.putFloat(value)
            }
        }
    }

    private fun createDesireBuffer(): ByteBuffer {
        val buf = ByteBuffer.allocateDirect(
            1 * SupercomboConstants.DESIRE_LEN * 4
        ).order(ByteOrder.nativeOrder())
        // All zeros = no lane change intent
        for (i in 0 until SupercomboConstants.DESIRE_LEN) {
            buf.putFloat(0f)
        }
        buf.rewind()
        return buf
    }

    private fun createTrafficBuffer(): ByteBuffer {
        val buf = ByteBuffer.allocateDirect(
            1 * SupercomboConstants.TRAFFIC_CONVENTION_LEN * 4
        ).order(ByteOrder.nativeOrder())
        // [0, 1] = right-hand traffic (China, most countries)
        buf.putFloat(0f)
        buf.putFloat(1f)
        buf.rewind()
        return buf
    }

    private fun createHiddenStateBuffer(): ByteBuffer {
        val buf = ByteBuffer.allocateDirect(
            1 * SupercomboConstants.FEATURE_LEN * 4
        ).order(ByteOrder.nativeOrder())
        for (v in hiddenState) {
            buf.putFloat(v)
        }
        buf.rewind()
        return buf
    }

    data class ModelInputs(
        val img: ByteBuffer,
        val desire: ByteBuffer,
        val trafficConvention: ByteBuffer,
        val prevFeatures: ByteBuffer
    )
}
