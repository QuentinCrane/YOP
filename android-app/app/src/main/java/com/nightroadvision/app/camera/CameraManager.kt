package com.nightroadvision.app.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.YuvImage
import android.util.Log
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors

/**
 * Callback type for processed frames.
 * Receives a letterboxed float32 ByteBuffer (1 x 320 x 512 x 3) and the source ImageProxy.
 * The callback is responsible for closing the ImageProxy when done.
 */
typealias FrameCallback = (ByteBuffer, ImageProxy) -> Unit

/**
 * Manages CameraX pipeline: preview + image analysis.
 *
 * Frames are converted to letterboxed RGB float32 ByteBuffers (512x320)
 * and delivered via [onFrameReady]. The callback receives both the buffer
 * and the original ImageProxy so the caller can close it after processing.
 */
class CameraManager(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val onFrameReady: FrameCallback
) {
    companion object {
        private const val TAG = "CameraManager"
        const val ANALYSIS_WIDTH = 1280
        const val ANALYSIS_HEIGHT = 720
    }

    private var cameraProvider: ProcessCameraProvider? = null
    private var camera: Camera? = null
    @Volatile
    private var analysisExecutor = Executors.newSingleThreadExecutor()

    /**
     * Bind camera preview and analysis use cases.
     */
    fun startCamera(previewSurfaceProvider: Preview.SurfaceProvider) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            try {
                val provider = cameraProviderFuture.get()
                cameraProvider = provider
                bindUseCases(provider, previewSurfaceProvider)
            } catch (e: Exception) {
                Log.e(TAG, "Camera provider failed", e)
            }
        }, ContextCompat.getMainExecutor(context))
    }

    private fun bindUseCases(
        provider: ProcessCameraProvider,
        previewSurfaceProvider: Preview.SurfaceProvider
    ) {
        val preview = Preview.Builder()
            .setTargetResolution(android.util.Size(ANALYSIS_WIDTH, ANALYSIS_HEIGHT))
            .build()
            .also { it.setSurfaceProvider(previewSurfaceProvider) }

        val analysis = ImageAnalysis.Builder()
            .setTargetResolution(android.util.Size(ANALYSIS_WIDTH, ANALYSIS_HEIGHT))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .build()
            .also {
                it.setAnalyzer(analysisExecutor) { imageProxy ->
                    processFrame(imageProxy)
                }
            }

        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

        try {
            provider.unbindAll()
            camera = provider.bindToLifecycle(lifecycleOwner, cameraSelector, preview, analysis)
            Log.i(TAG, "Camera bound successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Camera bind failed", e)
        }
    }

    /**
     * Convert ImageProxy to letterboxed ByteBuffer and deliver via callback.
     * The callback receives the ImageProxy and is responsible for closing it.
     */
    private fun processFrame(imageProxy: ImageProxy) {
        try {
            val buffer = yuvToLetterboxedBuffer(imageProxy)
            onFrameReady(buffer, imageProxy)
        } catch (e: Exception) {
            Log.e(TAG, "Frame processing failed", e)
            imageProxy.close()
        }
    }

    /**
     * Convert YUV_420_888 ImageProxy to letterboxed 512x320 RGB float32 ByteBuffer.
     *
     * Properly handles YUV_420_888 plane layout including pixelStride and rowStride.
     * Produces NV21 format (Y + interleaved VU) for YuvImage compression.
     */
    private fun yuvToLetterboxedBuffer(imageProxy: ImageProxy): ByteBuffer {
        val yPlane = imageProxy.planes[0]
        val uPlane = imageProxy.planes[1]
        val vPlane = imageProxy.planes[2]

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        val width = imageProxy.width
        val height = imageProxy.height
        val yRowStride = yPlane.rowStride
        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride

        // NV21: Y plane (width*height) + interleaved VU (width*height/2)
        val nv21 = ByteArray(width * height + width * height / 2)

        // Copy Y plane row by row (rowStride may be > width)
        var pos = 0
        for (row in 0 until height) {
            yBuffer.position(row * yRowStride)
            yBuffer.get(nv21, pos, width)
            pos += width
        }

        // Interleave V and U bytes (NV21 = VUVUVU...)
        // Bounds-check UV buffer access for non-standard camera HALs (Samsung Exynos, MediaTek)
        val uvHeight = height / 2
        val uvWidth = width / 2
        val vBufLimit = vBuffer.limit()
        val uBufLimit = uBuffer.limit()
        for (row in 0 until uvHeight) {
            for (col in 0 until uvWidth) {
                val uvIndex = row * uvRowStride + col * uvPixelStride
                if (uvIndex < vBufLimit && uvIndex < uBufLimit) {
                    nv21[pos++] = vBuffer.get(uvIndex) // V
                    nv21[pos++] = uBuffer.get(uvIndex) // U
                } else {
                    // Pad with neutral chroma if buffer is too small
                    nv21[pos++] = 128.toByte() // V
                    nv21[pos++] = 128.toByte() // U
                }
            }
        }

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(
            android.graphics.Rect(0, 0, width, height), 90, out
        )
        val jpegBytes = out.toByteArray()

        val bitmap = android.graphics.BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
            ?: throw IllegalStateException("Failed to decode JPEG from YUV conversion")
        return letterboxBitmapToBuffer(bitmap)
    }

    /**
     * Letterbox a Bitmap to 512x320 and return normalized float32 ByteBuffer.
     */
    private fun letterboxBitmapToBuffer(bitmap: Bitmap): ByteBuffer {
        val modelW = 512
        val modelH = 320

        val srcW = bitmap.width
        val srcH = bitmap.height

        val scaleX = modelW.toFloat() / srcW
        val scaleY = modelH.toFloat() / srcH
        val scale = minOf(scaleX, scaleY)

        val scaledW = (srcW * scale).toInt()
        val scaledH = (srcH * scale).toInt()
        val padX = (modelW - scaledW) / 2
        val padY = (modelH - scaledH) / 2

        val letterboxed = Bitmap.createBitmap(modelW, modelH, Bitmap.Config.ARGB_8888)
        val scaled = Bitmap.createScaledBitmap(bitmap, scaledW, scaledH, true)
        try {
            val canvas = android.graphics.Canvas(letterboxed)
            canvas.drawColor(android.graphics.Color.rgb(114, 114, 114))
            canvas.drawBitmap(scaled, padX.toFloat(), padY.toFloat(), null)
        } finally {
            if (scaled !== bitmap) {
                scaled.recycle()
            }
            bitmap.recycle()
        }

        val buffer = ByteBuffer.allocateDirect(1 * modelH * modelW * 3 * 4)
        buffer.order(ByteOrder.nativeOrder())

        try {
            val pixels = IntArray(modelW * modelH)
            letterboxed.getPixels(pixels, 0, modelW, 0, 0, modelW, modelH)
            for (pixel in pixels) {
                buffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f) // R
                buffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)  // G
                buffer.putFloat((pixel and 0xFF) / 255.0f)           // B
            }
        } finally {
            letterboxed.recycle()
        }
        buffer.rewind()
        return buffer
    }

    /**
     * Enable or disable the camera flashlight (torch).
     */
    fun setTorchEnabled(enabled: Boolean): Boolean {
        return try {
            camera?.cameraControl?.enableTorch(enabled)
            true
        } catch (e: Exception) {
            Log.e(TAG, "Torch control failed", e)
            false
        }
    }

    fun isTorchEnabled(): Boolean = camera?.cameraInfo?.torchState?.value == 1

    fun hasFlashUnit(): Boolean = camera?.cameraInfo?.hasFlashUnit() == true

    /**
     * Release all camera resources.
     */
    fun stopCamera() {
        cameraProvider?.unbindAll()
        camera = null
    }

    /**
     * Release all resources including the executor.
     * Call this only when the CameraManager will never be used again.
     */
    fun release() {
        stopCamera()
        analysisExecutor.shutdown()
    }
}
