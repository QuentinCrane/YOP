package com.nightroadvision.app.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.YuvImage
import android.hardware.camera2.CaptureRequest
import android.util.Log
import android.view.Surface
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.nightroadvision.app.FileLogger
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
    private val onFrameReady: FrameCallback,
    modelInputWidth: Int = 512,
    modelInputHeight: Int = 320,
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

    // Model input dimensions for letterboxing (updatable when model changes)
    @Volatile
    var targetModelWidth = modelInputWidth
        set(value) {
            if (field != value) {
                FileLogger.i(TAG, "targetModelWidth changed: $field -> $value")
                field = value
            }
        }
    @Volatile
    var targetModelHeight = modelInputHeight
        set(value) {
            if (field != value) {
                FileLogger.i(TAG, "targetModelHeight changed: $field -> $value")
                field = value
            }
        }

    // Actual frame dimensions from the camera (may differ from ANALYSIS_WIDTH/HEIGHT)
    @Volatile
    var actualFrameWidth = ANALYSIS_WIDTH
        private set
    @Volatile
    var actualFrameHeight = ANALYSIS_HEIGHT
        private set

    // Camera sensor rotation in degrees (0, 90, 180, 270)
    @Volatile
    var sensorRotationDegrees = 0
        private set

    // Camera exposure settings
    enum class ExposureMode { AUTO, MANUAL }
    enum class FocusMode { AUTO, MANUAL, CONTINUOUS }

    @Volatile private var exposureMode = ExposureMode.AUTO
    @Volatile private var isoValue = 800
    @Volatile private var exposureTimeNs = 10_000_000L // 10ms in nanoseconds
    @Volatile private var exposureCompensation = 0 // EV steps
    @Volatile private var focusMode = FocusMode.CONTINUOUS

    private var previewSurfaceProvider: Preview.SurfaceProvider? = null
    private var lastDisplayRotation: Int? = null

    private var frameCount = 0L

    /**
     * Bind camera preview and analysis use cases.
     */
    fun startCamera(previewSurfaceProvider: Preview.SurfaceProvider) {
        this.previewSurfaceProvider = previewSurfaceProvider
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            try {
                val provider = cameraProviderFuture.get()
                cameraProvider = provider
                bindUseCases(provider, previewSurfaceProvider)
            } catch (e: Exception) {
                Log.e(TAG, "Camera provider failed", e)
                FileLogger.e(TAG, "Camera provider failed: ${e.message}")
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

        // Get display rotation to match preview orientation
        val displayRotation = getDisplayRotation()
        lastDisplayRotation = displayRotation
        sensorRotationDegrees = rotationDegrees(displayRotation)
        FileLogger.i(TAG, "Display rotation: $displayRotation (display=$displayRotation, " +
            "sensor rotation will be updated from first frame)")

        val analysisBuilder = ImageAnalysis.Builder()
            .setTargetResolution(android.util.Size(ANALYSIS_WIDTH, ANALYSIS_HEIGHT))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .setTargetRotation(displayRotation)

        // Apply Camera2Interop exposure settings
        applyExposureSettings(analysisBuilder)

        val analysis = analysisBuilder.build()
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
            FileLogger.i(TAG, "Camera bound, target rotation: $displayRotation")
        } catch (e: Exception) {
            Log.e(TAG, "Camera bind failed", e)
            FileLogger.e(TAG, "Camera bind failed: ${e.message}", e)
        }
    }

    private fun getDisplayRotation(): Int {
        return try {
            val display = if (android.os.Build.VERSION.SDK_INT >= 30) {
                context.display
            } else {
                @Suppress("DEPRECATION")
                (context.getSystemService(Context.WINDOW_SERVICE) as? android.view.WindowManager)?.defaultDisplay
            }
            display?.rotation ?: Surface.ROTATION_0
        } catch (_: Exception) {
            Surface.ROTATION_0
        }
    }

    private fun rotationDegrees(rotation: Int): Int = when (rotation) {
        Surface.ROTATION_0 -> 0
        Surface.ROTATION_90 -> 90
        Surface.ROTATION_180 -> 180
        Surface.ROTATION_270 -> 270
        else -> 0
    }

    /** Rebind preview and analysis when the phone rotates without recreating the Activity. */
    fun refreshDisplayRotation() {
        val displayRotation = getDisplayRotation()
        if (displayRotation == lastDisplayRotation) return

        val provider = cameraProvider ?: return
        val surfaceProvider = previewSurfaceProvider ?: return
        FileLogger.i(TAG, "Display rotation changed: $lastDisplayRotation -> $displayRotation")
        bindUseCases(provider, surfaceProvider)
    }

    /**
     * Convert ImageProxy to letterboxed ByteBuffer and deliver via callback.
     * The callback receives the ImageProxy and is responsible for closing it.
     */
    private fun processFrame(imageProxy: ImageProxy) {
        try {
            val frameRotation = imageProxy.imageInfo.rotationDegrees
            val frameGeometryChanged = imageProxy.width != actualFrameWidth ||
                imageProxy.height != actualFrameHeight ||
                frameRotation != sensorRotationDegrees
            if (frameCount == 0L || frameGeometryChanged) {
                actualFrameWidth = imageProxy.width
                actualFrameHeight = imageProxy.height
                sensorRotationDegrees = frameRotation
                FileLogger.i(TAG, "Frame geometry: ${imageProxy.width}x${imageProxy.height}, " +
                    "sensorRotation: ${sensorRotationDegrees}")
            }
            frameCount++

            val buffer = yuvToLetterboxedBuffer(imageProxy)
            onFrameReady(buffer, imageProxy)
        } catch (e: Exception) {
            Log.e(TAG, "Frame processing failed", e)
            FileLogger.e(TAG, "Frame processing failed: ${e.message}", e)
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
     * Letterbox a Bitmap to model input size and return normalized float32 ByteBuffer.
     */
    private fun letterboxBitmapToBuffer(bitmap: Bitmap): ByteBuffer {
        val modelW = targetModelWidth
        val modelH = targetModelHeight

        val srcW = bitmap.width
        val srcH = bitmap.height

        if (frameCount == 1L) {
            FileLogger.i(TAG, "Letterbox: src=${srcW}x${srcH} -> target=${modelW}x${modelH}")
        }

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

    // -- Exposure control ---------------------------------------------------

    private fun applyExposureSettings(builder: ImageAnalysis.Builder) {
        val ext = Camera2Interop.Extender(builder)
        when (exposureMode) {
            ExposureMode.AUTO -> {
                ext.setCaptureRequestOption(
                    CaptureRequest.CONTROL_AE_MODE,
                    CaptureRequest.CONTROL_AE_MODE_ON
                )
                ext.setCaptureRequestOption(
                    CaptureRequest.CONTROL_AF_MODE,
                    if (focusMode == FocusMode.CONTINUOUS)
                        CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO
                    else
                        CaptureRequest.CONTROL_AF_MODE_AUTO
                )
            }
            ExposureMode.MANUAL -> {
                ext.setCaptureRequestOption(
                    CaptureRequest.CONTROL_AE_MODE,
                    CaptureRequest.CONTROL_AE_MODE_OFF
                )
                ext.setCaptureRequestOption(
                    CaptureRequest.SENSOR_SENSITIVITY,
                    isoValue
                )
                ext.setCaptureRequestOption(
                    CaptureRequest.SENSOR_EXPOSURE_TIME,
                    exposureTimeNs
                )
                ext.setCaptureRequestOption(
                    CaptureRequest.CONTROL_AF_MODE,
                    if (focusMode == FocusMode.MANUAL)
                        CaptureRequest.CONTROL_AF_MODE_OFF
                    else if (focusMode == FocusMode.CONTINUOUS)
                        CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO
                    else
                        CaptureRequest.CONTROL_AF_MODE_AUTO
                )
            }
        }
        FileLogger.d(TAG, "Exposure settings applied: mode=$exposureMode, iso=$isoValue, " +
            "shutter=${exposureTimeNs}ns, comp=$exposureCompensation, focus=$focusMode")
    }

    data class CameraExposureState(
        val exposureMode: ExposureMode = ExposureMode.AUTO,
        val isoValue: Int = 800,
        val exposureTimeNs: Long = 10_000_000L,
        val exposureCompensation: Int = 0,
        val focusMode: FocusMode = FocusMode.CONTINUOUS,
        val compensationRange: IntRange = -6..6,
    )

    fun getExposureState(): CameraExposureState {
        val range = try {
            camera?.cameraInfo?.exposureState?.exposureCompensationRange
        } catch (_: Exception) { null }
        return CameraExposureState(
            exposureMode = exposureMode,
            isoValue = isoValue,
            exposureTimeNs = exposureTimeNs,
            exposureCompensation = exposureCompensation,
            focusMode = focusMode,
            compensationRange = (range?.lower ?: -6)..(range?.upper ?: 6),
        )
    }

    /**
     * Update camera exposure settings and rebind camera.
     * This causes a brief camera restart.
     */
    fun updateExposureSettings(
        mode: ExposureMode = exposureMode,
        iso: Int = isoValue,
        shutterNs: Long = exposureTimeNs,
        compensation: Int = exposureCompensation,
        focus: FocusMode = focusMode,
    ) {
        exposureMode = mode
        isoValue = iso.coerceIn(50, 6400)
        exposureTimeNs = shutterNs.coerceIn(100_000L, 500_000_000L) // 0.1ms to 500ms
        exposureCompensation = compensation
        focusMode = focus

        // Apply exposure compensation via standard CameraX API (doesn't need rebind)
        if (mode == ExposureMode.AUTO) {
            try {
                camera?.cameraControl?.setExposureCompensationIndex(compensation)
                    ?.addListener({}, ContextCompat.getMainExecutor(context))
            } catch (e: Exception) {
                FileLogger.e(TAG, "Exposure compensation failed: ${e.message}")
            }
        }

        // Rebind camera to apply Camera2Interop changes
        val provider = cameraProvider ?: return
        val sp = previewSurfaceProvider ?: return
        bindUseCases(provider, sp)
        FileLogger.i(TAG, "Camera rebound with new exposure settings")
    }

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
