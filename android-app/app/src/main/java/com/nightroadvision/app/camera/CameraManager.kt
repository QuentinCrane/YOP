package com.nightroadvision.app.camera

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.util.Log
import android.view.Surface
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraInfo
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.nightroadvision.app.FileLogger
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlin.math.roundToInt

/**
 * Callback type for processed frames.
 * Receives a letterboxed float32 ByteBuffer, the source ImageProxy, and the exact
 * upright geometry used by the model.
 * The callback is responsible for closing the ImageProxy when done.
 */
typealias FrameCallback = (ByteBuffer, ImageProxy, FrameGeometry) -> Unit

/**
 * Callback type for raw NV21 YUV frames (used by supercombo preprocessor).
 * Receives NV21 byte array, width, and height.
 */
typealias YuvFrameCallback = (ByteArray, Int, Int) -> Unit

/** Geometry of the camera image after [rotationDegrees] has been applied. */
data class FrameGeometry(
    val width: Int,
    val height: Int,
    val rotationDegrees: Int,
    val modelScale: Float,
    val modelOffsetX: Float,
    val modelOffsetY: Float,
    val centerCrop: Boolean,
)

/**
 * Manages CameraX pipeline: preview + image analysis.
 *
 * Frames are converted to letterboxed RGB float32 ByteBuffers (512x320)
 * and delivered via [onFrameReady]. The callback receives both the buffer
 * and the original ImageProxy so the caller can close it after processing.
 */
@androidx.annotation.OptIn(markerClass = [ExperimentalCamera2Interop::class])
class CameraManager(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val onFrameReady: FrameCallback,
    modelInputWidth: Int = 512,
    modelInputHeight: Int = 320,
    private val onYuvFrameReady: YuvFrameCallback? = null,
    private val shouldAnalyzeFrame: () -> Boolean = { true },
    private val shouldCaptureYuvFrame: () -> Boolean = { false },
) {
    companion object {
        private const val TAG = "CameraManager"
        const val ANALYSIS_WIDTH = 1280
        const val ANALYSIS_HEIGHT = 720
    }

    /** A CameraX-selectable rear camera or a zoom preset on the logical camera. */
    data class LensOption(
        val id: String,
        val logicalCameraId: String,
        val physicalCameraId: String? = null,
        val focalLengthMm: Float? = null,
        val zoomRatio: Float? = null,
        val label: String,
        val description: String,
    )

    private val _availableLenses = MutableStateFlow<List<LensOption>>(emptyList())
    val availableLenses: StateFlow<List<LensOption>> = _availableLenses.asStateFlow()

    private val _currentLens = MutableStateFlow<LensOption?>(null)
    val currentLens: StateFlow<LensOption?> = _currentLens.asStateFlow()

    private var cameraProvider: ProcessCameraProvider? = null
    private var camera: Camera? = null
    private var previewUseCase: Preview? = null
    private var analysisUseCase: ImageAnalysis? = null
    @Volatile
    private var analysisExecutor = Executors.newSingleThreadExecutor()

    // Model input dimensions for letterboxing (updatable when model changes)
    @Volatile
    var targetModelWidth = modelInputWidth
        set(value) {
            if (field != value) {
                FileLogger.i(TAG, "targetModelWidth changed: $field -> $value")
                field = value
                cachedFloatBuffer = null  // invalidate pooled buffer
                cachedModelRow = null
                invalidatePixelMap()
            }
        }
    @Volatile
    var targetModelHeight = modelInputHeight
        set(value) {
            if (field != value) {
                FileLogger.i(TAG, "targetModelHeight changed: $field -> $value")
                field = value
                cachedFloatBuffer = null  // invalidate pooled buffer
                invalidatePixelMap()
            }
        }

    // Actual frame dimensions from the camera (may differ from ANALYSIS_WIDTH/HEIGHT)
    @Volatile
    var actualFrameWidth = ANALYSIS_WIDTH
        private set
    @Volatile
    var actualFrameHeight = ANALYSIS_HEIGHT
        private set

    // Pooled float buffer for letterboxing — reused across frames to avoid allocation
    private var cachedFloatBuffer: ByteBuffer? = null
    private var cachedBufferModelW = 0
    private var cachedBufferModelH = 0
    private var cachedModelRow: FloatArray? = null

    private data class PixelMapKey(
        val sourceWidth: Int,
        val sourceHeight: Int,
        val outputWidth: Int,
        val outputHeight: Int,
        val rotationDegrees: Int,
        val modelWidth: Int,
        val modelHeight: Int,
        val yRowStride: Int,
        val yPixelStride: Int,
        val uRowStride: Int,
        val uPixelStride: Int,
        val vRowStride: Int,
        val vPixelStride: Int,
        val modelScale: Float,
        val modelOffsetX: Float,
        val modelOffsetY: Float,
    )

    private var cachedPixelMapKey: PixelMapKey? = null
    private var cachedYOffsets: IntArray? = null
    private var cachedUOffsets: IntArray? = null
    private var cachedVOffsets: IntArray? = null

    // Pooled NV21 byte array — reused across frames to avoid 1.4MB allocation per frame
    private var cachedNv21: ByteArray? = null
    private var cachedNv21Size = 0

    @Volatile
    var frameGeometry = createFrameGeometry(ANALYSIS_WIDTH, ANALYSIS_HEIGHT, 0)
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
    private var targetRotation: Int = Surface.ROTATION_0

    private var frameCount = 0L

    /**
     * Bind camera preview and analysis use cases.
     */
    fun startCamera(
        previewSurfaceProvider: Preview.SurfaceProvider,
        targetRotation: Int = Surface.ROTATION_0,
    ) {
        this.previewSurfaceProvider = previewSurfaceProvider
        this.targetRotation = targetRotation
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            try {
                val provider = cameraProviderFuture.get()
                cameraProvider = provider
                discoverCameras(provider)
                bindUseCases(provider, previewSurfaceProvider)
            } catch (e: Exception) {
                Log.e(TAG, "Camera provider failed", e)
                FileLogger.e(TAG, "Camera provider failed: ${e.message}")
            }
        }, ContextCompat.getMainExecutor(context))
    }

    /**
     * Discover available back cameras and identify the ultra-wide lens.
     */
    private fun discoverCameras(provider: ProcessCameraProvider) {
        try {
            val camera2Manager = context.getSystemService(Context.CAMERA_SERVICE)
                as android.hardware.camera2.CameraManager
            val backInfos = provider.availableCameraInfos.filter {
                it.lensFacing == CameraSelector.LENS_FACING_BACK
            }
            if (backInfos.isEmpty()) {
                _availableLenses.value = emptyList()
                _currentLens.value = null
                return
            }

            val defaultInfo = CameraSelector.DEFAULT_BACK_CAMERA.filter(backInfos).firstOrNull()
                ?: backInfos.first()
            val defaultId = cameraId(defaultInfo)
            val defaultChars = camera2Manager.getCameraCharacteristics(defaultId)
            val defaultFocal = defaultChars
                .get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                ?.firstOrNull()

            val automatic = LensOption(
                id = "logical:$defaultId",
                logicalCameraId = defaultId,
                focalLengthMm = defaultFocal,
                label = "AUTO",
                description = "自动镜头",
            )

            val physicalIds = if (android.os.Build.VERSION.SDK_INT >= 28) {
                defaultChars.physicalCameraIds
            } else {
                emptySet()
            }
            val physicalOptions = physicalIds.mapNotNull { physicalId ->
                runCatching {
                    val chars = camera2Manager.getCameraCharacteristics(physicalId)
                    if (chars.get(CameraCharacteristics.LENS_FACING) != CameraMetadata.LENS_FACING_BACK) {
                        return@runCatching null
                    }
                    val focal = chars
                        .get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                        ?.firstOrNull()
                    LensOption(
                        id = "physical:$defaultId:$physicalId",
                        logicalCameraId = defaultId,
                        physicalCameraId = physicalId,
                        focalLengthMm = focal,
                        label = lensLabel(focal, defaultFocal),
                        description = lensDescription(focal, defaultFocal),
                    )
                }.getOrNull()
            }.filterNotNull().distinctBy { it.id }.sortedBy { it.focalLengthMm ?: Float.MAX_VALUE }

            val logicalOptions = backInfos.mapNotNull { info ->
                runCatching {
                    val id = cameraId(info)
                    val chars = camera2Manager.getCameraCharacteristics(id)
                    val focal = chars
                        .get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                        ?.firstOrNull()
                    LensOption(
                        id = "logical:$id",
                        logicalCameraId = id,
                        focalLengthMm = focal,
                        label = lensLabel(focal, defaultFocal),
                        description = lensDescription(focal, defaultFocal),
                    )
                }.getOrNull()
            }.distinctBy { it.id }.sortedBy { it.focalLengthMm ?: Float.MAX_VALUE }

            val options = if (physicalOptions.size > 1) {
                listOf(automatic) + physicalOptions
            } else {
                logicalOptions.ifEmpty { listOf(automatic) }
            }.distinctBy { it.id }

            _availableLenses.value = options
            _currentLens.value = options.firstOrNull { it.id == automatic.id } ?: options.first()
            FileLogger.i(TAG, "Rear lens options: ${options.joinToString { "${it.label}[${it.id}]" }}")
        } catch (e: Exception) {
            Log.e(TAG, "Camera discovery failed", e)
            FileLogger.e(TAG, "Camera discovery failed: ${e.message}")
        }
    }

    /** Select a concrete rear lens or logical-camera zoom preset. */
    fun switchCamera(lens: LensOption): Boolean {
        if (lens !in _availableLenses.value) return false
        val previous = _currentLens.value
        if (lens.id == previous?.id) return true

        if (lens.logicalCameraId == previous?.logicalCameraId &&
            lens.physicalCameraId == previous.physicalCameraId &&
            lens.zoomRatio != null
        ) {
            return runCatching {
                camera?.cameraControl?.setZoomRatio(lens.zoomRatio)
                _currentLens.value = lens
                true
            }.getOrElse { false }
        }

        _currentLens.value = lens
        val provider = cameraProvider ?: return false
        val surfaceProvider = previewSurfaceProvider ?: return false
        FileLogger.i(TAG, "Switching camera to ${lens.label} (${lens.id})")
        if (bindUseCases(provider, surfaceProvider)) return true

        _currentLens.value = previous
        if (previous != null) bindUseCases(provider, surfaceProvider)
        return false
    }

    /** Cycle to the next available rear camera. */
    fun cycleCamera(): LensOption? {
        val options = _availableLenses.value
        if (options.isEmpty()) return null
        val currentIndex = options.indexOfFirst { it.id == _currentLens.value?.id }
        val next = options[(currentIndex + 1) % options.size]
        return if (switchCamera(next)) next else _currentLens.value
    }

    private fun buildCameraSelector(): CameraSelector {
        val lens = _currentLens.value ?: return CameraSelector.DEFAULT_BACK_CAMERA
        val builder = CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .addCameraFilter { infos ->
                infos.filter { info -> runCatching { cameraId(info) == lens.logicalCameraId }.getOrDefault(false) }
            }
        lens.physicalCameraId?.let(builder::setPhysicalCameraId)
        return builder.build()
    }

    private fun cameraId(cameraInfo: CameraInfo): String =
        Camera2CameraInfo.from(cameraInfo).cameraId

    private fun lensLabel(focal: Float?, reference: Float?): String {
        if (focal == null || reference == null || reference <= 0f) return "CAM"
        val ratio = focal / reference
        val rounded = (ratio * 10f).roundToInt() / 10f
        return if (rounded == 1f) "1×" else "${rounded}×"
    }

    private fun lensDescription(focal: Float?, reference: Float?): String {
        if (focal == null || reference == null || reference <= 0f) return "后置镜头"
        val ratio = focal / reference
        return when {
            ratio < 0.85f -> "超广角"
            ratio > 1.35f -> "长焦"
            else -> "主摄"
        }
    }

    private fun bindUseCases(
        provider: ProcessCameraProvider,
        previewSurfaceProvider: Preview.SurfaceProvider
    ): Boolean {
        val resolutionSelector = ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
            .setResolutionStrategy(
                ResolutionStrategy(
                    android.util.Size(ANALYSIS_WIDTH, ANALYSIS_HEIGHT),
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
                )
            )
            .build()
        val preview = Preview.Builder()
            .setResolutionSelector(resolutionSelector)
            .setTargetRotation(targetRotation)
            .build()
            .also { it.setSurfaceProvider(previewSurfaceProvider) }

        val analysisBuilder = ImageAnalysis.Builder()
            .setResolutionSelector(resolutionSelector)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .setTargetRotation(targetRotation)

        // Apply Camera2Interop exposure settings
        applyExposureSettings(analysisBuilder)

        val analysis = analysisBuilder.build()
            .also {
                it.setAnalyzer(analysisExecutor) { imageProxy ->
                    processFrame(imageProxy)
                }
            }

        // Select camera based on current lens preference
        val cameraSelector = buildCameraSelector()

        try {
            provider.unbindAll()
            camera = provider.bindToLifecycle(lifecycleOwner, cameraSelector, preview, analysis)
            previewUseCase = preview
            analysisUseCase = analysis
            applySelectedZoom()
            addZoomPresetsIfNeeded()
            Log.i(TAG, "Camera bound successfully (lens=${_currentLens.value?.label})")
            FileLogger.i(TAG, "Camera bound, lens=${_currentLens.value?.id}, target rotation: $targetRotation")
            return true
        } catch (e: Exception) {
            Log.e(TAG, "Camera bind failed", e)
            FileLogger.e(TAG, "Camera bind failed: ${e.message}", e)
            previewUseCase = null
            analysisUseCase = null
            return false
        }
    }

    /** Update orientation without tearing down the camera session. */
    fun updateTargetRotation(rotation: Int) {
        if (rotation == targetRotation) return
        targetRotation = rotation
        previewUseCase?.targetRotation = rotation
        analysisUseCase?.targetRotation = rotation
        FileLogger.i(TAG, "Target rotation updated: $rotation")
    }

    private fun applySelectedZoom() {
        val ratio = _currentLens.value?.zoomRatio ?: 1f
        runCatching { camera?.cameraControl?.setZoomRatio(ratio) }
    }

    /**
     * Some OEMs expose a multi-camera only as one logical CameraX camera. In that
     * case zoom presets still let CameraX/OEM camera policy switch physical lenses.
     */
    private fun addZoomPresetsIfNeeded() {
        val options = _availableLenses.value
        if (options.size > 1) return
        val base = options.firstOrNull() ?: return
        val zoomState = camera?.cameraInfo?.zoomState?.value ?: return
        val ratios = buildList {
            if (zoomState.minZoomRatio < 0.95f) add(zoomState.minZoomRatio)
            add(1f)
            if (zoomState.maxZoomRatio >= 2f) add(2f)
            if (zoomState.maxZoomRatio >= 3f) add(3f)
        }.distinct().filter { it in zoomState.minZoomRatio..zoomState.maxZoomRatio }
        if (ratios.size <= 1) return

        val presets = ratios.map { ratio ->
            val rounded = (ratio * 10f).roundToInt() / 10f
            base.copy(
                id = "${base.id}:zoom:$rounded",
                zoomRatio = ratio,
                label = if (rounded == 1f) "1×" else "${rounded}×",
                description = when {
                    ratio < 0.85f -> "超广角"
                    ratio > 1.35f -> "变焦"
                    else -> "主摄"
                },
            )
        }
        _availableLenses.value = presets
        _currentLens.value = presets.minByOrNull { kotlin.math.abs((it.zoomRatio ?: 1f) - 1f) }
    }

    /**
     * Convert ImageProxy to letterboxed ByteBuffer and deliver via callback.
     * The callback receives the ImageProxy and is responsible for closing it.
     */
    private fun processFrame(imageProxy: ImageProxy) {
        try {
            val rawW = imageProxy.width
            val rawH = imageProxy.height
            val rotation = normalizeRotation(imageProxy.imageInfo.rotationDegrees)
            val swapsDimensions = rotation == 90 || rotation == 270
            val outputW = if (swapsDimensions) rawH else rawW
            val outputH = if (swapsDimensions) rawW else rawH
            val geometry = createFrameGeometry(outputW, outputH, rotation)
            val frameGeometryChanged = geometry != frameGeometry
            if (frameCount == 0L || frameGeometryChanged) {
                frameGeometry = geometry
                actualFrameWidth = outputW
                actualFrameHeight = outputH
                invalidatePixelMap()
                FileLogger.i(TAG, "Frame geometry: ${rawW}x${rawH} -> ${outputW}x${outputH} @ ${rotation}°, " +
                    "${if (geometry.centerCrop) "center-crop" else "letterbox"}, " +
                    "scale=${geometry.modelScale}, offset=${geometry.modelOffsetX},${geometry.modelOffsetY}")
            }
            frameCount++

            // Gate before any full-frame copy or color conversion.
            if (!shouldAnalyzeFrame()) {
                imageProxy.close()
                return
            }

            if (shouldCaptureYuvFrame()) {
                val nv21 = yuvToNv21(imageProxy)
                onYuvFrameReady?.invoke(nv21, rawW, rawH)
            }

            val buffer = yuvToLetterboxedBuffer(imageProxy, geometry)
            onFrameReady(buffer, imageProxy, geometry)
        } catch (e: Exception) {
            Log.e(TAG, "Frame processing failed", e)
            FileLogger.e(TAG, "Frame processing failed: ${e.message}", e)
            imageProxy.close()
        }
    }

    /**
     * Convert YUV_420_888 ImageProxy directly to letterboxed float32 ByteBuffer.
     *
     * Optimized pipeline: NV21 extraction → direct YUV-to-float32 with letterboxing.
     * Skips the expensive JPEG encode/decode round-trip that the old pipeline used.
     */
    private fun yuvToNv21(imageProxy: ImageProxy): ByteArray {
        val yPlane = imageProxy.planes[0]
        val uPlane = imageProxy.planes[1]
        val vPlane = imageProxy.planes[2]

        // Work on duplicates so optional NV21 export never changes the buffers used by YOLO.
        val yBuffer = yPlane.buffer.duplicate()
        val uBuffer = uPlane.buffer.duplicate()
        val vBuffer = vPlane.buffer.duplicate()

        val width = imageProxy.width
        val height = imageProxy.height
        val yRowStride = yPlane.rowStride
        val yPixelStride = yPlane.pixelStride
        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride
        val yBase = yBuffer.position()
        val uBase = uBuffer.position()
        val vBase = vBuffer.position()

        // Build NV21 byte array for supercombo preprocessor (pooled)
        val nv21Size = width * height * 3 / 2
        val nv21 = if (cachedNv21 != null && cachedNv21Size == nv21Size) {
            cachedNv21!!
        } else {
            ByteArray(nv21Size).also { cachedNv21 = it; cachedNv21Size = nv21Size }
        }
        var pos = 0
        // Bulk-copy Y plane when stride equals width (common case)
        if (yRowStride == width && yPixelStride == 1 && yBuffer.remaining() >= width * height) {
            yBuffer.get(nv21, 0, width * height)
            pos = width * height
        } else {
            for (row in 0 until height) {
                for (col in 0 until width) {
                    nv21[pos++] = yBuffer.get(yBase + row * yRowStride + col * yPixelStride)
                }
            }
        }
        val uvHeight = height / 2
        val uvWidth = width / 2
        val vBufLimit = vBuffer.limit()
        val uBufLimit = uBuffer.limit()
        for (row in 0 until uvHeight) {
            for (col in 0 until uvWidth) {
                val uvIndex = row * uvRowStride + col * uvPixelStride
                if (vBase + uvIndex < vBufLimit && uBase + uvIndex < uBufLimit) {
                    nv21[pos++] = vBuffer.get(vBase + uvIndex)
                    nv21[pos++] = uBuffer.get(uBase + uvIndex)
                } else {
                    nv21[pos++] = 128.toByte()
                    nv21[pos++] = 128.toByte()
                }
            }
        }

        return nv21
    }

    /** Directly sample YUV planes into the model tensor; no full-frame NV21 copy. */
    private fun yuvToLetterboxedBuffer(
        imageProxy: ImageProxy,
        geometry: FrameGeometry,
    ): ByteBuffer {
        val srcW = imageProxy.width
        val srcH = imageProxy.height
        val modelW = targetModelWidth
        val modelH = targetModelHeight
        val planes = imageProxy.planes
        val yPlane = planes[0]
        val uPlane = planes[1]
        val vPlane = planes[2]
        val yBuffer = yPlane.buffer.duplicate()
        val uBuffer = uPlane.buffer.duplicate()
        val vBuffer = vPlane.buffer.duplicate()

        val mapKey = PixelMapKey(
            sourceWidth = srcW,
            sourceHeight = srcH,
            outputWidth = geometry.width,
            outputHeight = geometry.height,
            rotationDegrees = geometry.rotationDegrees,
            modelWidth = modelW,
            modelHeight = modelH,
            yRowStride = yPlane.rowStride,
            yPixelStride = yPlane.pixelStride,
            uRowStride = uPlane.rowStride,
            uPixelStride = uPlane.pixelStride,
            vRowStride = vPlane.rowStride,
            vPixelStride = vPlane.pixelStride,
            modelScale = geometry.modelScale,
            modelOffsetX = geometry.modelOffsetX,
            modelOffsetY = geometry.modelOffsetY,
        )
        ensurePixelMap(mapKey)
        val yOffsets = checkNotNull(cachedYOffsets)
        val uOffsets = checkNotNull(cachedUOffsets)
        val vOffsets = checkNotNull(cachedVOffsets)
        val yBase = yBuffer.position()
        val uBase = uBuffer.position()
        val vBase = vBuffer.position()

        // Reuse pooled buffer if dimensions match, otherwise allocate new one
        val requiredSize = modelH * modelW * 3 * 4
        val buffer = if (cachedFloatBuffer != null &&
            cachedBufferModelW == modelW && cachedBufferModelH == modelH) {
            cachedFloatBuffer!!
        } else {
            ByteBuffer.allocateDirect(requiredSize).also {
                cachedFloatBuffer = it
                cachedBufferModelW = modelW
                cachedBufferModelH = modelH
            }
        }
        buffer.clear()
        buffer.order(ByteOrder.nativeOrder())

        val floatBuf = buffer.asFloatBuffer()
        val row = cachedModelRow?.takeIf { it.size == modelW * 3 }
            ?: FloatArray(modelW * 3).also { cachedModelRow = it }
        val padPixel = 0.447f
        var pixelIndex = 0
        for (my in 0 until modelH) {
            var ri = 0
            repeat(modelW) {
                val yOffset = yOffsets[pixelIndex]
                if (yOffset < 0) {
                    row[ri++] = padPixel
                    row[ri++] = padPixel
                    row[ri++] = padPixel
                } else {
                    val y = yBuffer.get(yBase + yOffset).toInt() and 0xFF
                    val uIndex = (uBase + uOffsets[pixelIndex]).coerceAtMost(uBuffer.limit() - 1)
                    val vIndex = (vBase + vOffsets[pixelIndex]).coerceAtMost(vBuffer.limit() - 1)
                    val u = (uBuffer.get(uIndex).toInt() and 0xFF) - 128
                    val v = (vBuffer.get(vIndex).toInt() and 0xFF) - 128
                    val rv = y + (v * 1436 shr 10)
                    val gv = y - ((u * 352 + v * 731) shr 10)
                    val bv = y + (u * 1815 shr 10)
                    row[ri++] = rv.coerceIn(0, 255) / 255.0f
                    row[ri++] = gv.coerceIn(0, 255) / 255.0f
                    row[ri++] = bv.coerceIn(0, 255) / 255.0f
                }
                pixelIndex++
            }
            floatBuf.put(row, 0, ri)
        }
        buffer.rewind()
        return buffer
    }

    private fun ensurePixelMap(key: PixelMapKey) {
        if (cachedPixelMapKey == key) return
        val count = key.modelWidth * key.modelHeight
        val yOffsets = IntArray(count) { -1 }
        val uOffsets = IntArray(count) { -1 }
        val vOffsets = IntArray(count) { -1 }
        var index = 0
        for (my in 0 until key.modelHeight) {
            for (mx in 0 until key.modelWidth) {
                val ex = kotlin.math.floor((mx - key.modelOffsetX) / key.modelScale).toInt()
                val ey = kotlin.math.floor((my - key.modelOffsetY) / key.modelScale).toInt()
                if (ex in 0 until key.outputWidth && ey in 0 until key.outputHeight) {
                    var sx = ex
                    var sy = ey
                    when (key.rotationDegrees) {
                        90 -> {
                            sx = ey
                            sy = key.sourceHeight - 1 - ex
                        }
                        180 -> {
                            sx = key.sourceWidth - 1 - ex
                            sy = key.sourceHeight - 1 - ey
                        }
                        270 -> {
                            sx = key.sourceWidth - 1 - ey
                            sy = ex
                        }
                    }
                    yOffsets[index] = sy * key.yRowStride + sx * key.yPixelStride
                    uOffsets[index] = (sy / 2) * key.uRowStride + (sx / 2) * key.uPixelStride
                    vOffsets[index] = (sy / 2) * key.vRowStride + (sx / 2) * key.vPixelStride
                }
                index++
            }
        }
        cachedPixelMapKey = key
        cachedYOffsets = yOffsets
        cachedUOffsets = uOffsets
        cachedVOffsets = vOffsets
    }

    private fun normalizeRotation(degrees: Int): Int =
        ((degrees % 360) + 360) % 360

    private fun createFrameGeometry(
        outputWidth: Int,
        outputHeight: Int,
        rotationDegrees: Int,
    ): FrameGeometry {
        val modelW = targetModelWidth.coerceAtLeast(1)
        val modelH = targetModelHeight.coerceAtLeast(1)
        val width = outputWidth.coerceAtLeast(1)
        val height = outputHeight.coerceAtLeast(1)
        val scaleX = modelW.toFloat() / width
        val scaleY = modelH.toFloat() / height
        // A portrait frame letterboxed into a landscape tensor wastes most tensor pixels.
        // Crop the central road region so target scale stays comparable to landscape mode.
        val centerCrop = height > width && modelW > modelH
        val scale = if (centerCrop) maxOf(scaleX, scaleY) else minOf(scaleX, scaleY)
        val offsetX = (modelW - width * scale) / 2f
        val offsetY = (modelH - height * scale) / 2f
        return FrameGeometry(
            width = width,
            height = height,
            rotationDegrees = normalizeRotation(rotationDegrees),
            modelScale = scale,
            modelOffsetX = offsetX,
            modelOffsetY = offsetY,
            centerCrop = centerCrop,
        )
    }

    private fun invalidatePixelMap() {
        cachedPixelMapKey = null
        cachedYOffsets = null
        cachedUOffsets = null
        cachedVOffsets = null
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
        previewUseCase = null
        analysisUseCase = null
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
