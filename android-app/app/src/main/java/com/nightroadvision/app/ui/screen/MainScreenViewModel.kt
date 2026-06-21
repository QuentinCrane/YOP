package com.nightroadvision.app.ui.screen

import android.app.Application
import android.graphics.RectF
import android.util.Log
import androidx.camera.core.ImageProxy
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.nightroadvision.app.FileLogger
import com.nightroadvision.app.alert.RidingRisk
import com.nightroadvision.app.alert.RidingRiskEvaluator
import com.nightroadvision.app.alert.RidingVibrationController
import com.nightroadvision.app.camera.CameraManager
import com.nightroadvision.app.inference.InferenceEngine
import com.nightroadvision.app.model.ModelManager
import com.nightroadvision.app.performance.PerformanceMonitor
import com.nightroadvision.app.tracking.Detection as TrackingDetection
import com.nightroadvision.app.tracking.ObjectTracker
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicBoolean

/**
 * ViewModel for MainScreen.
 *
 * Manages the full detection pipeline: CameraX frames -> InferenceEngine -> UI state.
 * Exposes detections in camera coordinate space and performance metrics.
 */
class MainScreenViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "MainScreenVM"
        private val COCO_ROAD_USER_CLASS_IDS = setOf(0, 1, 2, 3, 5, 7)
        private val NIGHT_ROAD_CLASS_NAMES = listOf(
            "person", "rider", "bicycle", "motorcycle", "car", "bus", "truck",
        )
    }

    // -- Engine & manager -----------------------------------------------------

    private val inferenceEngine = InferenceEngine(application)
    private val modelManager = ModelManager(application)
    private val performanceMonitor = PerformanceMonitor(application)
    private val objectTracker = ObjectTracker()
    private val ridingRiskEvaluator = RidingRiskEvaluator()
    private val ridingVibrationController = RidingVibrationController(application)

    // -- Camera ---------------------------------------------------------------

    @Volatile
    private var cameraManager: CameraManager? = null

    // Actual camera frame dimensions (updated when first frame arrives)
    private var cameraFrameWidth = InferenceEngine.DEFAULT_CAMERA_WIDTH
    private var cameraFrameHeight = InferenceEngine.DEFAULT_CAMERA_HEIGHT

    // -- Detections -----------------------------------------------------------

    private val _detections = MutableStateFlow<List<InferenceEngine.Detection>>(emptyList())
    val detections: StateFlow<List<InferenceEngine.Detection>> = _detections.asStateFlow()

    private val _ridingRisk = MutableStateFlow(RidingRisk())
    val ridingRisk: StateFlow<RidingRisk> = _ridingRisk.asStateFlow()

    // -- Settings (driven by UI's InferenceSettings data class) ----------------

    private val _settings = MutableStateFlow(InferenceSettings())
    val settings: StateFlow<InferenceSettings> = _settings.asStateFlow()

    // -- Current model name ---------------------------------------------------

    private val _currentModelName = MutableStateFlow(modelManager.getCurrentModel().name)
    val currentModelName: StateFlow<String> = _currentModelName.asStateFlow()

    // -- Performance metrics --------------------------------------------------

    private val _performanceMetrics = MutableStateFlow(
        PerformanceMetrics(backend = "Initializing...")
    )
    val performanceMetrics: StateFlow<PerformanceMetrics> = _performanceMetrics.asStateFlow()

    // -- Active delegate type -------------------------------------------------

    private val _activeDelegate = MutableStateFlow(InferenceEngine.DelegateType.CPU)
    val activeDelegate: StateFlow<InferenceEngine.DelegateType> = _activeDelegate.asStateFlow()

    // -- Error state ----------------------------------------------------------

    private val _errorMessage = MutableStateFlow<String?>(null)
    val errorMessage: StateFlow<String?> = _errorMessage.asStateFlow()

    // -- FPS tracking ---------------------------------------------------------

    private var frameCount = 0L
    private var fpsWindowStartNs = System.nanoTime()
    private var totalFrameCount = 0L

    // -- Inference guard (keep-only-latest) -----------------------------------

    private val isProcessing = AtomicBoolean(false)
    private var frameSkipCounter = 0
    private var errorCount = 0

    // -- Initialization -------------------------------------------------------

    init {
        FileLogger.i(TAG, "ViewModel init start")
        viewModelScope.launch(Dispatchers.Default) {
            try {
                val savedModel = modelManager.getCurrentModel()
                FileLogger.i(TAG, "Loading model: ${savedModel.name} (${savedModel.path})")
                inferenceEngine.loadModel(savedModel.path)

                if (inferenceEngine.isReady) {
                    inferenceEngine.warmUp()
                    _currentModelName.value = savedModel.name
                    _activeDelegate.value = inferenceEngine.getActiveDelegate()
                    _performanceMetrics.update {
                        it.copy(backend = delegateLabel())
                    }

                    // Update CameraManager with actual model dimensions
                    cameraManager?.let { cm ->
                        cm.targetModelWidth = inferenceEngine.modelInputWidth
                        cm.targetModelHeight = inferenceEngine.modelInputHeight
                        FileLogger.i(TAG, "CameraManager updated: target=${inferenceEngine.modelInputWidth}x${inferenceEngine.modelInputHeight}")
                    }

                    FileLogger.i(TAG, "Model loaded OK: ${savedModel.name}, " +
                        "input=${inferenceEngine.modelInputWidth}x${inferenceEngine.modelInputHeight}, " +
                        "isRawFormat=${inferenceEngine.isRawFormat}, " +
                        "delegate: ${inferenceEngine.getActiveDelegate()}")
                } else {
                    FileLogger.e(TAG, "Model load failed: ${savedModel.name} — interpreter is null")
                    _errorMessage.value = "模型加载失败: ${savedModel.name}"
                    _performanceMetrics.update { it.copy(backend = "ERROR") }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Init failed", e)
                FileLogger.e(TAG, "Init failed: ${e.message}", e)
                _errorMessage.value = "初始化失败: ${e.message}"
                _performanceMetrics.update { it.copy(backend = "ERROR") }
            }
        }
    }

    // -- Camera integration ---------------------------------------------------

    /**
     * Create a CameraManager for the given lifecycle owner.
     * The camera callback feeds frames into the inference pipeline.
     */
    fun createCameraManager(
        lifecycleOwner: androidx.lifecycle.LifecycleOwner
    ): CameraManager {
        val currentModel = modelManager.getCurrentModel()
        val manager = CameraManager(
            context = getApplication(),
            lifecycleOwner = lifecycleOwner,
            onFrameReady = { buffer, imageProxy ->
                onCameraFrame(buffer, imageProxy)
            },
            modelInputWidth = currentModel.inputWidth,
            modelInputHeight = currentModel.inputHeight,
        )
        cameraManager = manager
        FileLogger.i(TAG, "createCameraManager: model=${currentModel.name}, " +
            "input=${currentModel.inputWidth}x${currentModel.inputHeight}")
        return manager
    }

    /**
     * Called by CameraManager when a new frame is available.
     * Runs inference on the analysis thread with keep-only-latest backpressure.
     */
    private fun onCameraFrame(inputBuffer: ByteBuffer, imageProxy: ImageProxy) {
        // Keep inference geometry synchronized when orientation or camera output changes.
        if (imageProxy.width != cameraFrameWidth || imageProxy.height != cameraFrameHeight) {
            cameraFrameWidth = imageProxy.width
            cameraFrameHeight = imageProxy.height
            inferenceEngine.setActualCameraDimensions(imageProxy.width, imageProxy.height)
            FileLogger.i(TAG, "Camera frame: ${imageProxy.width}x${imageProxy.height}")
        }
        totalFrameCount++

        // Frame skipping based on settings
        frameSkipCounter++
        val skipInterval = _settings.value.frameSkip
        if (frameSkipCounter < skipInterval) {
            imageProxy.close()
            return
        }
        frameSkipCounter = 0

        // Keep-only-latest: if inference is running, drop this frame
        if (!isProcessing.compareAndSet(false, true)) {
            imageProxy.close()
            return
        }

        try {
            if (!inferenceEngine.isReady) {
                // Interpreter not ready — skip this frame silently
                isProcessing.set(false)
                imageProxy.close()
                return
            }

            // Defensive: ensure CameraManager letterboxes to correct model dimensions
            val cm = cameraManager
            if (cm != null && (cm.targetModelWidth != inferenceEngine.modelInputWidth ||
                                cm.targetModelHeight != inferenceEngine.modelInputHeight)) {
                cm.targetModelWidth = inferenceEngine.modelInputWidth
                cm.targetModelHeight = inferenceEngine.modelInputHeight
                FileLogger.w(TAG, "Force-synced CameraManager dimensions: " +
                    "${inferenceEngine.modelInputWidth}x${inferenceEngine.modelInputHeight}")
            }

            val s = _settings.value
            FileLogger.d(TAG, "Running inference: buffer=${inputBuffer.remaining()} bytes, " +
                "conf=${s.confidenceThreshold}, iou=${s.iouThreshold}, " +
                "modelInput=${inferenceEngine.modelInputWidth}x${inferenceEngine.modelInputHeight}, " +
                "camera=${inferenceEngine.actualCameraWidth}x${inferenceEngine.actualCameraHeight}, " +
                "isRaw=${inferenceEngine.isRawFormat}")

            val result = inferenceEngine.runInference(
                inputBuffer = inputBuffer,
                confidenceThreshold = s.confidenceThreshold,
                iouThreshold = s.iouThreshold,
            )

            val roadDetections = result.detections.filter(::isRoadUserDetection)
            val stabilizedDetections = stabilizeDetections(roadDetections)
            _detections.value = stabilizedDetections
            val ridingRisk = ridingRiskEvaluator.evaluate(
                detections = stabilizedDetections,
                rotationDegrees = getSensorRotationDegrees(),
                cameraWidth = getCameraFrameWidth(),
                cameraHeight = getCameraFrameHeight(),
            )
            _ridingRisk.value = ridingRisk
            ridingVibrationController.update(
                risk = ridingRisk,
                enabled = s.vibrationAlertsEnabled,
            )
            _activeDelegate.value = result.delegateUsed
            errorCount = 0 // reset on success

            FileLogger.d(TAG, "Inference result: ${result.detections.size} raw, " +
                "${roadDetections.size} road-user, " +
                "${stabilizedDetections.size} tracked, " +
                "time=${result.inferenceTimeMs}ms, delegate=${result.delegateUsed}")

            if (stabilizedDetections.isNotEmpty()) {
                val d = stabilizedDetections[0]
                FileLogger.d(TAG, "First detection: " +
                    "cam=(${d.x1.toInt()},${d.y1.toInt()})-(${d.x2.toInt()},${d.y2.toInt()}) " +
                    "conf=${d.confidence} cls=${d.className}")
            }

            updatePerformanceMetrics(result.inferenceTimeMs, stabilizedDetections.size)
        } catch (e: Exception) {
            errorCount++
            if (errorCount <= 5 || errorCount % 30 == 0) {
                // Log first 5 errors and then every 30th to avoid log spam
                FileLogger.e(TAG, "Inference failed (count=$errorCount): ${e.message}", e)
            }
            if (errorCount == 5) {
                _errorMessage.value = "推理失败: ${e.message}"
            }
        } finally {
            isProcessing.set(false)
            imageProxy.close()
        }
    }

    // -- Public API: settings -------------------------------------------------

    fun updateSettings(newSettings: InferenceSettings) {
        val oldSettings = _settings.value
        _settings.value = newSettings
        FileLogger.d(TAG, "Settings updated: conf=${newSettings.confidenceThreshold}, " +
            "iou=${newSettings.iouThreshold}, mode=${newSettings.detectionMode}, " +
            "skip=${newSettings.frameSkip}, backend=${newSettings.backendPreference}, " +
            "gpuPrecision=${newSettings.gpuPrecision}")

        // Apply backend preference change
        if (oldSettings.backendPreference != newSettings.backendPreference) {
            viewModelScope.launch(Dispatchers.Default) {
                try {
                    val delegate = when (newSettings.backendPreference) {
                        BackendPreference.AUTO -> null // let engine pick
                        BackendPreference.GPU -> InferenceEngine.DelegateType.GPU
                        BackendPreference.NNAPI -> InferenceEngine.DelegateType.NNAPI
                        BackendPreference.CPU -> InferenceEngine.DelegateType.CPU
                    }
                    if (delegate != null) {
                        inferenceEngine.selectBackend(delegate)
                    }
                    _activeDelegate.value = inferenceEngine.getActiveDelegate()
                    _performanceMetrics.update { it.copy(backend = delegateLabel()) }
                    FileLogger.i(TAG, "Backend applied: ${inferenceEngine.getActiveDelegate()}")
                } catch (e: Exception) {
                    FileLogger.e(TAG, "Backend switch failed: ${e.message}", e)
                    _errorMessage.value = "后端切换失败: ${e.message}"
                }
            }
        }

        // Apply GPU precision change
        if (oldSettings.gpuPrecision != newSettings.gpuPrecision) {
            viewModelScope.launch(Dispatchers.Default) {
                try {
                    val gpuConfig = InferenceEngine.GpuConfig(
                        precisionLossAllowed = newSettings.gpuPrecision == GpuPrecision.FP16
                    )
                    inferenceEngine.updateGpuConfig(gpuConfig)
                    FileLogger.i(TAG, "GPU precision applied: ${newSettings.gpuPrecision}")
                } catch (e: Exception) {
                    FileLogger.e(TAG, "GPU config update failed: ${e.message}", e)
                }
            }
        }
    }

    fun clearError() {
        _errorMessage.value = null
    }

    // -- Public API: model switching ------------------------------------------

    fun getAvailableModels(): List<ModelManager.ModelInfo> = modelManager.getAvailableModels()

    fun switchModel(model: ModelManager.ModelInfo) {
        viewModelScope.launch(Dispatchers.Default) {
            try {
                FileLogger.i(TAG, "Switching model: ${model.name} (${model.path})")
                _performanceMetrics.update { it.copy(backend = "Loading...") }

                modelManager.switchModel(model)
                inferenceEngine.loadModel(model.path)
                synchronized(objectTracker) {
                    objectTracker.reset()
                }
                ridingRiskEvaluator.reset()
                ridingVibrationController.reset()
                _ridingRisk.value = RidingRisk()
                _detections.value = emptyList()

                if (inferenceEngine.isReady) {
                    inferenceEngine.warmUp()
                    _currentModelName.value = model.name
                    _activeDelegate.value = inferenceEngine.getActiveDelegate()
                    _settings.update { it.copy(selectedModelId = model.id) }
                    _performanceMetrics.update { it.copy(backend = delegateLabel()) }
                    _errorMessage.value = null
                    errorCount = 0

                    // Update CameraManager letterboxing to match new model dimensions
                    cameraManager?.let { cm ->
                        cm.targetModelWidth = inferenceEngine.modelInputWidth
                        cm.targetModelHeight = inferenceEngine.modelInputHeight
                        FileLogger.i(TAG, "CameraManager updated: target=${inferenceEngine.modelInputWidth}x${inferenceEngine.modelInputHeight}")
                    }

                    FileLogger.i(TAG, "Model switched OK: ${model.name}, " +
                        "input=${inferenceEngine.modelInputWidth}x${inferenceEngine.modelInputHeight}, " +
                        "delegate: ${inferenceEngine.getActiveDelegate()}")
                } else {
                    FileLogger.e(TAG, "Model switch failed: ${model.name} — interpreter is null after loadModel")
                    _errorMessage.value = "模型加载失败: ${model.name}"
                    _performanceMetrics.update { it.copy(backend = "ERROR") }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to switch model: ${model.name}", e)
                FileLogger.e(TAG, "Failed to switch model: ${model.name}: ${e.message}", e)
                _errorMessage.value = "模型切换失败: ${e.message}"
                _performanceMetrics.update { it.copy(backend = "ERROR") }
            }
        }
    }

    // -- Public API: backend selection ----------------------------------------

    fun selectBackend(delegate: InferenceEngine.DelegateType) {
        viewModelScope.launch(Dispatchers.Default) {
            try {
                FileLogger.i(TAG, "Selecting backend: $delegate")
                inferenceEngine.selectBackend(delegate)
                _activeDelegate.value = inferenceEngine.getActiveDelegate()
                _performanceMetrics.update { it.copy(backend = delegateLabel()) }
                FileLogger.i(TAG, "Backend selected: ${inferenceEngine.getActiveDelegate()}")
            } catch (e: Exception) {
                FileLogger.e(TAG, "Backend selection failed: ${e.message}", e)
                _errorMessage.value = "后端切换失败: ${e.message}"
            }
        }
    }

    // -- Public API: torch control --------------------------------------------

    /**
     * Toggle flashlight on/off. Returns the new torch state (true=on, false=off).
     */
    fun toggleFlashlight(): Boolean {
        val currentlyOn = cameraManager?.isTorchEnabled() == true
        val newState = !currentlyOn
        cameraManager?.setTorchEnabled(newState)
        FileLogger.d(TAG, "Flashlight toggled: $newState")
        return newState
    }

    // -- Public API: camera dimensions ----------------------------------------

    fun getCameraFrameWidth(): Int = cameraManager?.actualFrameWidth ?: cameraFrameWidth
    fun getCameraFrameHeight(): Int = cameraManager?.actualFrameHeight ?: cameraFrameHeight
    fun getSensorRotationDegrees(): Int = cameraManager?.sensorRotationDegrees ?: 90

    // -- Public API: camera exposure ----------------------------------------

    fun getExposureState(): CameraManager.CameraExposureState =
        cameraManager?.getExposureState() ?: CameraManager.CameraExposureState()

    fun updateExposureSettings(
        mode: CameraManager.ExposureMode,
        iso: Int,
        shutterNs: Long,
        compensation: Int,
        focus: CameraManager.FocusMode,
    ) {
        cameraManager?.updateExposureSettings(mode, iso, shutterNs, compensation, focus)
        FileLogger.d(TAG, "Exposure updated: mode=$mode, iso=$iso, shutter=${shutterNs}ns, comp=$compensation, focus=$focus")
    }

    // -- Internals ------------------------------------------------------------

    private fun isRoadUserDetection(detection: InferenceEngine.Detection): Boolean {
        val model = modelManager.getCurrentModel()
        return if (model.numClasses <= NIGHT_ROAD_CLASS_NAMES.size) {
            detection.classId in NIGHT_ROAD_CLASS_NAMES.indices
        } else {
            detection.classId in COCO_ROAD_USER_CLASS_IDS
        }
    }

    private fun roadClassName(classId: Int): String {
        val model = modelManager.getCurrentModel()
        return if (model.numClasses <= NIGHT_ROAD_CLASS_NAMES.size) {
            NIGHT_ROAD_CLASS_NAMES.getOrElse(classId) { "target" }
        } else {
            InferenceEngine.CLASS_NAMES.getOrElse(classId) { "target" }
        }
    }

    /**
     * Confirm detections across frames and smooth their coordinates before drawing.
     * This suppresses one-frame giant boxes and reduces HUD jitter while driving.
     */
    private fun stabilizeDetections(
        detections: List<InferenceEngine.Detection>,
    ): List<InferenceEngine.Detection> {
        val trackingInput = detections.map { detection ->
            TrackingDetection(
                classId = detection.classId,
                confidence = detection.confidence,
                box = RectF(detection.x1, detection.y1, detection.x2, detection.y2),
            )
        }

        val tracks = synchronized(objectTracker) {
            objectTracker.update(trackingInput)
        }

        return tracks.map { track ->
            val box = track.smoothedBox
            InferenceEngine.Detection(
                x1 = box.left,
                y1 = box.top,
                x2 = box.right,
                y2 = box.bottom,
                confidence = track.confidence,
                classId = track.classId,
                className = roadClassName(track.classId),
                trackId = track.id,
                cameraWidth = cameraFrameWidth,
                cameraHeight = cameraFrameHeight,
            )
        }
    }

    private fun updatePerformanceMetrics(inferenceTimeMs: Long, detectionCount: Int) {
        frameCount++
        val now = System.nanoTime()
        val elapsedNs = now - fpsWindowStartNs

        if (elapsedNs >= 1_000_000_000L) {
            val fps = frameCount * 1_000_000_000f / elapsedNs
            _performanceMetrics.update {
                it.copy(
                    fps = fps,
                    inferenceLatencyMs = inferenceTimeMs.toFloat(),
                    detectionCount = detectionCount,
                    backend = delegateLabel(),
                    thermalStatus = performanceMonitor.getThermalStatusLabel(),
                )
            }
            frameCount = 0
            fpsWindowStartNs = now
        } else {
            _performanceMetrics.update {
                it.copy(
                    inferenceLatencyMs = inferenceTimeMs.toFloat(),
                    detectionCount = detectionCount,
                )
            }
        }
    }

    private fun delegateLabel(): String {
        return when (inferenceEngine.getActiveDelegate()) {
            InferenceEngine.DelegateType.GPU -> "GPU (Adreno)"
            InferenceEngine.DelegateType.NNAPI -> "NNAPI (Hexagon)"
            InferenceEngine.DelegateType.CPU -> "CPU (XNNPACK)"
        }
    }

    override fun onCleared() {
        super.onCleared()
        cameraManager?.release()
        ridingVibrationController.reset()
        inferenceEngine.close()
        FileLogger.i(TAG, "onCleared — resources released")
    }
}
