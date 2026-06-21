package com.nightroadvision.app.ui.screen

import android.app.Application
import android.graphics.RectF
import android.util.Log
import androidx.camera.core.ImageProxy
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.nightroadvision.app.FileLogger
import com.nightroadvision.app.alert.AlertSoundPlayer
import com.nightroadvision.app.alert.RidingRisk
import com.nightroadvision.app.alert.RidingRiskEvaluator
import com.nightroadvision.app.alert.RidingVibrationController
import com.nightroadvision.app.camera.CameraManager
import com.nightroadvision.app.camera.FrameGeometry
import com.nightroadvision.app.inference.InferenceEngine
import com.nightroadvision.app.inference.LeadVehicleMerger
import com.nightroadvision.app.inference.SupercomboConstants
import com.nightroadvision.app.inference.SupercomboEngine
import com.nightroadvision.app.inference.SupercomboPreprocessor
import com.nightroadvision.app.model.ModelManager
import com.nightroadvision.app.model.ModelQuantization
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
    private val settingsStore = InferenceSettingsStore(application)
    private val performanceMonitor = PerformanceMonitor(application)
    private val objectTracker = ObjectTracker()
    private val ridingRiskEvaluator = RidingRiskEvaluator()
    private val ridingVibrationController = RidingVibrationController(application)
    private val alertSoundPlayer = AlertSoundPlayer()

    // -- Supercombo (openpilot) ------------------------------------------------

    private val supercomboEngine = SupercomboEngine(application)
    private val supercomboPreprocessor = SupercomboPreprocessor()
    private val leadVehicleMerger = LeadVehicleMerger()

    private val _supercomboEnabled = MutableStateFlow(false)
    val supercomboEnabled: StateFlow<Boolean> = _supercomboEnabled.asStateFlow()

    private val _supercomboLatencyMs = MutableStateFlow(0L)
    val supercomboLatencyMs: StateFlow<Long> = _supercomboLatencyMs.asStateFlow()

    @Volatile
    private var supercomboReady = false

    // -- Camera ---------------------------------------------------------------

    @Volatile
    private var cameraManager: CameraManager? = null

    // Actual camera frame dimensions (updated when first frame arrives)
    private var cameraFrameWidth = InferenceEngine.DEFAULT_CAMERA_WIDTH
    private var cameraFrameHeight = InferenceEngine.DEFAULT_CAMERA_HEIGHT
    private var lastFrameGeometry: FrameGeometry? = null

    // -- Detections -----------------------------------------------------------

    private val _detections = MutableStateFlow<List<InferenceEngine.Detection>>(emptyList())
    val detections: StateFlow<List<InferenceEngine.Detection>> = _detections.asStateFlow()

    private val _ridingRisk = MutableStateFlow(RidingRisk())
    val ridingRisk: StateFlow<RidingRisk> = _ridingRisk.asStateFlow()

    // -- Settings (driven by UI's InferenceSettings data class) ----------------

    private val _settings = MutableStateFlow(
        settingsStore.load().copy(selectedModelId = modelManager.getCurrentModel().id)
    )
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

    // -- Camera-frame gate -----------------------------------------------------

    private var frameSkipCounter = 0
    private var errorCount = 0

    // -- Initialization -------------------------------------------------------

    init {
        FileLogger.i(TAG, "ViewModel init start")
        val initialSettings = _settings.value
        inferenceEngine.updateCpuThreadCount(initialSettings.cpuThreads)
        inferenceEngine.updateGpuConfig(
            InferenceEngine.GpuConfig(
                precisionLossAllowed = initialSettings.gpuPrecision == GpuPrecision.FP16,
            )
        )
        applyTrackerConfig(initialSettings)
        viewModelScope.launch(Dispatchers.Default) {
            try {
                val savedModel = modelManager.getCurrentModel()
                FileLogger.i(TAG, "Loading model: ${savedModel.name} (${savedModel.path})")
                inferenceEngine.loadModel(savedModel.path)

                when (initialSettings.backendPreference) {
                    BackendPreference.AUTO -> Unit
                    BackendPreference.GPU -> inferenceEngine.selectBackend(InferenceEngine.DelegateType.GPU)
                    BackendPreference.NNAPI -> inferenceEngine.selectBackend(InferenceEngine.DelegateType.NNAPI)
                    BackendPreference.CPU -> inferenceEngine.selectBackend(InferenceEngine.DelegateType.CPU)
                }

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

                    // Try loading supercombo model (optional, non-fatal)
                    try {
                        val scPath = "models/driving_supercombo.onnx"
                        supercomboReady = supercomboEngine.loadModel(scPath)
                        if (supercomboReady) {
                            FileLogger.i(TAG, "Supercombo model loaded OK")
                        } else {
                            FileLogger.w(TAG, "Supercombo model not found (optional)")
                        }
                    } catch (e: Exception) {
                        FileLogger.w(TAG, "Supercombo load skipped: ${e.message}")
                    }
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
            onFrameReady = { buffer, imageProxy, geometry ->
                onCameraFrame(buffer, imageProxy, geometry)
            },
            modelInputWidth = currentModel.inputWidth,
            modelInputHeight = currentModel.inputHeight,
            analysisWidth = _settings.value.analysisResolution.width,
            analysisHeight = _settings.value.analysisResolution.height,
            onYuvFrameReady = { nv21, width, height ->
                if (_supercomboEnabled.value && supercomboReady) {
                    supercomboPreprocessor.feedFrame(nv21, width, height)
                }
            },
            shouldAnalyzeFrame = ::shouldAnalyzeCameraFrame,
            shouldCaptureYuvFrame = { _supercomboEnabled.value && supercomboReady },
        )
        cameraManager = manager
        manager.setZoomRatio(_settings.value.digitalZoomRatio)
        manager.setExposureCompensation(_settings.value.exposureCompensation)
        FileLogger.i(TAG, "createCameraManager: model=${currentModel.name}, " +
            "input=${currentModel.inputWidth}x${currentModel.inputHeight}")
        return manager
    }

    /**
     * Called by CameraManager when a new frame is available.
     * Runs inference on the analysis thread with keep-only-latest backpressure.
     */
    private fun shouldAnalyzeCameraFrame(): Boolean {
        totalFrameCount++
        if (!inferenceEngine.isReady) return false
        frameSkipCounter++
        val interval = _settings.value.frameSkip.coerceAtLeast(1)
        if (frameSkipCounter < interval) return false
        frameSkipCounter = 0
        return true
    }

    private fun onCameraFrame(
        inputBuffer: ByteBuffer,
        imageProxy: ImageProxy,
        geometry: FrameGeometry,
    ) {
        // The model tensor and its output use the same upright coordinate space.
        if (geometry != lastFrameGeometry) {
            val dimensionsChanged = geometry.width != cameraFrameWidth ||
                geometry.height != cameraFrameHeight
            lastFrameGeometry = geometry
            cameraFrameWidth = geometry.width
            cameraFrameHeight = geometry.height
            inferenceEngine.setInputTransform(
                cameraWidth = geometry.width,
                cameraHeight = geometry.height,
                scale = geometry.modelScale,
                offsetX = geometry.modelOffsetX,
                offsetY = geometry.modelOffsetY,
            )
            if (dimensionsChanged) {
                synchronized(objectTracker) { objectTracker.reset() }
                ridingRiskEvaluator.reset()
                _detections.value = emptyList()
            }
            FileLogger.i(TAG, "Model frame geometry: ${geometry.width}x${geometry.height} @ " +
                "${geometry.rotationDegrees}°, crop=${geometry.centerCrop}")
        }
        val currentSettings = _settings.value

        try {
            if (!inferenceEngine.isReady) {
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

            val result = inferenceEngine.runInference(
                inputBuffer = inputBuffer,
                confidenceThreshold = currentSettings.confidenceThreshold,
                iouThreshold = currentSettings.iouThreshold,
                classConfidenceThresholds = classThresholds(currentSettings),
                maxDetections = currentSettings.maxDetections,
                classAwareNms = currentSettings.classAwareNms,
            )

            // Hoist model lookup outside filter to avoid per-detection getCurrentModel() call
            val currentModel = modelManager.getCurrentModel()
            val useNightRoad = currentModel.numClasses <= NIGHT_ROAD_CLASS_NAMES.size
            var roadDetections = if (useNightRoad) {
                result.detections.filter { it.classId in NIGHT_ROAD_CLASS_NAMES.indices }
            } else {
                result.detections.filter { it.classId in COCO_ROAD_USER_CLASS_IDS }
            }

            // Run supercombo if enabled and ready
            if (_supercomboEnabled.value && supercomboReady) {
                val scInputs = supercomboPreprocessor.prepareInputs()
                if (scInputs != null) {
                    val scOutput = supercomboEngine.runInference(scInputs)
                    if (scOutput != null) {
                        _supercomboLatencyMs.value = scOutput.inferenceTimeMs
                        // Merge YOLO detections with supercombo lead vehicles
                        roadDetections = leadVehicleMerger.merge(
                            roadDetections, scOutput,
                            cameraFrameWidth, cameraFrameHeight
                        )
                    }
                }
            }

            val stabilizedDetections = if (currentSettings.trackingEnabled) {
                stabilizeDetections(roadDetections)
            } else {
                roadDetections
            }
            _detections.value = stabilizedDetections
            val ridingRisk = ridingRiskEvaluator.evaluate(
                detections = stabilizedDetections,
                rotationDegrees = 0,
                cameraWidth = cameraFrameWidth,
                cameraHeight = cameraFrameHeight,
            )
            _ridingRisk.value = ridingRisk
            ridingVibrationController.update(
                risk = ridingRisk,
                enabled = currentSettings.vibrationAlertsEnabled,
            )
            alertSoundPlayer.setEnabled(currentSettings.soundAlertsEnabled)
            alertSoundPlayer.onRiskChanged(ridingRisk)
            _activeDelegate.value = result.delegateUsed
            errorCount = 0 // reset on success

            // Periodic summary log every 60 frames to avoid hot-path overhead
            if (totalFrameCount % 60 == 0L) {
                FileLogger.d(TAG, "Inference: ${result.detections.size} raw, " +
                    "${roadDetections.size} road, ${stabilizedDetections.size} tracked, " +
                    "${result.inferenceTimeMs}ms, ${result.delegateUsed}")
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
            imageProxy.close()
        }
    }

    // -- Public API: settings -------------------------------------------------

    fun updateSettings(newSettings: InferenceSettings) {
        val oldSettings = _settings.value
        val sanitized = newSettings.sanitized()
        _settings.value = sanitized
        settingsStore.save(sanitized)
        applyTrackerConfig(sanitized)
        FileLogger.d(TAG, "Settings updated: conf=${newSettings.confidenceThreshold}, " +
            "iou=${newSettings.iouThreshold}, mode=${newSettings.detectionMode}, " +
            "skip=${newSettings.frameSkip}, backend=${newSettings.backendPreference}, " +
            "gpuPrecision=${newSettings.gpuPrecision}")

        val engineConfigurationChanged =
            oldSettings.backendPreference != sanitized.backendPreference ||
                oldSettings.gpuPrecision != sanitized.gpuPrecision ||
                oldSettings.cpuThreads != sanitized.cpuThreads
        if (engineConfigurationChanged) {
            viewModelScope.launch(Dispatchers.Default) {
                try {
                    if (oldSettings.cpuThreads != sanitized.cpuThreads) {
                        inferenceEngine.updateCpuThreadCount(sanitized.cpuThreads)
                    }
                    if (oldSettings.gpuPrecision != sanitized.gpuPrecision) {
                        inferenceEngine.updateGpuConfig(
                            InferenceEngine.GpuConfig(
                                precisionLossAllowed = sanitized.gpuPrecision == GpuPrecision.FP16,
                            )
                        )
                    }
                    when (sanitized.backendPreference) {
                        BackendPreference.AUTO -> inferenceEngine.selectAutomaticBackend()
                        BackendPreference.GPU -> inferenceEngine.selectBackend(InferenceEngine.DelegateType.GPU)
                        BackendPreference.NNAPI -> inferenceEngine.selectBackend(InferenceEngine.DelegateType.NNAPI)
                        BackendPreference.CPU -> inferenceEngine.selectBackend(InferenceEngine.DelegateType.CPU)
                    }
                    _activeDelegate.value = inferenceEngine.getActiveDelegate()
                    _performanceMetrics.update { it.copy(backend = delegateLabel()) }
                    FileLogger.i(TAG, "Engine settings applied: backend=${inferenceEngine.getActiveDelegate()}, " +
                        "precision=${sanitized.gpuPrecision}, threads=${sanitized.cpuThreads}")
                } catch (e: Exception) {
                    FileLogger.e(TAG, "Engine settings failed: ${e.message}", e)
                    _errorMessage.value = "推理设置应用失败: ${e.message}"
                }
            }
        }

        cameraManager?.let { manager ->
            if (oldSettings.digitalZoomRatio != sanitized.digitalZoomRatio) {
                manager.setZoomRatio(sanitized.digitalZoomRatio)
            }
            if (oldSettings.exposureCompensation != sanitized.exposureCompensation) {
                manager.setExposureCompensation(sanitized.exposureCompensation)
            }
            if (oldSettings.analysisResolution != sanitized.analysisResolution) {
                manager.updateAnalysisResolution(
                    sanitized.analysisResolution.width,
                    sanitized.analysisResolution.height,
                )
            }
        }
    }

    fun resetSettings() {
        updateSettings(
            InferenceSettings(
                selectedModelId = modelManager.getCurrentModel().id,
                vibrationAlertsEnabled = _settings.value.vibrationAlertsEnabled,
                soundAlertsEnabled = _settings.value.soundAlertsEnabled,
            )
        )
    }

    fun clearError() {
        _errorMessage.value = null
    }

    // -- Public API: supercombo toggle -----------------------------------------

    fun setSupercomboEnabled(enabled: Boolean) {
        _supercomboEnabled.value = enabled
        if (!enabled) {
            supercomboPreprocessor.reset()
            _supercomboLatencyMs.value = 0L
        }
        FileLogger.i(TAG, "Supercombo ${if (enabled) "enabled" else "disabled"}")
    }

    // -- Public API: camera switching ------------------------------------------

    fun cycleCamera(): CameraManager.LensOption? {
        val selected = cameraManager?.cycleCamera() ?: return null
        synchronized(objectTracker) { objectTracker.reset() }
        ridingRiskEvaluator.reset()
        _ridingRisk.value = RidingRisk()
        _detections.value = emptyList()
        return selected
    }

    // -- Public API: model switching ------------------------------------------

    fun getAvailableModels(): List<ModelManager.ModelInfo> = modelManager.getAvailableModels()

    fun isInt8Available(): Boolean =
        ModelQuantization.INT8 in modelManager.getAvailableQuantizations()

    fun selectQuantization(quantization: ModelQuantization) {
        if (quantization == ModelQuantization.AUTO) {
            updateSettings(_settings.value.copy(quantizationPreference = quantization))
            return
        }
        val current = modelManager.getCurrentModel()
        val variant = modelManager.findInstalledVariant(current.family, quantization)
            ?: modelManager.getInstalledModels().firstOrNull { it.quantization == quantization }
        if (variant == null) {
            _errorMessage.value = if (quantization == ModelQuantization.INT8) {
                "INT8 模型未安装，请先导出并放入 assets/models"
            } else {
                "没有可用的 ${quantization.label} 模型"
            }
            return
        }
        updateSettings(_settings.value.copy(quantizationPreference = quantization))
        if (variant.id != current.id) switchModel(variant)
    }

    fun switchModel(model: ModelManager.ModelInfo) {
        viewModelScope.launch(Dispatchers.Default) {
            try {
                FileLogger.i(TAG, "Switching model: ${model.name} (${model.path})")
                _performanceMetrics.update { it.copy(backend = "Loading...") }

                modelManager.switchModel(model)
                inferenceEngine.loadModel(model.path)
                when (_settings.value.backendPreference) {
                    BackendPreference.AUTO -> Unit
                    BackendPreference.GPU -> inferenceEngine.selectBackend(InferenceEngine.DelegateType.GPU)
                    BackendPreference.NNAPI -> inferenceEngine.selectBackend(InferenceEngine.DelegateType.NNAPI)
                    BackendPreference.CPU -> inferenceEngine.selectBackend(InferenceEngine.DelegateType.CPU)
                }
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
                    _settings.update { current ->
                        current.copy(
                            selectedModelId = model.id,
                            quantizationPreference = if (current.quantizationPreference == ModelQuantization.AUTO) {
                                ModelQuantization.AUTO
                            } else {
                                model.quantization
                            },
                        ).also(settingsStore::save)
                    }
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

    private fun applyTrackerConfig(settings: InferenceSettings) {
        synchronized(objectTracker) {
            objectTracker.updateConfig(
                ObjectTracker.Config(
                    iouThreshold = settings.trackerIouThreshold,
                    confirmFrames = settings.trackerConfirmFrames,
                    maxMissedFrames = settings.trackerMaxMissedFrames,
                    boxSmoothing = settings.boxSmoothing,
                )
            )
        }
    }

    private fun classThresholds(settings: InferenceSettings): Map<Int, Float> {
        val model = modelManager.getCurrentModel()
        return if (model.numClasses <= NIGHT_ROAD_CLASS_NAMES.size) {
            buildMap {
                for (classId in 0..3) put(classId, settings.vulnerableUserConfidence)
                for (classId in 4..6) put(classId, settings.vehicleConfidence)
            }
        } else {
            mapOf(
                0 to settings.vulnerableUserConfidence,
                1 to settings.vulnerableUserConfidence,
                3 to settings.vulnerableUserConfidence,
                2 to settings.vehicleConfidence,
                5 to settings.vehicleConfidence,
                7 to settings.vehicleConfidence,
            )
        }
    }

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
            val className = roadClassName(track.classId)

            // Match against the tracker's latest unsmoothed box. The previous center-key
            // lookup almost never matched after EMA smoothing and silently lost distance.
            val originalIndex = detections.indices
                .asSequence()
                .filter { detections[it].classId == track.classId }
                .maxByOrNull { ObjectTracker.computeIoU(track.box, trackingInput[it].box) }
            val original = originalIndex
                ?.takeIf { ObjectTracker.computeIoU(track.box, trackingInput[it].box) >= 0.5f }
                ?.let(detections::get)

            InferenceEngine.Detection(
                x1 = box.left,
                y1 = box.top,
                x2 = box.right,
                y2 = box.bottom,
                confidence = track.confidence,
                classId = track.classId,
                className = className,
                trackId = track.id,
                cameraWidth = cameraFrameWidth,
                cameraHeight = cameraFrameHeight,
                distanceMeters = original?.distanceMeters,
                velocityMps = original?.velocityMps,
                isLeadVehicle = original?.isLeadVehicle ?: false,
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
        alertSoundPlayer.release()
        inferenceEngine.close()
        supercomboEngine.close()
        FileLogger.i(TAG, "onCleared — resources released")
    }
}
