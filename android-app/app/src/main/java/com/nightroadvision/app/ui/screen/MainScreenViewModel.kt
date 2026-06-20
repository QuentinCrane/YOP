package com.nightroadvision.app.ui.screen

import android.app.Application
import android.util.Log
import androidx.camera.core.ImageProxy
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.nightroadvision.app.camera.CameraManager
import com.nightroadvision.app.inference.InferenceEngine
import com.nightroadvision.app.model.ModelManager
import com.nightroadvision.app.performance.PerformanceMonitor
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
 * Exposes detections in camera coordinate space (1280x720) and performance metrics.
 */
class MainScreenViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "MainScreenVM"
    }

    // -- Engine & manager -----------------------------------------------------

    private val inferenceEngine = InferenceEngine(application)
    private val modelManager = ModelManager(application)
    private val performanceMonitor = PerformanceMonitor(application)

    // -- Camera ---------------------------------------------------------------

    private var cameraManager: CameraManager? = null

    // -- Detections -----------------------------------------------------------

    private val _detections = MutableStateFlow<List<InferenceEngine.Detection>>(emptyList())
    val detections: StateFlow<List<InferenceEngine.Detection>> = _detections.asStateFlow()

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

    // -- FPS tracking ---------------------------------------------------------

    private var frameCount = 0L
    private var fpsWindowStartNs = System.nanoTime()

    // -- Inference guard (keep-only-latest) -----------------------------------

    private val isProcessing = AtomicBoolean(false)
    private var frameSkipCounter = 0

    // -- Initialization -------------------------------------------------------

    init {
        viewModelScope.launch(Dispatchers.Default) {
            val savedModel = modelManager.getCurrentModel()
            inferenceEngine.loadModel(savedModel.path)
            inferenceEngine.warmUp()

            _currentModelName.value = savedModel.name
            _activeDelegate.value = inferenceEngine.getActiveDelegate()
            _performanceMetrics.update {
                it.copy(backend = delegateLabel())
            }

            Log.i(TAG, "Initialized with model: ${savedModel.name}, delegate: ${inferenceEngine.getActiveDelegate()}")
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
        val manager = CameraManager(
            context = getApplication(),
            lifecycleOwner = lifecycleOwner,
            onFrameReady = { buffer, imageProxy ->
                onCameraFrame(buffer, imageProxy)
            }
        )
        cameraManager = manager
        return manager
    }

    /**
     * Called by CameraManager when a new frame is available.
     * Runs inference on the analysis thread with keep-only-latest backpressure.
     * Applies frame skip and detection mode presets from settings.
     */
    private fun onCameraFrame(inputBuffer: ByteBuffer, imageProxy: ImageProxy) {
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
            val s = _settings.value
            // Apply detection mode presets if confidence/iou are at defaults
            val confThreshold = s.confidenceThreshold
            val iouThreshold = s.iouThreshold

            val result = inferenceEngine.runInference(
                inputBuffer = inputBuffer,
                confidenceThreshold = confThreshold,
                iouThreshold = iouThreshold,
            )

            _detections.value = result.detections
            _activeDelegate.value = result.delegateUsed

            updatePerformanceMetrics(result.inferenceTimeMs, result.detections.size)
        } catch (e: Exception) {
            Log.e(TAG, "Inference failed", e)
        } finally {
            isProcessing.set(false)
            imageProxy.close()
        }
    }

    // -- Public API: settings -------------------------------------------------

    fun updateSettings(newSettings: InferenceSettings) {
        _settings.value = newSettings
    }

    // -- Public API: model switching ------------------------------------------

    fun getAvailableModels(): List<ModelManager.ModelInfo> = modelManager.getAvailableModels()

    fun switchModel(model: ModelManager.ModelInfo) {
        viewModelScope.launch(Dispatchers.Default) {
            try {
                modelManager.switchModel(model)
                inferenceEngine.loadModel(model.path)
                inferenceEngine.warmUp()
                _currentModelName.value = model.name
                _activeDelegate.value = inferenceEngine.getActiveDelegate()
                _settings.update { it.copy(selectedModelId = model.id) }
                _performanceMetrics.update { it.copy(backend = delegateLabel()) }
                Log.i(TAG, "Switched to model: ${model.name}")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to switch model: ${model.name}", e)
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
        Log.d(TAG, "Flashlight toggled: $newState")
        return newState
    }

    // -- Internals ------------------------------------------------------------

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
        inferenceEngine.close()
        Log.i(TAG, "onCleared -- resources released")
    }
}
