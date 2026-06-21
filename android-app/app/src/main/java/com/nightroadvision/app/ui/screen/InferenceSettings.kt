package com.nightroadvision.app.ui.screen

import android.content.Context
import com.nightroadvision.app.model.ModelQuantization

enum class DetectionMode(val label: String, val description: String) {
    ECO("ECO", "Low power"),
    BALANCED("BALANCED", "Daily use"),
    FINE("FINE", "More recall"),
    LONG_RANGE("LONG", "Distant targets"),
    CUSTOM("CUSTOM", "Manual tuning"),
}

enum class BackendPreference(val label: String) {
    AUTO("Auto"),
    GPU("GPU"),
    NNAPI("NNAPI"),
    CPU("CPU"),
}

enum class GpuPrecision(val label: String) {
    FP16("FP16 (Fast)"),
    FP32("FP32 (Precise)"),
}

enum class AnalysisResolution(
    val label: String,
    val width: Int,
    val height: Int,
    val description: String,
) {
    PERFORMANCE("540p", 960, 540, "Lower camera cost"),
    BALANCED("720p", 1280, 720, "Recommended"),
    DETAIL("1080p", 1920, 1080, "More source detail"),
}

data class InferenceSettings(
    val confidenceThreshold: Float = 0.25f,
    val vulnerableUserConfidence: Float = 0.18f,
    val vehicleConfidence: Float = 0.23f,
    val iouThreshold: Float = 0.45f,
    val maxDetections: Int = 60,
    val classAwareNms: Boolean = true,
    val detectionMode: DetectionMode = DetectionMode.BALANCED,
    val frameSkip: Int = 1,
    val selectedModelId: String = "yolo26n",
    val backendPreference: BackendPreference = BackendPreference.AUTO,
    val gpuPrecision: GpuPrecision = GpuPrecision.FP16,
    val quantizationPreference: ModelQuantization = ModelQuantization.AUTO,
    val cpuThreads: Int = 4,
    val trackingEnabled: Boolean = true,
    val trackerIouThreshold: Float = 0.22f,
    val trackerConfirmFrames: Int = 2,
    val trackerMaxMissedFrames: Int = 8,
    val boxSmoothing: Float = 0.60f,
    val digitalZoomRatio: Float = 1.0f,
    val exposureCompensation: Int = 0,
    val analysisResolution: AnalysisResolution = AnalysisResolution.BALANCED,
    val showLabels: Boolean = true,
    val showConfidence: Boolean = true,
    val showTrackIds: Boolean = false,
    val vibrationAlertsEnabled: Boolean = true,
    val soundAlertsEnabled: Boolean = true,
)

fun InferenceSettings.withPreset(mode: DetectionMode): InferenceSettings = when (mode) {
    DetectionMode.ECO -> copy(
        confidenceThreshold = 0.38f,
        vulnerableUserConfidence = 0.28f,
        vehicleConfidence = 0.34f,
        iouThreshold = 0.45f,
        maxDetections = 30,
        frameSkip = 3,
        cpuThreads = 2,
        trackerIouThreshold = 0.28f,
        trackerConfirmFrames = 2,
        trackerMaxMissedFrames = 6,
        boxSmoothing = 0.55f,
        digitalZoomRatio = 1.0f,
        analysisResolution = AnalysisResolution.PERFORMANCE,
        detectionMode = mode,
    )

    DetectionMode.BALANCED -> InferenceSettings(
        selectedModelId = selectedModelId,
        backendPreference = backendPreference,
        gpuPrecision = gpuPrecision,
        quantizationPreference = quantizationPreference,
        vibrationAlertsEnabled = vibrationAlertsEnabled,
        soundAlertsEnabled = soundAlertsEnabled,
        showLabels = showLabels,
        showConfidence = showConfidence,
        showTrackIds = showTrackIds,
    )

    DetectionMode.FINE -> copy(
        confidenceThreshold = 0.20f,
        vulnerableUserConfidence = 0.14f,
        vehicleConfidence = 0.18f,
        iouThreshold = 0.50f,
        maxDetections = 100,
        frameSkip = 1,
        cpuThreads = 4,
        trackerIouThreshold = 0.18f,
        trackerConfirmFrames = 2,
        trackerMaxMissedFrames = 10,
        boxSmoothing = 0.68f,
        analysisResolution = AnalysisResolution.DETAIL,
        detectionMode = mode,
    )

    DetectionMode.LONG_RANGE -> copy(
        confidenceThreshold = 0.18f,
        vulnerableUserConfidence = 0.10f,
        vehicleConfidence = 0.15f,
        iouThreshold = 0.52f,
        maxDetections = 100,
        frameSkip = 1,
        cpuThreads = 4,
        trackerIouThreshold = 0.12f,
        trackerConfirmFrames = 1,
        trackerMaxMissedFrames = 12,
        boxSmoothing = 0.72f,
        digitalZoomRatio = 1.25f,
        analysisResolution = AnalysisResolution.DETAIL,
        detectionMode = mode,
    )

    DetectionMode.CUSTOM -> copy(detectionMode = mode)
}

fun InferenceSettings.sanitized(): InferenceSettings = copy(
    confidenceThreshold = confidenceThreshold.coerceIn(0.05f, 0.90f),
    vulnerableUserConfidence = vulnerableUserConfidence.coerceIn(0.05f, 0.90f),
    vehicleConfidence = vehicleConfidence.coerceIn(0.05f, 0.90f),
    iouThreshold = iouThreshold.coerceIn(0.10f, 0.90f),
    maxDetections = maxDetections.coerceIn(10, 300),
    frameSkip = frameSkip.coerceIn(1, 6),
    cpuThreads = cpuThreads.coerceIn(1, 8),
    trackerIouThreshold = trackerIouThreshold.coerceIn(0.05f, 0.80f),
    trackerConfirmFrames = trackerConfirmFrames.coerceIn(1, 5),
    trackerMaxMissedFrames = trackerMaxMissedFrames.coerceIn(1, 30),
    boxSmoothing = boxSmoothing.coerceIn(0.05f, 1.0f),
    digitalZoomRatio = digitalZoomRatio.coerceIn(1.0f, 3.0f),
    exposureCompensation = exposureCompensation.coerceIn(-6, 6),
)

/** Persists advanced controls without tying the ViewModel to UI lifecycle state. */
class InferenceSettingsStore(context: Context) {
    private val preferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun load(): InferenceSettings {
        val defaults = InferenceSettings()
        return InferenceSettings(
            confidenceThreshold = preferences.getFloat("confidence", defaults.confidenceThreshold),
            vulnerableUserConfidence = preferences.getFloat("vulnerable_confidence", defaults.vulnerableUserConfidence),
            vehicleConfidence = preferences.getFloat("vehicle_confidence", defaults.vehicleConfidence),
            iouThreshold = preferences.getFloat("iou", defaults.iouThreshold),
            maxDetections = preferences.getInt("max_detections", defaults.maxDetections),
            classAwareNms = preferences.getBoolean("class_aware_nms", defaults.classAwareNms),
            detectionMode = enumValue("detection_mode", defaults.detectionMode),
            frameSkip = preferences.getInt("frame_skip", defaults.frameSkip),
            selectedModelId = preferences.getString("selected_model", defaults.selectedModelId) ?: defaults.selectedModelId,
            backendPreference = enumValue("backend", defaults.backendPreference),
            gpuPrecision = enumValue("gpu_precision", defaults.gpuPrecision),
            quantizationPreference = enumValue("model_quantization", defaults.quantizationPreference),
            cpuThreads = preferences.getInt("cpu_threads", defaults.cpuThreads),
            trackingEnabled = preferences.getBoolean("tracking", defaults.trackingEnabled),
            trackerIouThreshold = preferences.getFloat("tracker_iou", defaults.trackerIouThreshold),
            trackerConfirmFrames = preferences.getInt("tracker_confirm", defaults.trackerConfirmFrames),
            trackerMaxMissedFrames = preferences.getInt("tracker_missed", defaults.trackerMaxMissedFrames),
            boxSmoothing = preferences.getFloat("box_smoothing", defaults.boxSmoothing),
            digitalZoomRatio = preferences.getFloat("digital_zoom", defaults.digitalZoomRatio),
            exposureCompensation = preferences.getInt("exposure_compensation", defaults.exposureCompensation),
            analysisResolution = enumValue("analysis_resolution", defaults.analysisResolution),
            showLabels = preferences.getBoolean("show_labels", defaults.showLabels),
            showConfidence = preferences.getBoolean("show_confidence", defaults.showConfidence),
            showTrackIds = preferences.getBoolean("show_track_ids", defaults.showTrackIds),
            vibrationAlertsEnabled = preferences.getBoolean("vibration", defaults.vibrationAlertsEnabled),
            soundAlertsEnabled = preferences.getBoolean("sound", defaults.soundAlertsEnabled),
        ).sanitized()
    }

    fun save(settings: InferenceSettings) {
        val value = settings.sanitized()
        preferences.edit()
            .putFloat("confidence", value.confidenceThreshold)
            .putFloat("vulnerable_confidence", value.vulnerableUserConfidence)
            .putFloat("vehicle_confidence", value.vehicleConfidence)
            .putFloat("iou", value.iouThreshold)
            .putInt("max_detections", value.maxDetections)
            .putBoolean("class_aware_nms", value.classAwareNms)
            .putString("detection_mode", value.detectionMode.name)
            .putInt("frame_skip", value.frameSkip)
            .putString("selected_model", value.selectedModelId)
            .putString("backend", value.backendPreference.name)
            .putString("gpu_precision", value.gpuPrecision.name)
            .putString("model_quantization", value.quantizationPreference.name)
            .putInt("cpu_threads", value.cpuThreads)
            .putBoolean("tracking", value.trackingEnabled)
            .putFloat("tracker_iou", value.trackerIouThreshold)
            .putInt("tracker_confirm", value.trackerConfirmFrames)
            .putInt("tracker_missed", value.trackerMaxMissedFrames)
            .putFloat("box_smoothing", value.boxSmoothing)
            .putFloat("digital_zoom", value.digitalZoomRatio)
            .putInt("exposure_compensation", value.exposureCompensation)
            .putString("analysis_resolution", value.analysisResolution.name)
            .putBoolean("show_labels", value.showLabels)
            .putBoolean("show_confidence", value.showConfidence)
            .putBoolean("show_track_ids", value.showTrackIds)
            .putBoolean("vibration", value.vibrationAlertsEnabled)
            .putBoolean("sound", value.soundAlertsEnabled)
            .apply()
    }

    private inline fun <reified T : Enum<T>> enumValue(key: String, fallback: T): T =
        runCatching {
            enumValueOf<T>(preferences.getString(key, fallback.name) ?: fallback.name)
        }.getOrDefault(fallback)

    private companion object {
        const val PREFS_NAME = "inference_settings"
    }
}
