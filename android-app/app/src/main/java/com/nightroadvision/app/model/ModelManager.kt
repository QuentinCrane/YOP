package com.nightroadvision.app.model

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.BufferedReader
import java.io.InputStreamReader

/**
 * Manages YOLO model selection and switching.
 * Supports multiple YOLOv8 variants with different speed/accuracy tradeoffs.
 * Uses SharedPreferences for persistence and StateFlow for reactive UI updates.
 *
 * Models are hot-swappable at runtime -- call [switchModel] to change the active model,
 * and the caller (ViewModel) should invoke [InferenceEngine.loadModel] to apply the change.
 */
class ModelManager(private val context: Context) {

    companion object {
        private const val TAG = "ModelManager"
        private const val PREFS_NAME = "model_preferences"
        private const val KEY_CURRENT_MODEL = "current_model"
        private const val DEFAULT_MODEL = "models/yolov8n.tflite"

        /** Default YOLOv8 input dimensions (width x height). */
        const val DEFAULT_INPUT_WIDTH = 512
        const val DEFAULT_INPUT_HEIGHT = 320
        const val DEFAULT_NUM_CLASSES = 80 // COCO 80 classes
    }

    /**
     * Represents metadata about a YOLO model.
     *
     * @param id Short identifier (e.g. "yolov8n")
     * @param name Display name (e.g. "YOLOv8n")
     * @param path Asset path (e.g. "models/yolov8n.tflite")
     * @param size Human-readable file size (e.g. "~6 MB")
     * @param sizeBytes Actual file size in bytes (0 if unknown)
     * @param description One-line description
     * @param parameterCount Human-readable parameter count (e.g. "3.2M")
     * @param inputWidth Model input width in pixels
     * @param inputHeight Model input height in pixels
     * @param numClasses Number of output classes
     */
    data class ModelInfo(
        val id: String,
        val name: String,
        val path: String,
        val size: String,
        val sizeBytes: Long = 0L,
        val description: String,
        val parameterCount: String = "",
        val inputWidth: Int = DEFAULT_INPUT_WIDTH,
        val inputHeight: Int = DEFAULT_INPUT_HEIGHT,
        val numClasses: Int = DEFAULT_NUM_CLASSES,
    )

    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    private val allModels = listOf(
        ModelInfo(
            id = "yolo26n",
            name = "YOLO26n",
            path = "models/yolo26n_float16.tflite",
            size = "~5 MB",
            sizeBytes = 5_388_624L,
            description = "YOLO26 Nano FP16 -- custom model, fast inference with good accuracy.",
            parameterCount = "2.8M",
            inputWidth = 512,
            inputHeight = 320,
            numClasses = 80,
        ),
        ModelInfo(
            id = "yolov8n",
            name = "YOLOv8n",
            path = "models/yolov8n.tflite",
            size = "~6 MB",
            sizeBytes = 6_476_225L,
            description = "Nano -- fastest inference, lowest accuracy. Best for real-time on low-end devices.",
            parameterCount = "3.2M",
            inputWidth = 512,
            inputHeight = 320,
            numClasses = 80,
        ),
        ModelInfo(
            id = "yolov8s",
            name = "YOLOv8s",
            path = "models/yolov8s.tflite",
            size = "~22 MB",
            sizeBytes = 22_485_506L,
            description = "Small -- good balance of speed and accuracy. Recommended for most devices.",
            parameterCount = "11.2M",
            inputWidth = 512,
            inputHeight = 320,
            numClasses = 80,
        ),
        ModelInfo(
            id = "yolov8m",
            name = "YOLOv8m",
            path = "models/yolov8m.tflite",
            size = "~50 MB",
            sizeBytes = 51_963_355L,
            description = "Medium -- higher accuracy, slower inference. Best for high-end devices.",
            parameterCount = "25.9M",
            inputWidth = 512,
            inputHeight = 320,
            numClasses = 80,
        ),
    )

    // -- Class labels ---------------------------------------------------------

    /** COCO 80 class names, loaded from assets/labels.txt or hardcoded fallback. */
    val classNames: List<String> = loadClassLabels()

    // -- Reactive state -------------------------------------------------------

    private val _currentModel = MutableStateFlow(loadCurrentModel())
    val currentModel: StateFlow<ModelInfo> = _currentModel.asStateFlow()

    private val _installedModels = MutableStateFlow<List<ModelInfo>>(emptyList())
    val installedModels: StateFlow<List<ModelInfo>> = _installedModels.asStateFlow()

    init {
        refreshInstalledModels()
        Log.i(TAG, "ModelManager initialized. ${_installedModels.value.size} models installed. " +
                "Labels loaded: ${classNames.size} classes.")
    }

    // -- Public API -----------------------------------------------------------

    fun getAvailableModels(): List<ModelInfo> = _installedModels.value.ifEmpty { allModels }

    fun getInstalledModels(): List<ModelInfo> = _installedModels.value

    fun getCurrentModel(): ModelInfo = _currentModel.value

    /**
     * Switches the active model.
     * Persists the selection across app restarts.
     * The caller should also call [InferenceEngine.loadModel] to hot-swap the model.
     */
    fun switchModel(model: ModelInfo) {
        require(_installedModels.value.any { it.id == model.id }) {
            "Model ${model.id} is not installed"
        }
        prefs.edit().putString(KEY_CURRENT_MODEL, model.path).apply()
        _currentModel.value = model
        Log.i(TAG, "Switched to model: ${model.name} (${model.path})")
    }

    /**
     * Switches the active model by its id string.
     * Returns true if the switch was successful, false if the model is not installed.
     */
    fun switchModelById(modelId: String): Boolean {
        val model = allModels.find { it.id == modelId } ?: return false
        if (_installedModels.value.none { it.id == modelId }) return false
        switchModel(model)
        return true
    }

    /**
     * Returns the asset file path for the current model,
     * suitable for use with TFLite Interpreter.
     */
    fun getCurrentModelPath(): String = _currentModel.value.path

    /**
     * Scans the assets/models directory for actually present model files
     * and updates [installedModels].
     */
    fun refreshInstalledModels() {
        val assetFiles = try {
            context.assets.list("models")?.toSet().orEmpty()
        } catch (_: Exception) {
            emptySet()
        }

        val installed = allModels.filter { info -> assetFiles.contains(info.fileName()) }

        // Update installed list FIRST so switchModel's require guard passes
        _installedModels.value = installed

        // If current model is not installed, fall back to first installed.
        if (installed.none { it.id == _currentModel.value.id } && installed.isNotEmpty()) {
            switchModel(installed.first())
        }

        Log.d(TAG, "Installed models: ${installed.map { it.id }}")
    }

    /**
     * Get class name for a given class ID.
     * Returns "cls_$classId" if the ID is out of range.
     */
    fun getClassName(classId: Int): String {
        return if (classId in classNames.indices) classNames[classId] else "cls_$classId"
    }

    // -- Helpers --------------------------------------------------------------

    private fun loadCurrentModel(): ModelInfo {
        val savedPath = prefs.getString(KEY_CURRENT_MODEL, DEFAULT_MODEL) ?: DEFAULT_MODEL
        return allModels.firstOrNull { it.path == savedPath } ?: allModels.first()
    }

    /**
     * Load class labels from assets/labels.txt (one per line).
     * Falls back to hardcoded COCO 80-class list if the file is missing.
     */
    private fun loadClassLabels(): List<String> {
        return try {
            context.assets.open("labels.txt").use { stream ->
                BufferedReader(InputStreamReader(stream)).readLines()
                    .map { it.trim() }
                    .filter { it.isNotEmpty() }
            }
        } catch (_: Exception) {
            Log.w(TAG, "labels.txt not found in assets, using hardcoded COCO classes")
            cocoClassNames
        }
    }

    private fun ModelInfo.fileName(): String = path.substringAfterLast('/')

    /** COCO 80 class names as fallback. */
    private val cocoClassNames = listOf(
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    )
}
