package com.nightroadvision.app.ui.screen

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.nightroadvision.app.model.ModelManager
import com.nightroadvision.app.model.ModelQuantization
import java.util.Locale

// ──────────────────────────────────────────────────────────────────────────────
// Settings categories
// ──────────────────────────────────────────────────────────────────────────────

private enum class SettingsCategory(
    val label: String,
    val icon: ImageVector,
) {
    MODEL("模型", Icons.Filled.Memory),
    DETECTION("检测", Icons.Filled.Visibility),
    TRACKING("跟踪", Icons.Filled.TrackChanges),
    CAMERA("相机", Icons.Filled.CameraAlt),
    BACKEND("后端", Icons.Filled.DeveloperBoard),
    ALERTS("警报", Icons.Filled.NotificationsActive),
    DISPLAY("显示", Icons.Filled.Palette),
}

// ──────────────────────────────────────────────────────────────────────────────
// Full-screen settings page
// ──────────────────────────────────────────────────────────────────────────────

@Composable
fun SettingsScreen(
    settings: InferenceSettings,
    currentModelName: String,
    onSettingsChanged: (InferenceSettings) -> Unit,
    onResetSettings: () -> Unit,
    int8Available: Boolean,
    onQuantizationSelected: (ModelQuantization) -> Unit,
    onDismiss: () -> Unit,
    supercomboEnabled: Boolean,
    onSupercomboToggle: (Boolean) -> Unit,
    supercomboLatencyMs: Long,
    modifier: Modifier = Modifier,
) {
    var selectedCategory by remember { mutableStateOf(SettingsCategory.MODEL) }
    val configuration = LocalConfiguration.current
    val isLandscape = configuration.orientation == android.content.res.Configuration.ORIENTATION_LANDSCAPE
    val sidebarWidth = if (isLandscape) 200.dp else 160.dp
    val contentPadding = if (isLandscape) 32.dp else 20.dp

    Surface(
        modifier = modifier.fillMaxSize(),
        color = NightVisionColors.Background,
    ) {
        Column(modifier = Modifier.fillMaxSize()) {
            // ── Top bar ──
            TopBar(
                onDismiss = onDismiss,
                onReset = onResetSettings,
                compact = isLandscape,
            )

            HorizontalDivider(
                color = NightVisionColors.Border,
                thickness = 0.5.dp,
            )

            // ── Body: sidebar + content ──
            Row(modifier = Modifier.fillMaxSize()) {
                // Sidebar
                SettingsSidebar(
                    selected = selectedCategory,
                    onSelect = { selectedCategory = it },
                    modifier = Modifier
                        .width(sidebarWidth)
                        .fillMaxHeight()
                        .background(NightVisionColors.Surface),
                )

                VerticalDivider(
                    color = NightVisionColors.Border,
                    thickness = 0.5.dp,
                )

                // Content
                AnimatedContent(
                    targetState = selectedCategory,
                    transitionSpec = { fadeIn() togetherWith fadeOut() },
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxHeight(),
                    label = "settings_content",
                ) { category ->
                    SettingsContent(
                        category = category,
                        settings = settings,
                        currentModelName = currentModelName,
                        onSettingsChanged = onSettingsChanged,
                        int8Available = int8Available,
                        onQuantizationSelected = onQuantizationSelected,
                        supercomboEnabled = supercomboEnabled,
                        onSupercomboToggle = onSupercomboToggle,
                        supercomboLatencyMs = supercomboLatencyMs,
                        horizontalPadding = contentPadding,
                    )
                }
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Top bar
// ──────────────────────────────────────────────────────────────────────────────

@Composable
private fun TopBar(
    onDismiss: () -> Unit,
    onReset: () -> Unit,
    compact: Boolean = false,
) {
    val verticalPadding = if (compact) 6.dp else 12.dp
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(NightVisionColors.Surface)
            .statusBarsPadding()
            .padding(horizontal = 16.dp, vertical = verticalPadding),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        IconButton(onClick = onDismiss) {
            Icon(
                imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                contentDescription = "返回",
                tint = NightVisionColors.Accent,
            )
        }
        Spacer(Modifier.width(8.dp))
        Text(
            text = "设置",
            style = if (compact) MaterialTheme.typography.titleMedium
                    else MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Bold,
            color = NightVisionColors.Text,
        )
        Spacer(Modifier.weight(1f))
        TextButton(onClick = onReset) {
            Text("重置", color = NightVisionColors.TextSecondary)
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Sidebar
// ──────────────────────────────────────────────────────────────────────────────

@Composable
private fun SettingsSidebar(
    selected: SettingsCategory,
    onSelect: (SettingsCategory) -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier
            .verticalScroll(rememberScrollState())
            .padding(vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(2.dp),
    ) {
        SettingsCategory.entries.forEach { category ->
            val isSelected = category == selected
            val background = if (isSelected) {
                NightVisionColors.Accent.copy(alpha = 0.12f)
            } else {
                Color.Transparent
            }
            val textColor = if (isSelected) NightVisionColors.Accent else NightVisionColors.TextSecondary
            val iconColor = if (isSelected) NightVisionColors.Accent else NightVisionColors.TextMuted

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 8.dp)
                    .clip(RoundedCornerShape(10.dp))
                    .background(background)
                    .clickable { onSelect(category) }
                    .padding(horizontal = 14.dp, vertical = 14.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Icon(
                    imageVector = category.icon,
                    contentDescription = null,
                    tint = iconColor,
                    modifier = Modifier.size(20.dp),
                )
                Spacer(Modifier.width(12.dp))
                Text(
                    text = category.label,
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
                    color = textColor,
                )
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Content area
// ──────────────────────────────────────────────────────────────────────────────

@Composable
private fun SettingsContent(
    category: SettingsCategory,
    settings: InferenceSettings,
    currentModelName: String,
    onSettingsChanged: (InferenceSettings) -> Unit,
    int8Available: Boolean,
    onQuantizationSelected: (ModelQuantization) -> Unit,
    supercomboEnabled: Boolean,
    onSupercomboToggle: (Boolean) -> Unit,
    supercomboLatencyMs: Long,
    horizontalPadding: Dp = 20.dp,
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = horizontalPadding, vertical = 20.dp),
        verticalArrangement = Arrangement.spacedBy(24.dp),
    ) {
        when (category) {
            SettingsCategory.MODEL -> ModelSection(
                settings = settings,
                currentModelName = currentModelName,
                int8Available = int8Available,
                onQuantizationSelected = onQuantizationSelected,
            )
            SettingsCategory.DETECTION -> DetectionSection(
                settings = settings,
                onSettingsChanged = onSettingsChanged,
            )
            SettingsCategory.TRACKING -> TrackingSection(
                settings = settings,
                onSettingsChanged = onSettingsChanged,
            )
            SettingsCategory.CAMERA -> CameraSection(
                settings = settings,
                onSettingsChanged = onSettingsChanged,
            )
            SettingsCategory.BACKEND -> BackendSection(
                settings = settings,
                onSettingsChanged = onSettingsChanged,
                supercomboEnabled = supercomboEnabled,
                onSupercomboToggle = onSupercomboToggle,
                supercomboLatencyMs = supercomboLatencyMs,
            )
            SettingsCategory.ALERTS -> AlertSection(
                settings = settings,
                onSettingsChanged = onSettingsChanged,
            )
            SettingsCategory.DISPLAY -> DisplaySection(
                settings = settings,
                onSettingsChanged = onSettingsChanged,
            )
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Section helpers
// ──────────────────────────────────────────────────────────────────────────────

@Composable
private fun SectionHeader(title: String) {
    Text(
        text = title,
        style = MaterialTheme.typography.titleMedium,
        fontWeight = FontWeight.Bold,
        color = NightVisionColors.Accent,
        letterSpacing = 0.5.sp,
    )
}

@Composable
private fun SectionCard(
    title: String? = null,
    content: @Composable ColumnScope.() -> Unit,
) {
    Surface(
        shape = RoundedCornerShape(12.dp),
        color = NightVisionColors.Surface,
        border = BorderStroke(1.dp, NightVisionColors.Border),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            title?.let {
                Text(
                    text = it,
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold,
                    color = NightVisionColors.TextMuted,
                    letterSpacing = 1.sp,
                )
            }
            content()
        }
    }
}

@Composable
private fun SettingRow(
    label: String,
    description: String? = null,
    trailing: @Composable () -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = label,
                style = MaterialTheme.typography.bodyMedium,
                color = NightVisionColors.Text,
            )
            description?.let {
                Text(
                    text = it,
                    style = MaterialTheme.typography.labelSmall,
                    color = NightVisionColors.TextMuted,
                )
            }
        }
        trailing()
    }
}

@Composable
private fun ChoiceChipRow(
    options: List<Pair<String, String>>,  // (label, description)
    selectedIndex: Int,
    onSelect: (Int) -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        options.forEachIndexed { index, (label, description) ->
            val isSelected = index == selectedIndex
            Surface(
                modifier = Modifier
                    .weight(1f)
                    .clickable { onSelect(index) },
                shape = RoundedCornerShape(10.dp),
                color = if (isSelected) NightVisionColors.Accent.copy(alpha = 0.15f)
                else NightVisionColors.Background,
                border = BorderStroke(
                    1.dp,
                    if (isSelected) NightVisionColors.Accent else NightVisionColors.Border
                ),
            ) {
                Column(
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 12.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    Text(
                        text = label,
                        style = MaterialTheme.typography.labelLarge,
                        fontWeight = FontWeight.Bold,
                        color = if (isSelected) NightVisionColors.Accent
                        else NightVisionColors.TextSecondary,
                    )
                    if (description.isNotEmpty()) {
                        Spacer(Modifier.height(2.dp))
                        Text(
                            text = description,
                            style = MaterialTheme.typography.labelSmall,
                            fontSize = 10.sp,
                            color = NightVisionColors.TextMuted,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun SliderWithValue(
    label: String,
    value: Float,
    valueRange: ClosedFloatingPointRange<Float>,
    steps: Int,
    valueText: String,
    onValueChange: (Float) -> Unit,
) {
    Column {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Text(
                text = label,
                style = MaterialTheme.typography.bodyMedium,
                color = NightVisionColors.Text,
            )
            Text(
                text = valueText,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Bold,
                fontFamily = FontFamily.Monospace,
                color = NightVisionColors.Accent,
            )
        }
        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = valueRange,
            steps = steps,
            colors = SliderDefaults.colors(
                thumbColor = NightVisionColors.Accent,
                activeTrackColor = NightVisionColors.Accent,
                inactiveTrackColor = NightVisionColors.Border,
            ),
        )
    }
}

@Composable
private fun ToggleSetting(
    label: String,
    description: String? = null,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = label,
                style = MaterialTheme.typography.bodyMedium,
                color = NightVisionColors.Text,
            )
            description?.let {
                Text(
                    text = it,
                    style = MaterialTheme.typography.labelSmall,
                    color = NightVisionColors.TextMuted,
                )
            }
        }
        Switch(
            checked = checked,
            onCheckedChange = onCheckedChange,
            colors = SwitchDefaults.colors(
                checkedThumbColor = NightVisionColors.Background,
                checkedTrackColor = NightVisionColors.Accent,
                uncheckedThumbColor = NightVisionColors.TextSecondary,
                uncheckedTrackColor = NightVisionColors.Border,
            ),
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// MODEL section
// ──────────────────────────────────────────────────────────────────────────────

@Composable
private fun ModelSection(
    settings: InferenceSettings,
    currentModelName: String,
    int8Available: Boolean,
    onQuantizationSelected: (ModelQuantization) -> Unit,
) {
    SectionHeader("模型设置")

    SectionCard(title = "当前模型") {
        Text(
            text = currentModelName,
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold,
            color = NightVisionColors.Accent,
        )
        Text(
            text = "在推理设置主页可切换模型",
            style = MaterialTheme.typography.labelSmall,
            color = NightVisionColors.TextMuted,
        )
    }

    SectionCard(title = "量化模式") {
        ChoiceChipRow(
            options = listOf(
                "自动" to "保持当前",
                "FP16" to "GPU 画质",
                "INT8" to if (int8Available) "NNAPI/CPU" else "未安装",
            ),
            selectedIndex = when (settings.quantizationPreference) {
                ModelQuantization.AUTO -> 0
                ModelQuantization.FP16 -> 1
                ModelQuantization.INT8 -> 2
                else -> 0
            },
            onSelect = { index ->
                val quantization = when (index) {
                    0 -> ModelQuantization.AUTO
                    1 -> ModelQuantization.FP16
                    2 -> ModelQuantization.INT8
                    else -> ModelQuantization.AUTO
                }
                onQuantizationSelected(quantization)
            },
        )
        Text(
            text = if (int8Available) "INT8 使用校准模型，可能略微降低精度"
            else "导出 yolo26n_balanced_512x320_int8.tflite 以启用 INT8",
            style = MaterialTheme.typography.labelSmall,
            color = NightVisionColors.TextMuted,
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// DETECTION section
// ──────────────────────────────────────────────────────────────────────────────

@Composable
private fun DetectionSection(
    settings: InferenceSettings,
    onSettingsChanged: (InferenceSettings) -> Unit,
) {
    SectionHeader("检测设置")

    SectionCard(title = "检测模式") {
        ChoiceChipRow(
            options = DetectionMode.entries.map { it.label to it.description },
            selectedIndex = settings.detectionMode.ordinal,
            onSelect = { index ->
                onSettingsChanged(settings.withPreset(DetectionMode.entries[index]))
            },
        )
    }

    SectionCard(title = "置信度阈值") {
        SliderWithValue(
            label = "最低检测置信度",
            value = settings.confidenceThreshold,
            valueRange = 0.1f..0.9f,
            steps = 7,
            valueText = String.format(Locale.US, "%.2f", settings.confidenceThreshold),
            onValueChange = {
                onSettingsChanged(settings.copy(
                    confidenceThreshold = it,
                    detectionMode = DetectionMode.CUSTOM,
                ))
            },
        )
    }

    SectionCard(title = "类别灵敏度") {
        SliderWithValue(
            label = "行人 / 自行车 / 摩托车",
            value = settings.vulnerableUserConfidence,
            valueRange = 0.05f..0.60f,
            steps = 10,
            valueText = String.format(Locale.US, "%.2f", settings.vulnerableUserConfidence),
            onValueChange = {
                onSettingsChanged(settings.copy(
                    vulnerableUserConfidence = it,
                    detectionMode = DetectionMode.CUSTOM,
                ))
            },
        )
        SliderWithValue(
            label = "轿车 / 公交 / 卡车",
            value = settings.vehicleConfidence,
            valueRange = 0.05f..0.70f,
            steps = 12,
            valueText = String.format(Locale.US, "%.2f", settings.vehicleConfidence),
            onValueChange = {
                onSettingsChanged(settings.copy(
                    vehicleConfidence = it,
                    detectionMode = DetectionMode.CUSTOM,
                ))
            },
        )
        Text(
            text = "较低值可提高远距离目标召回率，但可能增加误报",
            style = MaterialTheme.typography.labelSmall,
            color = NightVisionColors.TextMuted,
        )
    }

    SectionCard(title = "IoU 阈值") {
        SliderWithValue(
            label = "重叠抑制阈值 (NMS)",
            value = settings.iouThreshold,
            valueRange = 0.1f..0.9f,
            steps = 7,
            valueText = String.format(Locale.US, "%.2f", settings.iouThreshold),
            onValueChange = {
                onSettingsChanged(settings.copy(
                    iouThreshold = it,
                    detectionMode = DetectionMode.CUSTOM,
                ))
            },
        )
    }

    SectionCard(title = "检测过滤") {
        ToggleSetting(
            label = "类别感知重叠抑制",
            description = "防止车辆框遮挡重叠的骑行者或行人",
            checked = settings.classAwareNms,
            onCheckedChange = {
                onSettingsChanged(settings.copy(
                    classAwareNms = it,
                    detectionMode = DetectionMode.CUSTOM,
                ))
            },
        )
        Text(
            text = "最大检测数",
            style = MaterialTheme.typography.bodyMedium,
            color = NightVisionColors.Text,
        )
        ChoiceChipRow(
            options = listOf(
                "30" to "省电",
                "60" to "默认",
                "100" to "高",
                "200" to "最高",
            ),
            selectedIndex = when (settings.maxDetections) {
                30 -> 0; 60 -> 1; 100 -> 2; 200 -> 3
                else -> 1
            },
            onSelect = { index ->
                val count = listOf(30, 60, 100, 200)[index]
                onSettingsChanged(settings.copy(
                    maxDetections = count,
                    detectionMode = DetectionMode.CUSTOM,
                ))
            },
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// TRACKING section
// ──────────────────────────────────────────────────────────────────────────────

@Composable
private fun TrackingSection(
    settings: InferenceSettings,
    onSettingsChanged: (InferenceSettings) -> Unit,
) {
    SectionHeader("跟踪设置")

    SectionCard(title = "目标跟踪") {
        ToggleSetting(
            label = "时序跟踪",
            description = "稳定检测框，过滤单帧闪烁",
            checked = settings.trackingEnabled,
            onCheckedChange = { onSettingsChanged(settings.copy(trackingEnabled = it)) },
        )
        SliderWithValue(
            label = "匹配 IoU",
            value = settings.trackerIouThreshold,
            valueRange = 0.05f..0.60f,
            steps = 10,
            valueText = String.format(Locale.US, "%.2f", settings.trackerIouThreshold),
            onValueChange = {
                onSettingsChanged(settings.copy(
                    trackerIouThreshold = it,
                    detectionMode = DetectionMode.CUSTOM,
                ))
            },
        )
        SliderWithValue(
            label = "框响应速度",
            value = settings.boxSmoothing,
            valueRange = 0.10f..1.0f,
            steps = 8,
            valueText = String.format(Locale.US, "%.2f", settings.boxSmoothing),
            onValueChange = {
                onSettingsChanged(settings.copy(
                    boxSmoothing = it,
                    detectionMode = DetectionMode.CUSTOM,
                ))
            },
        )
    }

    SectionCard(title = "确认与保留") {
        Text(
            text = "显示前所需帧数",
            style = MaterialTheme.typography.bodyMedium,
            color = NightVisionColors.Text,
        )
        ChoiceChipRow(
            options = listOf("1" to "最快", "2" to "默认", "3" to "稳定", "4" to "最稳"),
            selectedIndex = settings.trackerConfirmFrames - 1,
            onSelect = { index ->
                onSettingsChanged(settings.copy(
                    trackerConfirmFrames = index + 1,
                    detectionMode = DetectionMode.CUSTOM,
                ))
            },
        )
        Text(
            text = "丢失帧后跟踪保留",
            style = MaterialTheme.typography.bodyMedium,
            color = NightVisionColors.Text,
        )
        ChoiceChipRow(
            options = listOf("6" to "短", "10" to "中", "15" to "长", "24" to "很长"),
            selectedIndex = when (settings.trackerMaxMissedFrames) {
                6 -> 0; 10 -> 1; 15 -> 2; 24 -> 3
                else -> 2
            },
            onSelect = { index ->
                val count = listOf(6, 10, 15, 24)[index]
                onSettingsChanged(settings.copy(
                    trackerMaxMissedFrames = count,
                    detectionMode = DetectionMode.CUSTOM,
                ))
            },
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// CAMERA section
// ──────────────────────────────────────────────────────────────────────────────

@Composable
private fun CameraSection(
    settings: InferenceSettings,
    onSettingsChanged: (InferenceSettings) -> Unit,
) {
    SectionHeader("相机设置")

    SectionCard(title = "分析源质量") {
        ChoiceChipRow(
            options = AnalysisResolution.entries.map { it.label to it.description },
            selectedIndex = settings.analysisResolution.ordinal,
            onSelect = { index ->
                onSettingsChanged(settings.copy(
                    analysisResolution = AnalysisResolution.entries[index],
                    detectionMode = DetectionMode.CUSTOM,
                ))
            },
        )
    }

    SectionCard(title = "镜头控制") {
        SliderWithValue(
            label = "检测缩放",
            value = settings.digitalZoomRatio,
            valueRange = 1f..3f,
            steps = 7,
            valueText = String.format(Locale.US, "%.2fx", settings.digitalZoomRatio),
            onValueChange = {
                onSettingsChanged(settings.copy(
                    digitalZoomRatio = it,
                    detectionMode = DetectionMode.CUSTOM,
                ))
            },
        )
        SliderWithValue(
            label = "自动曝光补偿",
            value = settings.exposureCompensation.toFloat(),
            valueRange = -3f..3f,
            steps = 5,
            valueText = "%+d".format(settings.exposureCompensation),
            onValueChange = {
                onSettingsChanged(settings.copy(
                    exposureCompensation = it.toInt(),
                    detectionMode = DetectionMode.CUSTOM,
                ))
            },
        )
        Text(
            text = "缩放可提升远距离目标像素，但会缩小视野；1080p 预处理耗时更长",
            style = MaterialTheme.typography.labelSmall,
            color = NightVisionColors.TextMuted,
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// BACKEND section
// ──────────────────────────────────────────────────────────────────────────────

@Composable
private fun BackendSection(
    settings: InferenceSettings,
    onSettingsChanged: (InferenceSettings) -> Unit,
    supercomboEnabled: Boolean,
    onSupercomboToggle: (Boolean) -> Unit,
    supercomboLatencyMs: Long,
) {
    SectionHeader("后端设置")

    SectionCard(title = "推理后端") {
        Text(
            text = "推理代理（自动 = GPU → NNAPI → CPU 回退）",
            style = MaterialTheme.typography.labelSmall,
            color = NightVisionColors.TextMuted,
        )
        ChoiceChipRow(
            options = BackendPreference.entries.map { it.label to "" },
            selectedIndex = settings.backendPreference.ordinal,
            onSelect = { index ->
                onSettingsChanged(settings.copy(
                    backendPreference = BackendPreference.entries[index],
                ))
            },
        )
    }

    SectionCard(title = "GPU 精度") {
        Text(
            text = "FP16 在 Adreno 750 上快 2 倍；FP32 更精确",
            style = MaterialTheme.typography.labelSmall,
            color = NightVisionColors.TextMuted,
        )
        ChoiceChipRow(
            options = GpuPrecision.entries.map { it.label to "" },
            selectedIndex = settings.gpuPrecision.ordinal,
            onSelect = { index ->
                onSettingsChanged(settings.copy(
                    gpuPrecision = GpuPrecision.entries[index],
                ))
            },
        )
    }

    SectionCard(title = "CPU 回退线程") {
        Text(
            text = "CPU 和不支持的代理操作使用；线程数越多不一定越快",
            style = MaterialTheme.typography.labelSmall,
            color = NightVisionColors.TextMuted,
        )
        ChoiceChipRow(
            options = listOf("1" to "", "2" to "", "4" to "", "6" to ""),
            selectedIndex = when (settings.cpuThreads) {
                1 -> 0; 2 -> 1; 4 -> 2; 6 -> 3
                else -> 2
            },
            onSelect = { index ->
                val count = listOf(1, 2, 4, 6)[index]
                onSettingsChanged(settings.copy(cpuThreads = count))
            },
        )
    }

    SectionCard(title = "超级组合") {
        Text(
            text = "openpilot 前车检测 + 真实距离 (需要 supercombo 模型文件)",
            style = MaterialTheme.typography.labelSmall,
            color = NightVisionColors.TextMuted,
        )
        ToggleSetting(
            label = "启用 Supercombo",
            checked = supercomboEnabled,
            onCheckedChange = onSupercomboToggle,
        )
        if (supercomboEnabled && supercomboLatencyMs > 0) {
            Text(
                text = "延迟: ${supercomboLatencyMs}ms",
                style = MaterialTheme.typography.labelSmall,
                color = NightVisionColors.TextMuted,
            )
        }
    }

    SectionCard(title = "跳帧") {
        Text(
            text = "每 N 帧处理一次（值越大越省电，检测越少）",
            style = MaterialTheme.typography.labelSmall,
            color = NightVisionColors.TextMuted,
        )
        ChoiceChipRow(
            options = listOf(
                "1/1" to "每帧",
                "1/2" to "隔一帧",
                "1/3" to "隔两帧",
            ),
            selectedIndex = settings.frameSkip - 1,
            onSelect = { index ->
                onSettingsChanged(settings.copy(
                    frameSkip = index + 1,
                    detectionMode = DetectionMode.CUSTOM,
                ))
            },
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ALERTS section
// ──────────────────────────────────────────────────────────────────────────────

@Composable
private fun AlertSection(
    settings: InferenceSettings,
    onSettingsChanged: (InferenceSettings) -> Unit,
) {
    SectionHeader("警报设置")

    SectionCard(title = "骑行警报") {
        ToggleSetting(
            label = "接近目标震动提醒",
            description = "仅对已确认、进入中央注意区域的目标触发",
            checked = settings.vibrationAlertsEnabled,
            onCheckedChange = { onSettingsChanged(settings.copy(vibrationAlertsEnabled = it)) },
        )
        HorizontalDivider(
            color = NightVisionColors.Border,
            thickness = 0.5.dp,
            modifier = Modifier.padding(vertical = 4.dp),
        )
        ToggleSetting(
            label = "接近目标声音提醒",
            description = "注意/紧急/危险 三级不同警示音",
            checked = settings.soundAlertsEnabled,
            onCheckedChange = { onSettingsChanged(settings.copy(soundAlertsEnabled = it)) },
        )
    }

    SectionCard(title = "报警距离 (米)") {
        SliderWithValue(
            label = "危险 · 立即反应",
            value = settings.dangerDistanceM,
            valueRange = 2f..30f,
            steps = 27,
            valueText = String.format(Locale.US, "%.0fm", settings.dangerDistanceM),
            onValueChange = {
                onSettingsChanged(settings.copy(dangerDistanceM = it))
            },
        )
        SliderWithValue(
            label = "紧急 · 减速观察",
            value = settings.urgentDistanceM,
            valueRange = 5f..60f,
            steps = 54,
            valueText = String.format(Locale.US, "%.0fm", settings.urgentDistanceM),
            onValueChange = {
                onSettingsChanged(settings.copy(urgentDistanceM = it))
            },
        )
        SliderWithValue(
            label = "注意 · 提前预警",
            value = settings.cautionDistanceM,
            valueRange = 10f..100f,
            steps = 89,
            valueText = String.format(Locale.US, "%.0fm", settings.cautionDistanceM),
            onValueChange = {
                onSettingsChanged(settings.copy(cautionDistanceM = it))
            },
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// DISPLAY section
// ──────────────────────────────────────────────────────────────────────────────

@Composable
private fun DisplaySection(
    settings: InferenceSettings,
    onSettingsChanged: (InferenceSettings) -> Unit,
) {
    SectionHeader("显示设置")

    SectionCard(title = "HUD 信息") {
        ToggleSetting(
            label = "类别标签",
            checked = settings.showLabels,
            onCheckedChange = { onSettingsChanged(settings.copy(showLabels = it)) },
        )
        HorizontalDivider(
            color = NightVisionColors.Border,
            thickness = 0.5.dp,
            modifier = Modifier.padding(vertical = 4.dp),
        )
        ToggleSetting(
            label = "置信度百分比",
            checked = settings.showConfidence,
            onCheckedChange = { onSettingsChanged(settings.copy(showConfidence = it)) },
        )
        HorizontalDivider(
            color = NightVisionColors.Border,
            thickness = 0.5.dp,
            modifier = Modifier.padding(vertical = 4.dp),
        )
        ToggleSetting(
            label = "跟踪 ID",
            checked = settings.showTrackIds,
            onCheckedChange = { onSettingsChanged(settings.copy(showTrackIds = it)) },
        )
    }
}
