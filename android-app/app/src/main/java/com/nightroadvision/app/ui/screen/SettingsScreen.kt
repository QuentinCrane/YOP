package com.nightroadvision.app.ui.screen

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.rememberScrollState
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
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.nightroadvision.app.model.ModelQuantization
import java.util.Locale

// ──────────────────────────────────────────────────────────────────────────────
// Settings categories
// ──────────────────────────────────────────────────────────────────────────────

private enum class SettingsCategory(
    val label: String,
    val icon: ImageVector,
) {
    ALERTS("警报", Icons.Filled.NotificationsActive),
    DETECTION("检测与跟踪", Icons.Filled.Visibility),
    CAMERA("相机与显示", Icons.Filled.CameraAlt),
    MODEL("模型与后端", Icons.Filled.Memory),
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
    onSelectModel: () -> Unit,
    int8Available: Boolean,
    onQuantizationSelected: (ModelQuantization) -> Unit,
    onDismiss: () -> Unit,
    supercomboEnabled: Boolean,
    supercomboAvailable: Boolean,
    onSupercomboToggle: (Boolean) -> Unit,
    supercomboLatencyMs: Long,
    onPreviewAlert: (com.nightroadvision.app.alert.RiskSeverity) -> Unit = {},
    modifier: Modifier = Modifier,
) {
    var selectedCategory by remember { mutableStateOf(SettingsCategory.ALERTS) }
    val configuration = LocalConfiguration.current
    val isLandscape = configuration.orientation == android.content.res.Configuration.ORIENTATION_LANDSCAPE
    val contentPadding = if (isLandscape) 32.dp else 16.dp

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

            if (isLandscape) {
                // Landscape: sidebar + content side by side
                Row(modifier = Modifier.fillMaxSize()) {
                    SettingsSidebar(
                        selected = selectedCategory,
                        onSelect = { selectedCategory = it },
                        modifier = Modifier
                            .width(200.dp)
                            .fillMaxHeight()
                            .background(NightVisionColors.Surface),
                    )

                    VerticalDivider(
                        color = NightVisionColors.Border,
                        thickness = 0.5.dp,
                    )

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
                            onSelectModel = onSelectModel,
                            int8Available = int8Available,
                            onQuantizationSelected = onQuantizationSelected,
                            supercomboEnabled = supercomboEnabled,
                            supercomboAvailable = supercomboAvailable,
                            onSupercomboToggle = onSupercomboToggle,
                            supercomboLatencyMs = supercomboLatencyMs,
                            onPreviewAlert = onPreviewAlert,
                            horizontalPadding = contentPadding,
                        )
                    }
                }
            } else {
                // Portrait: horizontal tab bar + content stacked vertically
                SettingsTabBar(
                    selected = selectedCategory,
                    onSelect = { selectedCategory = it },
                )

                HorizontalDivider(
                    color = NightVisionColors.Border,
                    thickness = 0.5.dp,
                )

                AnimatedContent(
                    targetState = selectedCategory,
                    transitionSpec = { fadeIn() togetherWith fadeOut() },
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth(),
                    label = "settings_content",
                ) { category ->
                    SettingsContent(
                        category = category,
                        settings = settings,
                        currentModelName = currentModelName,
                        onSettingsChanged = onSettingsChanged,
                        onSelectModel = onSelectModel,
                        int8Available = int8Available,
                        onQuantizationSelected = onQuantizationSelected,
                        supercomboEnabled = supercomboEnabled,
                        supercomboAvailable = supercomboAvailable,
                        onSupercomboToggle = onSupercomboToggle,
                        supercomboLatencyMs = supercomboLatencyMs,
                        onPreviewAlert = onPreviewAlert,
                        horizontalPadding = contentPadding,
                    )
                }
            }
        }
    }
}

@Composable
private fun SettingsTabBar(
    selected: SettingsCategory,
    onSelect: (SettingsCategory) -> Unit,
) {
    val scrollState = rememberScrollState()

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(NightVisionColors.Surface)
            .horizontalScroll(scrollState)
            .padding(horizontal = 8.dp, vertical = 6.dp),
        horizontalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        SettingsCategory.entries.forEach { category ->
            val isSelected = category == selected
            val background = if (isSelected) {
                NightVisionColors.Accent.copy(alpha = 0.14f)
            } else {
                Color.Transparent
            }
            val textColor = if (isSelected) NightVisionColors.Accent else NightVisionColors.TextSecondary
            val iconColor = if (isSelected) NightVisionColors.Accent else NightVisionColors.TextMuted

            Row(
                modifier = Modifier
                    .clip(RoundedCornerShape(8.dp))
                    .background(background)
                    .clickable { onSelect(category) }
                    .padding(horizontal = 10.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Icon(
                    imageVector = category.icon,
                    contentDescription = null,
                    tint = iconColor,
                    modifier = Modifier.size(16.dp),
                )
                Spacer(Modifier.width(6.dp))
                Text(
                    text = category.label,
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
                    color = textColor,
                )
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
                    .padding(horizontal = 14.dp, vertical = 16.dp),
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
    onSelectModel: () -> Unit,
    int8Available: Boolean,
    onQuantizationSelected: (ModelQuantization) -> Unit,
    supercomboEnabled: Boolean,
    supercomboAvailable: Boolean,
    onSupercomboToggle: (Boolean) -> Unit,
    supercomboLatencyMs: Long,
    onPreviewAlert: (com.nightroadvision.app.alert.RiskSeverity) -> Unit = {},
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
            SettingsCategory.MODEL -> ModelBackendSection(
                settings = settings,
                currentModelName = currentModelName,
                onSelectModel = onSelectModel,
                int8Available = int8Available,
                onQuantizationSelected = onQuantizationSelected,
                onSettingsChanged = onSettingsChanged,
                supercomboEnabled = supercomboEnabled,
                supercomboAvailable = supercomboAvailable,
                onSupercomboToggle = onSupercomboToggle,
                supercomboLatencyMs = supercomboLatencyMs,
            )
            SettingsCategory.DETECTION -> DetectionTrackingSection(
                settings = settings,
                onSettingsChanged = onSettingsChanged,
            )
            SettingsCategory.CAMERA -> CameraDisplaySection(
                settings = settings,
                onSettingsChanged = onSettingsChanged,
            )
            SettingsCategory.ALERTS -> AlertSection(
                settings = settings,
                onSettingsChanged = onSettingsChanged,
                onPreviewAlert = onPreviewAlert,
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
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 14.dp),
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
    enabled: Boolean = true,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = label,
                style = MaterialTheme.typography.bodyMedium,
                color = if (enabled) NightVisionColors.Text else NightVisionColors.TextMuted,
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
            enabled = enabled,
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
// MODEL + BACKEND merged section
// ──────────────────────────────────────────────────────────────────────────────

@Composable
private fun ModelBackendSection(
    settings: InferenceSettings,
    currentModelName: String,
    onSelectModel: () -> Unit,
    int8Available: Boolean,
    onQuantizationSelected: (ModelQuantization) -> Unit,
    onSettingsChanged: (InferenceSettings) -> Unit,
    supercomboEnabled: Boolean,
    supercomboAvailable: Boolean,
    onSupercomboToggle: (Boolean) -> Unit,
    supercomboLatencyMs: Long,
) {
    SectionHeader("模型与后端")

    SectionCard(title = "当前模型") {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = currentModelName,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = NightVisionColors.Accent,
                )
                Text(
                    text = "仅显示已打包到应用的模型",
                    style = MaterialTheme.typography.labelSmall,
                    color = NightVisionColors.TextMuted,
                )
            }
            TextButton(onClick = onSelectModel) {
                Text("切换模型", color = NightVisionColors.Accent)
            }
        }
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
            text = if (supercomboAvailable) {
                "openpilot 前车检测 + 真实距离"
            } else {
                "当前轻量安装包未包含 supercombo 模型"
            },
            style = MaterialTheme.typography.labelSmall,
            color = NightVisionColors.TextMuted,
        )
        ToggleSetting(
            label = "启用 Supercombo",
            checked = supercomboEnabled,
            onCheckedChange = onSupercomboToggle,
            enabled = supercomboAvailable,
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
// DETECTION + TRACKING merged section
// ──────────────────────────────────────────────────────────────────────────────

@Composable
private fun DetectionTrackingSection(
    settings: InferenceSettings,
    onSettingsChanged: (InferenceSettings) -> Unit,
) {
    SectionHeader("检测与跟踪")

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

    // ── Tracking subsection ──
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
// CAMERA + DISPLAY merged section
// ──────────────────────────────────────────────────────────────────────────────

@Composable
private fun CameraDisplaySection(
    settings: InferenceSettings,
    onSettingsChanged: (InferenceSettings) -> Unit,
) {
    SectionHeader("相机与显示")

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

    // ── Display subsection ──
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

    SectionCard(title = "行车辅助") {
        ToggleSetting(
            label = "路线预测显示",
            description = "在相机画面上叠加预测的行车轨迹 (需启用 Supercombo)",
            checked = settings.showRoutePrediction,
            onCheckedChange = { onSettingsChanged(settings.copy(showRoutePrediction = it)) },
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ──────────────────────────────────────────────────────────────────────────────
// ALERTS section
// ──────────────────────────────────────────────────────────────────────────────

@Composable
private fun AlertSection(
    settings: InferenceSettings,
    onSettingsChanged: (InferenceSettings) -> Unit,
    onPreviewAlert: (com.nightroadvision.app.alert.RiskSeverity) -> Unit = {},
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

    if (settings.soundAlertsEnabled) {
        SectionCard(title = "警报音效风格") {
            Text(
                text = com.nightroadvision.app.alert.AlertSoundStyle.entries
                    .first { it == settings.alertSoundStyle }.description,
                style = MaterialTheme.typography.labelSmall,
                color = NightVisionColors.TextMuted,
            )
            ChoiceChipRow(
                options = com.nightroadvision.app.alert.AlertSoundStyle.entries
                    .map { it.label to it.description },
                selectedIndex = settings.alertSoundStyle.ordinal,
                onSelect = { index ->
                    onSettingsChanged(settings.copy(
                        alertSoundStyle = com.nightroadvision.app.alert.AlertSoundStyle.entries[index]
                    ))
                },
            )
            // Preview buttons for each level
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                PreviewButton(
                    label = "试听 · 注意",
                    color = Color(0xFFDA6F25),
                    onClick = { onPreviewAlert(com.nightroadvision.app.alert.RiskSeverity.CAUTION) },
                    modifier = Modifier.weight(1f),
                )
                PreviewButton(
                    label = "试听 · 紧急",
                    color = Color(0xFFE8530E),
                    onClick = { onPreviewAlert(com.nightroadvision.app.alert.RiskSeverity.URGENT) },
                    modifier = Modifier.weight(1f),
                )
                PreviewButton(
                    label = "试听 · 危险",
                    color = Color(0xFFC92231),
                    onClick = { onPreviewAlert(com.nightroadvision.app.alert.RiskSeverity.DANGER) },
                    modifier = Modifier.weight(1f),
                )
            }
        }
    }

    SectionCard(title = "报警距离 (米)") {
        DistanceSliderWithPresets(
            label = "危险 · 立即反应",
            value = settings.dangerDistanceM,
            valueRange = 1f..50f,
            presets = listOf(3f, 5f, 8f, 12f, 20f),
            onValueChange = {
                onSettingsChanged(settings.copy(dangerDistanceM = it))
            },
        )
        HorizontalDivider(
            color = NightVisionColors.Border,
            thickness = 0.5.dp,
            modifier = Modifier.padding(vertical = 4.dp),
        )
        DistanceSliderWithPresets(
            label = "紧急 · 减速观察",
            value = settings.urgentDistanceM,
            valueRange = 3f..80f,
            presets = listOf(10f, 15f, 20f, 30f, 50f),
            onValueChange = {
                onSettingsChanged(settings.copy(urgentDistanceM = it))
            },
        )
        HorizontalDivider(
            color = NightVisionColors.Border,
            thickness = 0.5.dp,
            modifier = Modifier.padding(vertical = 4.dp),
        )
        DistanceSliderWithPresets(
            label = "注意 · 提前预警",
            value = settings.cautionDistanceM,
            valueRange = 5f..150f,
            presets = listOf(20f, 30f, 50f, 80f, 120f),
            onValueChange = {
                onSettingsChanged(settings.copy(cautionDistanceM = it))
            },
        )
        Text(
            text = "点击数值可手动输入精确距离",
            style = MaterialTheme.typography.labelSmall,
            color = NightVisionColors.TextMuted,
        )
    }
}

@Composable
private fun PreviewButton(
    label: String,
    color: Color,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Surface(
        onClick = onClick,
        modifier = modifier,
        shape = RoundedCornerShape(8.dp),
        color = color.copy(alpha = 0.12f),
        border = BorderStroke(0.5.dp, color.copy(alpha = 0.4f)),
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.Center,
        ) {
            Icon(
                imageVector = Icons.Filled.PlayArrow,
                contentDescription = null,
                tint = color,
                modifier = Modifier.size(16.dp),
            )
            Spacer(Modifier.width(4.dp))
            Text(
                text = label,
                style = MaterialTheme.typography.labelSmall,
                fontWeight = FontWeight.Medium,
                color = color,
            )
        }
    }
}

@Composable
private fun DistanceSliderWithPresets(
    label: String,
    value: Float,
    valueRange: ClosedFloatingPointRange<Float>,
    presets: List<Float>,
    onValueChange: (Float) -> Unit,
) {
    var showInputDialog by remember { mutableStateOf(false) }
    var inputText by remember { mutableStateOf("") }

    Column {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = label,
                style = MaterialTheme.typography.bodyMedium,
                color = NightVisionColors.Text,
            )
            Text(
                text = String.format(Locale.US, "%.0f m", value),
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Bold,
                fontFamily = FontFamily.Monospace,
                color = NightVisionColors.Accent,
                modifier = Modifier.clickable {
                    inputText = String.format(Locale.US, "%.0f", value)
                    showInputDialog = true
                },
            )
        }
        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = valueRange,
            colors = SliderDefaults.colors(
                thumbColor = NightVisionColors.Accent,
                activeTrackColor = NightVisionColors.Accent,
                inactiveTrackColor = NightVisionColors.Border,
            ),
        )
        // Preset quick-select buttons
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            presets.forEach { preset ->
                val isSelected = kotlin.math.abs(value - preset) < 0.5f
                Surface(
                    modifier = Modifier
                        .weight(1f)
                        .clickable { onValueChange(preset) },
                    shape = RoundedCornerShape(6.dp),
                    color = if (isSelected) NightVisionColors.Accent.copy(alpha = 0.15f)
                    else NightVisionColors.Background,
                    border = BorderStroke(
                        0.5.dp,
                        if (isSelected) NightVisionColors.Accent else NightVisionColors.Border
                    ),
                ) {
                    Text(
                        text = "${preset.toInt()}",
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
                        color = if (isSelected) NightVisionColors.Accent
                        else NightVisionColors.TextSecondary,
                        textAlign = androidx.compose.ui.text.style.TextAlign.Center,
                        modifier = Modifier.fillMaxWidth().padding(horizontal = 6.dp, vertical = 4.dp),
                    )
                }
            }
        }
    }

    // Numeric input dialog
    if (showInputDialog) {
        AlertDialog(
            onDismissRequest = { showInputDialog = false },
            title = {
                Text(
                    text = "输入距离 (米)",
                    color = NightVisionColors.Text,
                )
            },
            text = {
                OutlinedTextField(
                    value = inputText,
                    onValueChange = { inputText = it.filter { c -> c.isDigit() || c == '.' } },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    singleLine = true,
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedTextColor = NightVisionColors.Text,
                        unfocusedTextColor = NightVisionColors.Text,
                        focusedBorderColor = NightVisionColors.Accent,
                        unfocusedBorderColor = NightVisionColors.Border,
                    ),
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    inputText.toFloatOrNull()?.let { parsed ->
                        onValueChange(parsed.coerceIn(valueRange))
                    }
                    showInputDialog = false
                }) {
                    Text("确定", color = NightVisionColors.Accent)
                }
            },
            dismissButton = {
                TextButton(onClick = { showInputDialog = false }) {
                    Text("取消", color = NightVisionColors.TextSecondary)
                }
            },
            containerColor = NightVisionColors.Surface,
        )
    }
}
