package com.nightroadvision.app.ui.screen

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.FlashlightOff
import androidx.compose.material.icons.filled.FlashlightOn
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.camera.view.PreviewView
import com.nightroadvision.app.model.ModelManager
import com.nightroadvision.app.model.ModelQuantization
import com.nightroadvision.app.alert.RidingRisk
import com.nightroadvision.app.alert.RiskReason
import com.nightroadvision.app.alert.RiskSeverity
import com.nightroadvision.app.ui.overlay.DetectionOverlay
import java.util.Locale

// ──────────────────────────────────────────────────────────────────────────────
// Color Palette -- Night Vision Dark Theme
// ──────────────────────────────────────────────────────────────────────────────

object NightVisionColors {
    val Background     = Color(0xFF0D1117)
    val Surface        = Color(0xFF161B22)
    val SurfaceVariant = Color(0xFF1C2333)
    val Accent         = Color(0xFF00FF41)       // night-vision green
    val AccentDim      = Color(0xFF00CC33)
    val Text           = Color(0xFFE6EDF3)
    val TextSecondary  = Color(0xFF8B949E)
    val TextMuted      = Color(0xFF484F58)
    val Border         = Color(0xFF30363D)
    val Danger         = Color(0xFFFF4444)
    val Warning        = Color(0xFFFFAA00)
    val Success        = Color(0xFF00FF41)
    val Glow           = Color(0x3300FF41)       // subtle green glow

    val AccentGradient = Brush.horizontalGradient(
        colors = listOf(Accent.copy(alpha = 0.15f), Accent.copy(alpha = 0.05f))
    )
}

// ──────────────────────────────────────────────────────────────────────────────
// Theme wrapper
// ──────────────────────────────────────────────────────────────────────────────

@Composable
fun NightVisionTheme(content: @Composable () -> Unit) {
    val colorScheme = darkColorScheme(
        primary          = NightVisionColors.Accent,
        onPrimary        = NightVisionColors.Background,
        secondary        = NightVisionColors.AccentDim,
        background       = NightVisionColors.Background,
        surface          = NightVisionColors.Surface,
        surfaceVariant   = NightVisionColors.SurfaceVariant,
        onBackground     = NightVisionColors.Text,
        onSurface        = NightVisionColors.Text,
        onSurfaceVariant = NightVisionColors.TextSecondary,
        outline          = NightVisionColors.Border,
        error            = NightVisionColors.Danger,
    )
    MaterialTheme(colorScheme = colorScheme, content = content)
}

data class PerformanceMetrics(
    val fps: Float = 0f,
    val inferenceLatencyMs: Float = 0f,
    val detectionCount: Int = 0,
    val backend: String = "CPU",
    val thermalStatus: String = "Normal"
)

// ──────────────────────────────────────────────────────────────────────────────
// 1. MODEL SELECTOR BOTTOM SHEET
// ──────────────────────────────────────────────────────────────────────────────

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelSelectorBottomSheet(
    models: List<ModelManager.ModelInfo>,
    currentModelId: String,
    onModelSelected: (ModelManager.ModelInfo) -> Unit,
    onDismiss: () -> Unit
) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = sheetState,
        containerColor = NightVisionColors.Surface,
        contentColor = NightVisionColors.Text,
        dragHandle = {
            Column(
                modifier = Modifier.fillMaxWidth(),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Box(
                    modifier = Modifier
                        .padding(top = 8.dp)
                        .size(width = 40.dp, height = 4.dp)
                        .clip(RoundedCornerShape(2.dp))
                        .background(NightVisionColors.TextMuted)
                )
                Text(
                    text = "选择模型",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 2.sp,
                    color = NightVisionColors.Accent,
                    modifier = Modifier.padding(vertical = 16.dp)
                )
            }
        }
    ) {
        LazyColumn(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp)
                .padding(bottom = 32.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            items(models) { model ->
                val isSelected = model.id == currentModelId
                ModelCard(
                    model = model,
                    isSelected = isSelected,
                    onClick = { onModelSelected(model) }
                )
            }
        }
    }
}

@Composable
private fun ModelCard(
    model: ModelManager.ModelInfo,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    val borderColor = if (isSelected) NightVisionColors.Accent else NightVisionColors.Border
    val background = if (isSelected) {
        NightVisionColors.Accent.copy(alpha = 0.08f)
    } else {
        NightVisionColors.SurfaceVariant
    }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .border(1.dp, borderColor, RoundedCornerShape(12.dp))
            .clickable(onClick = onClick),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = background)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Model icon / indicator
            Box(
                modifier = Modifier
                    .size(48.dp)
                    .clip(CircleShape)
                    .background(
                        if (isSelected) NightVisionColors.Accent.copy(alpha = 0.2f)
                        else NightVisionColors.Background
                    )
                    .border(1.dp, borderColor, CircleShape),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = model.name.take(2).uppercase(),
                    color = if (isSelected) NightVisionColors.Accent else NightVisionColors.TextSecondary,
                    fontWeight = FontWeight.Bold,
                    fontSize = 14.sp,
                    fontFamily = FontFamily.Monospace
                )
            }

            Spacer(modifier = Modifier.width(14.dp))

            // Model info
            Column(modifier = Modifier.weight(1f)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = model.name,
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold,
                        color = if (isSelected) NightVisionColors.Accent else NightVisionColors.Text
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Surface(
                        shape = RoundedCornerShape(4.dp),
                        color = NightVisionColors.Background
                    ) {
                        Text(
                            text = model.size,
                            style = MaterialTheme.typography.labelSmall,
                            color = NightVisionColors.TextSecondary,
                            modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                        )
                    }
                    if (model.parameterCount.isNotEmpty()) {
                        Spacer(modifier = Modifier.width(4.dp))
                        Surface(
                            shape = RoundedCornerShape(4.dp),
                            color = NightVisionColors.Background
                        ) {
                            Text(
                                text = "${model.parameterCount} 参数",
                                style = MaterialTheme.typography.labelSmall,
                                color = NightVisionColors.TextMuted,
                                modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                            )
                        }
                    }
                }
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = model.description,
                    style = MaterialTheme.typography.bodySmall,
                    color = NightVisionColors.TextSecondary,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis
                )
            }

            // Selection indicator
            if (isSelected) {
                Spacer(modifier = Modifier.width(8.dp))
                Icon(
                    imageVector = Icons.Filled.CheckCircle,
                    contentDescription = "已选择",
                    tint = NightVisionColors.Accent,
                    modifier = Modifier.size(24.dp)
                )
            }
        }
    }
}

// 3. PERFORMANCE PANEL (top-left overlay)
// ──────────────────────────────────────────────────────────────────────────────

@Composable
fun PerformancePanel(
    modelName: String,
    metrics: PerformanceMetrics,
    modifier: Modifier = Modifier
) {
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(10.dp),
        color = NightVisionColors.Background.copy(alpha = 0.85f),
        border = BorderStroke(1.dp, NightVisionColors.Border.copy(alpha = 0.6f))
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            // Header
            Text(
                text = "遥测数据",
                style = MaterialTheme.typography.labelSmall,
                fontWeight = FontWeight.Bold,
                letterSpacing = 2.sp,
                color = NightVisionColors.Accent,
                fontSize = 9.sp
            )

            HorizontalDivider(
                color = NightVisionColors.Border,
                thickness = 0.5.dp,
                modifier = Modifier.padding(vertical = 2.dp)
            )

            // Model name
            MetricRow(
                label = "模型",
                value = modelName,
                valueColor = NightVisionColors.Accent
            )

            // FPS
            MetricRow(
                label = "FPS",
                value = String.format(Locale.US, "%.1f", metrics.fps),
                valueColor = when {
                    metrics.fps >= 24f -> NightVisionColors.Success
                    metrics.fps >= 15f -> NightVisionColors.Warning
                    else -> NightVisionColors.Danger
                }
            )

            // Inference latency
            MetricRow(
                label = "延迟",
                value = "${metrics.inferenceLatencyMs.toInt()} ms",
                valueColor = when {
                    metrics.inferenceLatencyMs <= 30f -> NightVisionColors.Success
                    metrics.inferenceLatencyMs <= 60f -> NightVisionColors.Warning
                    else -> NightVisionColors.Danger
                }
            )

            // Detection count
            MetricRow(
                label = "检测数",
                value = "${metrics.detectionCount}",
                valueColor = if (metrics.detectionCount > 0) NightVisionColors.Accent
                             else NightVisionColors.TextSecondary
            )

            // Backend
            MetricRow(
                label = "后端",
                value = metrics.backend,
                valueColor = if (metrics.backend.contains("GPU")) NightVisionColors.Accent
                             else NightVisionColors.TextSecondary
            )

            // Thermal status
            MetricRow(
                label = "温度",
                value = metrics.thermalStatus,
                valueColor = when (metrics.thermalStatus.lowercase()) {
                    "normal", "none" -> NightVisionColors.Success
                    "light", "moderate" -> NightVisionColors.Warning
                    "severe", "critical", "shutdown" -> NightVisionColors.Danger
                    else -> NightVisionColors.TextSecondary
                }
            )
        }
    }
}

@Composable
private fun MetricRow(
    label: String,
    value: String,
    valueColor: Color
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = NightVisionColors.TextMuted,
            fontFamily = FontFamily.Monospace,
            fontSize = 10.sp
        )
        Text(
            text = value,
            style = MaterialTheme.typography.labelSmall,
            fontWeight = FontWeight.Bold,
            color = valueColor,
            fontFamily = FontFamily.Monospace,
            fontSize = 11.sp
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// 4. Circle icon button (settings gear, flashlight, telemetry toggle)
// ──────────────────────────────────────────────────────────────────────────────

@Composable
private fun CircleIconButton(
    onClick: () -> Unit,
    icon: @Composable () -> Unit,
    modifier: Modifier = Modifier
) {
    Surface(
        onClick = onClick,
        modifier = modifier.size(48.dp),
        shape = CircleShape,
        color = Color.Black.copy(alpha = 0.48f),
        border = BorderStroke(1.dp, Color.White.copy(alpha = 0.14f))
    ) {
        Box(contentAlignment = Alignment.Center) {
            icon()
        }
    }
}

@Composable
private fun VisionStatusChip(
    modelName: String,
    modifier: Modifier = Modifier,
) {
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(18.dp),
        color = Color.Black.copy(alpha = 0.48f),
        border = BorderStroke(1.dp, NightVisionColors.Accent.copy(alpha = 0.34f)),
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 11.dp, vertical = 7.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Box(
                modifier = Modifier
                    .size(8.dp)
                    .background(NightVisionColors.Accent, CircleShape),
            )
            Spacer(Modifier.width(7.dp))
            Column {
                Text(
                    text = "视觉",
                    color = Color.White,
                    fontWeight = FontWeight.Black,
                    fontSize = 10.sp,
                    letterSpacing = 1.sp,
                )
                Text(
                    text = modelName,
                    color = Color.White.copy(alpha = 0.66f),
                    fontFamily = FontFamily.Monospace,
                    fontSize = 8.sp,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
        }
    }
}

@Composable
private fun TelemetryChip(
    metrics: PerformanceMetrics,
    active: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Surface(
        onClick = onClick,
        modifier = modifier.height(48.dp),
        shape = RoundedCornerShape(18.dp),
        color = Color.Black.copy(alpha = 0.48f),
        border = BorderStroke(
            1.dp,
            if (active) NightVisionColors.Accent.copy(alpha = 0.65f)
            else Color.White.copy(alpha = 0.14f),
        ),
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 11.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = String.format(Locale.US, "%.0f", metrics.fps),
                color = Color.White,
                fontWeight = FontWeight.Bold,
                fontFamily = FontFamily.Monospace,
                fontSize = 13.sp,
            )
            Text(
                text = " FPS  ·  ${metrics.inferenceLatencyMs.toInt()} ms",
                color = Color.White.copy(alpha = 0.65f),
                fontFamily = FontFamily.Monospace,
                fontSize = 9.sp,
            )
        }
    }
}

@Composable
private fun TargetAlertChip(
    risk: RidingRisk,
    modifier: Modifier = Modifier,
) {
    if (risk.severity == RiskSeverity.NONE) return

    val alertColor = when (risk.severity) {
        RiskSeverity.DANGER -> Color(0xFFC92231)
        RiskSeverity.URGENT -> Color(0xFFE8530E)
        RiskSeverity.CAUTION -> Color(0xFFDA6F25)
        RiskSeverity.NONE -> Color.Transparent
    }
    val reason = when (risk.reason) {
        RiskReason.IMMINENT -> "非常危险"
        RiskReason.VERY_NEAR -> "目标很近"
        RiskReason.APPROACHING_QUICKLY -> "正在快速接近"
        RiskReason.IN_ATTENTION_ZONE -> "进入注意区域"
        RiskReason.NONE -> ""
    }
    val target = buildString {
        append(risk.className)
        risk.trackId?.let { append("  #$it") }
    }
    val distText = risk.estimatedDistanceM?.let { "%.0fm".format(it) }

    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(18.dp),
        color = alertColor.copy(alpha = 0.92f),
        border = BorderStroke(1.dp, Color.White.copy(alpha = 0.18f)),
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 14.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = buildString {
                    append(target)
                    append("  ·  ")
                    append(reason)
                    distText?.let { append("  ·  $it") }
                },
                color = Color.White,
                fontWeight = FontWeight.Black,
                fontSize = 12.sp,
            )
        }
    }
}

@Composable
private fun HudControls(
    portrait: Boolean,
    flashlightOn: Boolean,
    onToggleFlashlight: () -> Unit,
    onOpenSettings: () -> Unit,
    modifier: Modifier = Modifier,
    showCameraSwitch: Boolean = false,
    cameraLensLabel: String = "CAM",
    onCycleCamera: () -> Unit = {},
) {
    if (portrait) {
        Column(
            modifier = modifier,
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            HudFlashlightButton(flashlightOn, onToggleFlashlight)
            if (showCameraSwitch) HudCameraSwitchButton(cameraLensLabel, onCycleCamera)
            HudSettingsButton(onOpenSettings)
        }
    } else {
        Row(
            modifier = modifier,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            HudFlashlightButton(flashlightOn, onToggleFlashlight)
            if (showCameraSwitch) HudCameraSwitchButton(cameraLensLabel, onCycleCamera)
            HudSettingsButton(onOpenSettings)
        }
    }
}

@Composable
private fun HudCameraSwitchButton(label: String, onClick: () -> Unit) {
    CircleIconButton(
        onClick = onClick,
        icon = {
            Text(
                text = label,
                color = Color.White.copy(alpha = 0.92f),
                fontSize = if (label.length > 3) 9.sp else 11.sp,
                fontWeight = FontWeight.Black,
                fontFamily = FontFamily.Monospace,
            )
        },
    )
}

@Composable
private fun HudFlashlightButton(flashlightOn: Boolean, onClick: () -> Unit) {
    CircleIconButton(
        onClick = onClick,
        icon = {
            Icon(
                imageVector = if (flashlightOn) Icons.Filled.FlashlightOn else Icons.Filled.FlashlightOff,
                contentDescription = if (flashlightOn) "关闭手电筒" else "打开手电筒",
                tint = if (flashlightOn) NightVisionColors.Warning else Color.White.copy(alpha = 0.78f),
                modifier = Modifier.size(21.dp),
            )
        },
    )
}

@Composable
private fun HudSettingsButton(onClick: () -> Unit) {
    CircleIconButton(
        onClick = onClick,
        icon = {
            Icon(
                imageVector = Icons.Filled.Settings,
                contentDescription = "设置",
                tint = Color.White.copy(alpha = 0.78f),
                modifier = Modifier.size(21.dp),
            )
        },
    )
}

// ──────────────────────────────────────────────────────────────────────────────
// 5. MAIN SCREEN (root composable)
// ──────────────────────────────────────────────────────────────────────────────

/**
 * MainScreen composable that displays the camera preview with detection overlay,
 * performance telemetry, and a full settings panel.
 *
 * The DetectionOverlay receives:
 *   - detections: in camera coordinate space (1280x720)
 *   - cameraWidth / cameraHeight: so the overlay can map camera coords -> screen coords
 */
@Composable
fun MainScreen(
    modifier: Modifier = Modifier,
    viewModel: MainScreenViewModel = viewModel(),
) {
    val lifecycleOwner = LocalLifecycleOwner.current
    val configurationOrientation = LocalConfiguration.current.orientation

    // ViewModel state
    val detections by viewModel.detections.collectAsState()
    val metrics by viewModel.performanceMetrics.collectAsState()
    val settings by viewModel.settings.collectAsState()
    val currentModelName by viewModel.currentModelName.collectAsState()
    val errorMessage by viewModel.errorMessage.collectAsState()
    val activeDelegate by viewModel.activeDelegate.collectAsState()
    val ridingRisk by viewModel.ridingRisk.collectAsState()
    val supercomboEnabled by viewModel.supercomboEnabled.collectAsState()
    val supercomboLatencyMs by viewModel.supercomboLatencyMs.collectAsState()

    // Local UI state
    var showSettingsPage by remember { mutableStateOf(false) }
    var showModelSelector by remember { mutableStateOf(false) }
    var showPerformancePanel by remember { mutableStateOf(false) }
    var flashlightOn by remember { mutableStateOf(false) }

    // Camera setup - runs once on first composition
    val cameraManager = remember {
        viewModel.createCameraManager(lifecycleOwner)
    }
    var cameraStarted by remember { mutableStateOf(false) }
    val availableLenses by cameraManager.availableLenses.collectAsState()
    val selectedLens by cameraManager.currentLens.collectAsState()

    // Cleanup camera when composable leaves composition
    DisposableEffect(Unit) {
        onDispose {
            cameraManager.release()
        }
    }

    NightVisionTheme {
        BoxWithConstraints(
            modifier = modifier
                .fillMaxSize()
                .background(NightVisionColors.Background)
        ) {
            val portrait = maxHeight > maxWidth
            val edgePadding = if (portrait) 10.dp else 14.dp

            // ── Camera Preview ──
            AndroidView(
                factory = { ctx ->
                    PreviewView(ctx).apply {
                        // PERFORMANCE may select SurfaceView, whose separate surface can be
                        // composed above the Compose Canvas on some devices. Use TextureView so
                        // the detection overlay is guaranteed to stay visible over the preview.
                        implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                        scaleType = PreviewView.ScaleType.FILL_CENTER
                    }
                },
                update = { previewView ->
                    val displayRotation = previewView.display?.rotation ?: if (
                        configurationOrientation == android.content.res.Configuration.ORIENTATION_LANDSCAPE
                    ) {
                        android.view.Surface.ROTATION_90
                    } else {
                        android.view.Surface.ROTATION_0
                    }
                    if (!cameraStarted) {
                        cameraStarted = true
                        cameraManager.startCamera(previewView.surfaceProvider, displayRotation)
                    } else {
                        cameraManager.updateTargetRotation(displayRotation)
                    }
                },
                modifier = Modifier.fillMaxSize()
            )

            // openpilot-style edge fade: enough contrast for corner HUD elements while
            // keeping the central camera feed effectively untouched.
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(
                        if (portrait) {
                            Brush.verticalGradient(
                                0f to Color.Black.copy(alpha = 0.34f),
                                0.14f to Color.Transparent,
                                0.88f to Color.Transparent,
                                1f to Color.Black.copy(alpha = 0.28f),
                            )
                        } else {
                            // Landscape: fade top/bottom edges with shorter stops,
                            // plus subtle left/right fade for HUD contrast.
                            Brush.verticalGradient(
                                0f to Color.Black.copy(alpha = 0.30f),
                                0.18f to Color.Transparent,
                                0.82f to Color.Transparent,
                                1f to Color.Black.copy(alpha = 0.24f),
                            )
                        },
                    ),
            )

            // ── Detection overlay ──
            DetectionOverlay(
                detections = detections,
                highlightedTrackId = ridingRisk.trackId,
                cameraWidth = viewModel.getCameraFrameWidth(),
                cameraHeight = viewModel.getCameraFrameHeight(),
                showLabels = settings.showLabels,
                showConfidence = settings.showConfidence,
                showTrackIds = settings.showTrackIds,
                modifier = Modifier.fillMaxSize()
            )

            // Like openpilot's state border, this remains peripheral and changes only
            // when a tracked person or vehicle occupies a large part of the frame.
            // Uses thick borders with a soft outer glow for high visibility.
            val riskBorderColor = when (ridingRisk.severity) {
                RiskSeverity.DANGER -> Color(0xFFC92231)
                RiskSeverity.URGENT -> Color(0xFFE8530E)
                RiskSeverity.CAUTION -> Color(0xFFDA6F25)
                RiskSeverity.NONE -> NightVisionColors.Accent
            }
            val riskBorderWidth = when (ridingRisk.severity) {
                RiskSeverity.DANGER -> 10.dp
                RiskSeverity.URGENT -> 6.dp
                RiskSeverity.CAUTION -> 3.dp
                RiskSeverity.NONE -> 1.dp
            }
            val glowWidth = when (ridingRisk.severity) {
                RiskSeverity.DANGER -> 24.dp
                RiskSeverity.URGENT -> 14.dp
                RiskSeverity.CAUTION -> 6.dp
                RiskSeverity.NONE -> 0.dp
            }

            // Pulsing animation for DANGER (always created to satisfy Compose rules)
            val infiniteTransition = rememberInfiniteTransition(label = "risk_pulse")
            val pulseRaw by infiniteTransition.animateFloat(
                initialValue = 0.25f,
                targetValue = 0.65f,
                animationSpec = infiniteRepeatable(
                    animation = tween(durationMillis = 500),
                    repeatMode = RepeatMode.Reverse,
                ),
                label = "danger_pulse",
            )
            val pulseAlpha = if (ridingRisk.severity == RiskSeverity.DANGER) pulseRaw else 0.35f

            if (ridingRisk.severity != RiskSeverity.NONE) {
                // Outer glow layer — wider, semi-transparent
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .border(
                            width = glowWidth,
                            color = riskBorderColor.copy(alpha = pulseAlpha * 0.5f),
                        ),
                )
                // Inner solid border — crisp, high-contrast
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .border(
                            width = riskBorderWidth,
                            color = riskBorderColor.copy(alpha = 0.90f),
                        ),
                )
            } else {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .border(
                            width = 1.dp,
                            color = riskBorderColor.copy(alpha = 0.20f),
                        ),
                )
            }

            VisionStatusChip(
                modelName = currentModelName,
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .statusBarsPadding()
                    .padding(start = edgePadding, top = 8.dp)
                    .widthIn(max = if (portrait) 142.dp else 180.dp),
            )

            TelemetryChip(
                metrics = metrics,
                active = showPerformancePanel,
                onClick = { showPerformancePanel = !showPerformancePanel },
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .statusBarsPadding()
                    .padding(end = edgePadding, top = 8.dp),
            )

            // ── Performance Panel (top-left) ──
            AnimatedVisibility(
                visible = showPerformancePanel,
                enter = fadeIn() + slideInHorizontally(initialOffsetX = { -it }),
                exit = fadeOut() + slideOutHorizontally(targetOffsetX = { -it }),
                modifier = Modifier
                    .statusBarsPadding()
                    .padding(end = edgePadding, top = 64.dp)
                    .align(Alignment.TopEnd)
                    .widthIn(min = 160.dp, max = 200.dp)
            ) {
                PerformancePanel(
                    modelName = currentModelName,
                    metrics = metrics
                )
            }

            TargetAlertChip(
                risk = ridingRisk,
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .navigationBarsPadding()
                    .padding(bottom = 10.dp),
            )

            HudControls(
                portrait = portrait,
                flashlightOn = flashlightOn,
                onToggleFlashlight = {
                    flashlightOn = viewModel.toggleFlashlight()
                },
                onOpenSettings = {
                    showSettingsPage = true
                },
                showCameraSwitch = availableLenses.size > 1,
                cameraLensLabel = selectedLens?.label ?: "CAM",
                onCycleCamera = {
                    viewModel.cycleCamera()
                    flashlightOn = false
                },
                modifier = Modifier
                    .align(Alignment.BottomEnd)
                    .navigationBarsPadding()
                    .padding(end = edgePadding, bottom = 8.dp),
            )

            // ── Error Snackbar ──
            errorMessage?.let { error ->
                Snackbar(
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .navigationBarsPadding()
                        .padding(bottom = 64.dp, start = 16.dp, end = 16.dp),
                    containerColor = NightVisionColors.Danger.copy(alpha = 0.9f),
                    contentColor = NightVisionColors.Text,
                    dismissAction = {
                        TextButton(onClick = { viewModel.clearError() }) {
                            Text("忽略", color = NightVisionColors.Accent, fontWeight = FontWeight.Bold)
                        }
                    },
                    action = {
                        TextButton(onClick = { viewModel.clearError() }) {
                            Text("确定", color = NightVisionColors.Accent, fontWeight = FontWeight.Bold)
                        }
                    }
                ) {
                    Text(text = error, style = MaterialTheme.typography.bodySmall)
                }
            }
        }

        // ── Settings Page (full-screen) ──
        if (showSettingsPage) {
            SettingsScreen(
                settings = settings,
                currentModelName = currentModelName,
                onSettingsChanged = { newSettings ->
                    viewModel.updateSettings(newSettings)
                },
                onResetSettings = viewModel::resetSettings,
                int8Available = viewModel.isInt8Available(),
                onQuantizationSelected = viewModel::selectQuantization,
                onDismiss = { showSettingsPage = false },
                supercomboEnabled = supercomboEnabled,
                onSupercomboToggle = { viewModel.setSupercomboEnabled(it) },
                supercomboLatencyMs = supercomboLatencyMs,
            )
        }

        // ── Model Selector Bottom Sheet ──
        if (showModelSelector) {
            val availableModels = remember { viewModel.getAvailableModels() }
            ModelSelectorBottomSheet(
                models = availableModels,
                currentModelId = settings.selectedModelId,
                onModelSelected = { model ->
                    viewModel.switchModel(model)
                    showModelSelector = false
                    showSettingsPage = true
                },
                onDismiss = { showModelSelector = false }
            )
        }
    }
}
