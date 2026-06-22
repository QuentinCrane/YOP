package com.nightroadvision.app.ui.screen

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
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
import androidx.core.content.ContextCompat
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.camera.view.PreviewView
import com.nightroadvision.app.model.ModelManager
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

            // Inference throughput; this is intentionally distinct from UI refresh rate.
            MetricRow(
                label = "推理 FPS",
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
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Surface(
        onClick = onClick,
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
                text = " AI FPS  ·  ${metrics.inferenceLatencyMs.toInt()} ms",
                color = Color.White.copy(alpha = 0.65f),
                fontFamily = FontFamily.Monospace,
                fontSize = 9.sp,
            )
        }
    }
}

@Composable
private fun SpeedChip(
    speedKmh: Float,
    speedStyle: SpeedStyle,
    modifier: Modifier = Modifier,
) {
    when (speedStyle) {
        SpeedStyle.COMPACT -> SpeedStyleBar(speedKmh, modifier)
        SpeedStyle.DIGITAL -> SpeedStyleGauge(speedKmh, modifier)
        SpeedStyle.LARGE -> SpeedStyleNeon(speedKmh, modifier)
        SpeedStyle.MINIMAL -> SpeedStyleHud(speedKmh, modifier)
    }
}

/** Style 1: Full-width gradient bar with speed + label */
@Composable
private fun SpeedStyleBar(speedKmh: Float, modifier: Modifier) {
    val active = speedKmh > 1f
    Surface(
        modifier = modifier.height(44.dp),
        shape = RoundedCornerShape(8.dp),
        color = Color.Transparent,
    ) {
        Box(
            modifier = Modifier
                .background(
                    brush = Brush.horizontalGradient(
                        colors = if (active) listOf(
                            NightVisionColors.Accent.copy(alpha = 0.35f),
                            Color.Black.copy(alpha = 0.6f),
                        ) else listOf(
                            Color.White.copy(alpha = 0.06f),
                            Color.Black.copy(alpha = 0.4f),
                        )
                    )
                )
                .border(
                    1.dp,
                    if (active) NightVisionColors.Accent.copy(alpha = 0.4f)
                    else Color.White.copy(alpha = 0.08f),
                    RoundedCornerShape(8.dp),
                )
                .padding(horizontal = 14.dp),
            contentAlignment = Alignment.Center,
        ) {
            Row(verticalAlignment = Alignment.Bottom) {
                Text(
                    text = String.format(Locale.US, "%.0f", speedKmh),
                    color = if (active) Color.White else Color.White.copy(alpha = 0.3f),
                    fontWeight = FontWeight.Black,
                    fontFamily = FontFamily.Monospace,
                    fontSize = 22.sp,
                )
                Spacer(Modifier.width(4.dp))
                Text(
                    text = "km/h",
                    color = if (active) NightVisionColors.Accent.copy(alpha = 0.7f)
                    else Color.White.copy(alpha = 0.2f),
                    fontWeight = FontWeight.Bold,
                    fontFamily = FontFamily.Monospace,
                    fontSize = 10.sp,
                    modifier = Modifier.padding(bottom = 3.dp),
                )
            }
        }
    }
}

/** Style 2: Circular gauge with arc progress */
@Composable
private fun SpeedStyleGauge(speedKmh: Float, modifier: Modifier) {
    val active = speedKmh > 1f
    val sweep = (speedKmh.coerceIn(0f, 60f) / 60f) * 270f
    val accentColor = when {
        speedKmh > 25 -> Color(0xFFFF4444)
        speedKmh > 15 -> Color(0xFFFFAA00)
        active -> NightVisionColors.Accent
        else -> Color.White.copy(alpha = 0.15f)
    }

    Box(modifier = modifier.size(64.dp), contentAlignment = Alignment.Center) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val stroke = 4.dp.toPx()
            val pad = stroke / 2
            val arcSize = Size(size.width - stroke, size.height - stroke)
            val topLeft = Offset(pad, pad)
            // Background arc
            drawArc(
                color = Color.White.copy(alpha = 0.08f),
                startAngle = 135f,
                sweepAngle = 270f,
                useCenter = false,
                topLeft = topLeft,
                size = arcSize,
                style = Stroke(width = stroke, cap = StrokeCap.Round),
            )
            // Active arc
            drawArc(
                color = accentColor.copy(alpha = 0.8f),
                startAngle = 135f,
                sweepAngle = sweep,
                useCenter = false,
                topLeft = topLeft,
                size = arcSize,
                style = Stroke(width = stroke, cap = StrokeCap.Round),
            )
        }
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                text = String.format(Locale.US, "%.0f", speedKmh),
                color = if (active) Color.White else Color.White.copy(alpha = 0.3f),
                fontWeight = FontWeight.Black,
                fontFamily = FontFamily.Monospace,
                fontSize = 18.sp,
            )
            Text(
                text = "km/h",
                color = accentColor.copy(alpha = 0.6f),
                fontFamily = FontFamily.Monospace,
                fontSize = 7.sp,
                fontWeight = FontWeight.Bold,
            )
        }
    }
}

/** Style 3: Neon glow — big number with colored glow effect */
@Composable
private fun SpeedStyleNeon(speedKmh: Float, modifier: Modifier) {
    val active = speedKmh > 1f
    val glowColor = when {
        speedKmh > 25 -> Color(0xFFFF2222)
        speedKmh > 15 -> Color(0xFFFF8800)
        active -> NightVisionColors.Accent
        else -> Color.White.copy(alpha = 0.1f)
    }

    Column(
        modifier = modifier.padding(horizontal = 8.dp),
        horizontalAlignment = Alignment.End,
    ) {
        Text(
            text = String.format(Locale.US, "%.0f", speedKmh),
            color = if (active) glowColor else Color.White.copy(alpha = 0.2f),
            fontWeight = FontWeight.Black,
            fontFamily = FontFamily.Monospace,
            fontSize = 42.sp,
            letterSpacing = (-1).sp,
        )
        Box(
            modifier = Modifier
                .width(60.dp)
                .height(2.dp)
                .background(
                    brush = Brush.horizontalGradient(
                        colors = listOf(glowColor.copy(alpha = 0f), glowColor.copy(alpha = 0.6f))
                    )
                )
        )
        Spacer(Modifier.height(2.dp))
        Text(
            text = "KM/H",
            color = glowColor.copy(alpha = 0.4f),
            fontFamily = FontFamily.Monospace,
            fontSize = 9.sp,
            fontWeight = FontWeight.Bold,
            letterSpacing = 3.sp,
        )
    }
}

/** Style 4: Military HUD — boxed tactical readout */
@Composable
private fun SpeedStyleHud(speedKmh: Float, modifier: Modifier) {
    val active = speedKmh > 1f
    val borderColor = if (active) Color.White.copy(alpha = 0.25f) else Color.White.copy(alpha = 0.08f)

    Box(
        modifier = modifier
            .border(1.dp, borderColor, RoundedCornerShape(2.dp))
            .padding(horizontal = 10.dp, vertical = 6.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            // Small bracket decoration
            Text(
                text = "[",
                color = borderColor,
                fontFamily = FontFamily.Monospace,
                fontWeight = FontWeight.Light,
                fontSize = 18.sp,
            )
            Column(
                modifier = Modifier.padding(horizontal = 4.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Text(
                    text = "SPD",
                    color = if (active) Color.White.copy(alpha = 0.35f) else Color.White.copy(alpha = 0.12f),
                    fontFamily = FontFamily.Monospace,
                    fontSize = 7.sp,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 2.sp,
                )
                Text(
                    text = String.format(Locale.US, "%03.0f", speedKmh),
                    color = if (active) Color.White else Color.White.copy(alpha = 0.25f),
                    fontWeight = FontWeight.Black,
                    fontFamily = FontFamily.Monospace,
                    fontSize = 20.sp,
                )
            }
            Text(
                text = "]",
                color = borderColor,
                fontFamily = FontFamily.Monospace,
                fontWeight = FontWeight.Light,
                fontSize = 18.sp,
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

/**
 * Thin state frame with a soft, tapered halo. The large radius and small edge
 * inset mimic the continuous screen corners used by modern iPhones without
 * turning the warning into an opaque band over the camera preview.
 */
@Composable
private fun RiskGlowFrame(
    severity: RiskSeverity,
    portrait: Boolean,
    modifier: Modifier = Modifier,
) {
    val primary = when (severity) {
        RiskSeverity.DANGER -> Color(0xFFFF3B4E)
        RiskSeverity.URGENT -> Color(0xFFFF7A1A)
        RiskSeverity.CAUTION -> Color(0xFFFFB340)
        RiskSeverity.NONE -> NightVisionColors.Accent
    }
    val highlight = when (severity) {
        RiskSeverity.DANGER -> Color(0xFFFF8A9A)
        RiskSeverity.URGENT -> Color(0xFFFFC06A)
        RiskSeverity.CAUTION -> Color(0xFFFFD98A)
        RiskSeverity.NONE -> NightVisionColors.AccentDim
    }

    val pulse = if (severity == RiskSeverity.NONE) {
        1f
    } else {
        val pulseTransition = rememberInfiniteTransition(label = "risk_frame_pulse")
        val animatedPulse by pulseTransition.animateFloat(
            initialValue = 0.62f,
            targetValue = 1f,
            animationSpec = infiniteRepeatable(
                animation = tween(
                    durationMillis = if (severity == RiskSeverity.DANGER) 520 else 920,
                ),
                repeatMode = RepeatMode.Reverse,
            ),
            label = "risk_frame_alpha",
        )
        animatedPulse
    }

    Canvas(modifier = modifier) {
        val inset = 3.dp.toPx()
        val frameSize = Size(
            width = (size.width - inset * 2f).coerceAtLeast(0f),
            height = (size.height - inset * 2f).coerceAtLeast(0f),
        )
        if (frameSize.width <= 0f || frameSize.height <= 0f) return@Canvas

        val radius = (if (portrait) 46.dp else 32.dp).toPx()
            .coerceAtMost(frameSize.minDimension / 2f)
        val corner = CornerRadius(radius, radius)
        val frameBrush = Brush.sweepGradient(
            colors = listOf(primary, highlight, primary, primary.copy(alpha = 0.72f), primary),
            center = Offset(size.width / 2f, size.height / 2f),
        )

        if (severity == RiskSeverity.NONE) {
            drawRoundRect(
                color = primary.copy(alpha = 0.16f),
                topLeft = Offset(inset, inset),
                size = frameSize,
                cornerRadius = corner,
                style = Stroke(width = 0.75.dp.toPx()),
            )
            return@Canvas
        }

        // Wide layers stay deliberately faint; only the 1.25dp core is crisp.
        fun drawHalo(widthPixels: Float, alpha: Float) {
            drawRoundRect(
                brush = frameBrush,
                topLeft = Offset(inset, inset),
                size = frameSize,
                cornerRadius = corner,
                alpha = alpha * pulse,
                style = Stroke(width = widthPixels),
            )
        }
        drawHalo(15.dp.toPx(), 0.035f)
        drawHalo(9.dp.toPx(), 0.060f)
        drawHalo(5.dp.toPx(), 0.105f)
        drawHalo(2.5.dp.toPx(), 0.18f)
        drawRoundRect(
            brush = frameBrush,
            topLeft = Offset(inset, inset),
            size = frameSize,
            cornerRadius = corner,
            alpha = 0.82f * pulse,
            style = Stroke(width = 1.25.dp.toPx()),
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// 5. MAIN SCREEN (root composable)
// ──────────────────────────────────────────────────────────────────────────────

@Composable
private fun BoxScope.PerceptionLayers(
    viewModel: MainScreenViewModel,
    settings: InferenceSettings,
    portrait: Boolean,
) {
    val detections by viewModel.detections.collectAsState()
    val ridingRisk by viewModel.ridingRisk.collectAsState()
    val drivingCorridor by viewModel.drivingCorridor.collectAsState()

    DetectionOverlay(
        detections = detections,
        highlightedTrackId = ridingRisk.trackId,
        cameraWidth = viewModel.getCameraFrameWidth(),
        cameraHeight = viewModel.getCameraFrameHeight(),
        showLabels = settings.showLabels,
        showConfidence = settings.showConfidence,
        showTrackIds = settings.showTrackIds,
        drivingCorridor = if (settings.showRoutePrediction) {
            drivingCorridor
        } else {
            com.nightroadvision.app.motion.DrivingCorridor.EMPTY
        },
        modifier = Modifier.fillMaxSize(),
    )

    RiskGlowFrame(
        severity = ridingRisk.severity,
        portrait = portrait,
        modifier = Modifier.fillMaxSize(),
    )

    TargetAlertChip(
        risk = ridingRisk,
        modifier = Modifier
            .align(Alignment.BottomCenter)
            .navigationBarsPadding()
            .padding(bottom = 10.dp),
    )
}

@Composable
private fun BoxScope.PerformanceLayers(
    viewModel: MainScreenViewModel,
    currentModelName: String,
    showPerformancePanel: Boolean,
    edgePadding: Dp,
    onTogglePanel: () -> Unit,
) {
    val metrics by viewModel.performanceMetrics.collectAsState()

    TelemetryChip(
        metrics = metrics,
        active = showPerformancePanel,
        onClick = onTogglePanel,
        modifier = Modifier
            .align(Alignment.TopEnd)
            .statusBarsPadding()
            .padding(end = edgePadding, top = 8.dp),
    )

    AnimatedVisibility(
        visible = showPerformancePanel,
        enter = fadeIn() + slideInHorizontally(initialOffsetX = { -it }),
        exit = fadeOut() + slideOutHorizontally(targetOffsetX = { -it }),
        modifier = Modifier
            .statusBarsPadding()
            .padding(end = edgePadding, top = 104.dp)
            .align(Alignment.TopEnd)
            .widthIn(min = 160.dp, max = 200.dp),
    ) {
        PerformancePanel(modelName = currentModelName, metrics = metrics)
    }
}

@Composable
private fun BoxScope.SpeedLayer(
    viewModel: MainScreenViewModel,
    speedStyle: SpeedStyle,
    edgePadding: Dp,
) {
    val speedKmh by viewModel.speedKmh.collectAsState()
    SpeedChip(
        speedKmh = speedKmh,
        speedStyle = speedStyle,
        modifier = Modifier
            .align(Alignment.TopEnd)
            .statusBarsPadding()
            .padding(end = edgePadding, top = 60.dp),
    )
}

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
    val settings by viewModel.settings.collectAsState()
    val currentModelName by viewModel.currentModelName.collectAsState()
    val errorMessage by viewModel.errorMessage.collectAsState()
    val activeDelegate by viewModel.activeDelegate.collectAsState()

    // GPS permission
    val context = androidx.compose.ui.platform.LocalContext.current
    val gpsPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val isGranted = permissions[Manifest.permission.ACCESS_FINE_LOCATION] == true
        if (isGranted && lifecycleOwner.lifecycle.currentState.isAtLeast(Lifecycle.State.STARTED)) {
            viewModel.startGpsSpeed()
        }
    }
    LaunchedEffect(Unit) {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION)
            != PackageManager.PERMISSION_GRANTED
        ) {
            gpsPermissionLauncher.launch(
                arrayOf(
                    Manifest.permission.ACCESS_COARSE_LOCATION,
                    Manifest.permission.ACCESS_FINE_LOCATION,
                )
            )
        }
    }

    // Local UI state
    var showSettingsPage by remember { mutableStateOf(false) }
    var showModelSelector by remember { mutableStateOf(false) }
    var showPerformancePanel by remember { mutableStateOf(false) }
    var flashlightOn by remember { mutableStateOf(false) }
    val settingsPageVisible by rememberUpdatedState(showSettingsPage)

    // Keep sensors and GPS scoped to the visible camera lifecycle to avoid background drain.
    DisposableEffect(lifecycleOwner, viewModel, context) {
        fun startMotionSources() {
            if (settingsPageVisible) return
            viewModel.startMotionEstimation()
            if (ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION)
                == PackageManager.PERMISSION_GRANTED
            ) {
                viewModel.startGpsSpeed()
            }
        }

        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_START -> startMotionSources()
                Lifecycle.Event.ON_STOP -> {
                    viewModel.stopMotionEstimation()
                    viewModel.stopGpsSpeed()
                }
                else -> Unit
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        if (lifecycleOwner.lifecycle.currentState.isAtLeast(Lifecycle.State.STARTED)) {
            startMotionSources()
        }
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
            viewModel.stopMotionEstimation()
            viewModel.stopGpsSpeed()
        }
    }

    DisposableEffect(viewModel) {
        onDispose { viewModel.setInferencePaused(false) }
    }

    // Camera setup - runs once on first composition
    val cameraManager = remember(viewModel, lifecycleOwner) {
        viewModel.createCameraManager(lifecycleOwner)
    }
    var cameraStarted by remember(cameraManager) { mutableStateOf(false) }
    val availableLenses by cameraManager.availableLenses.collectAsState()
    val selectedLens by cameraManager.currentLens.collectAsState()

    // A full-screen settings surface should be operationally full-screen too.
    // Stop CameraX and motion sources instead of spending GPU/CPU time behind an
    // opaque page. The preview is rebound to a fresh surface when returning.
    LaunchedEffect(
        showSettingsPage,
        showModelSelector,
        cameraManager,
        viewModel,
        lifecycleOwner,
        context,
    ) {
        viewModel.setInferencePaused(showSettingsPage || showModelSelector)
        if (showSettingsPage) {
            cameraStarted = false
            cameraManager.stopCamera()
            viewModel.stopMotionEstimation()
            viewModel.stopGpsSpeed()
        } else if (lifecycleOwner.lifecycle.currentState.isAtLeast(Lifecycle.State.STARTED)) {
            viewModel.startMotionEstimation()
            if (ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION)
                == PackageManager.PERMISSION_GRANTED
            ) {
                viewModel.startGpsSpeed()
            }
        }
    }

    // Cleanup camera when composable leaves composition
    DisposableEffect(cameraManager) {
        onDispose {
            cameraManager.release()
        }
    }

    NightVisionTheme {
        if (!showSettingsPage) {
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

            PerceptionLayers(viewModel = viewModel, settings = settings, portrait = portrait)

            VisionStatusChip(
                modelName = currentModelName,
                onClick = { showModelSelector = true },
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .statusBarsPadding()
                    .padding(start = edgePadding, top = 8.dp)
                    .widthIn(max = if (portrait) 142.dp else 180.dp),
            )

            PerformanceLayers(
                viewModel = viewModel,
                currentModelName = currentModelName,
                showPerformancePanel = showPerformancePanel,
                edgePadding = edgePadding,
                onTogglePanel = { showPerformancePanel = !showPerformancePanel },
            )
            SpeedLayer(viewModel, settings.speedStyle, edgePadding)

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
                onSelectModel = { showModelSelector = true },
                int8Available = viewModel.isInt8Available(),
                onQuantizationSelected = viewModel::selectQuantization,
                onDismiss = { showSettingsPage = false },
                onPreviewAlert = { severity -> viewModel.previewAlertSound(severity) },
            )
        }

        // ── Model Selector Bottom Sheet ──
        if (showModelSelector) {
            val availableModels = remember(viewModel) { viewModel.getAvailableModels() }
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
