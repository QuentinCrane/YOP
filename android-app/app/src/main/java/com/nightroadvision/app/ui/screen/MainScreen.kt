package com.nightroadvision.app.ui.screen

import androidx.compose.animation.AnimatedVisibility
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
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
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
import com.nightroadvision.app.ui.overlay.DetectionOverlay

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

// ──────────────────────────────────────────────────────────────────────────────
// Detection mode enum (shared with ViewModel)
// ──────────────────────────────────────────────────────────────────────────────

enum class DetectionMode(val label: String, val description: String) {
    ECO("ECO", "Minimal processing, best battery life"),
    BALANCED("BALANCED", "Recommended for most scenarios"),
    FINE("FINE", "Maximum accuracy, higher power usage")
}

// ──────────────────────────────────────────────────────────────────────────────
// Data classes used by ViewModel (defined here for same-package access)
// ──────────────────────────────────────────────────────────────────────────────

enum class BackendPreference(val label: String) {
    AUTO("Auto"),
    GPU("GPU"),
    NNAPI("NNAPI"),
    CPU("CPU")
}

enum class GpuPrecision(val label: String) {
    FP16("FP16 (Fast)"),
    FP32("FP32 (Precise)")
}

data class InferenceSettings(
    val confidenceThreshold: Float = 0.25f,
    val iouThreshold: Float = 0.45f,
    val detectionMode: DetectionMode = DetectionMode.BALANCED,
    val frameSkip: Int = 1,
    val selectedModelId: String = "yolov8n",
    val backendPreference: BackendPreference = BackendPreference.AUTO,
    val gpuPrecision: GpuPrecision = GpuPrecision.FP16
)

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
                    text = "SELECT MODEL",
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
                                text = "${model.parameterCount} params",
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
                    contentDescription = "Selected",
                    tint = NightVisionColors.Accent,
                    modifier = Modifier.size(24.dp)
                )
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// 2. SETTINGS DIALOG
// ──────────────────────────────────────────────────────────────────────────────

@Composable
fun SettingsDialog(
    settings: InferenceSettings,
    currentModelName: String,
    onSettingsChanged: (InferenceSettings) -> Unit,
    onOpenModelSelector: () -> Unit,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = NightVisionColors.Surface,
        titleContentColor = NightVisionColors.Accent,
        textContentColor = NightVisionColors.Text,
        title = {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector = Icons.Filled.Settings,
                    contentDescription = null,
                    tint = NightVisionColors.Accent,
                    modifier = Modifier.size(22.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = "INFERENCE SETTINGS",
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 1.5.sp,
                    fontSize = 16.sp
                )
            }
        },
        text = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
                    .padding(vertical = 4.dp),
                verticalArrangement = Arrangement.spacedBy(20.dp)
            ) {
                // ── Model Selection Button ──
                SettingsSection(label = "MODEL") {
                    Surface(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable(onClick = onOpenModelSelector),
                        shape = RoundedCornerShape(8.dp),
                        color = NightVisionColors.Background,
                        border = BorderStroke(1.dp, NightVisionColors.Border)
                    ) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(14.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Column {
                                Text(
                                    text = currentModelName,
                                    style = MaterialTheme.typography.bodyMedium,
                                    fontWeight = FontWeight.SemiBold,
                                    color = NightVisionColors.Accent
                                )
                                Text(
                                    text = "Tap to change model",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = NightVisionColors.TextMuted
                                )
                            }
                            Text(
                                text = "\u203A",
                                color = NightVisionColors.TextSecondary,
                                fontWeight = FontWeight.Bold,
                                fontSize = 18.sp
                            )
                        }
                    }
                }

                // ── Confidence Threshold Slider ──
                SettingsSection(label = "CONFIDENCE THRESHOLD") {
                    Column {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(
                                text = "Min detection confidence",
                                style = MaterialTheme.typography.bodySmall,
                                color = NightVisionColors.TextSecondary
                            )
                            Text(
                                text = String.format("%.2f", settings.confidenceThreshold),
                                style = MaterialTheme.typography.bodySmall,
                                fontWeight = FontWeight.Bold,
                                color = NightVisionColors.Accent,
                                fontFamily = FontFamily.Monospace
                            )
                        }
                        Spacer(modifier = Modifier.height(4.dp))
                        Slider(
                            value = settings.confidenceThreshold,
                            onValueChange = {
                                onSettingsChanged(settings.copy(confidenceThreshold = it))
                            },
                            valueRange = 0.1f..0.9f,
                            steps = 7,
                            colors = SliderDefaults.colors(
                                thumbColor      = NightVisionColors.Accent,
                                activeTrackColor  = NightVisionColors.Accent,
                                inactiveTrackColor = NightVisionColors.Border,
                                activeTickColor    = NightVisionColors.Accent.copy(alpha = 0.5f),
                                inactiveTickColor  = NightVisionColors.TextMuted
                            )
                        )
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text("0.10", fontSize = 10.sp, color = NightVisionColors.TextMuted)
                            Text("0.90", fontSize = 10.sp, color = NightVisionColors.TextMuted)
                        }
                    }
                }

                // ── IoU Threshold Slider ──
                SettingsSection(label = "IoU THRESHOLD (NMS)") {
                    Column {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(
                                text = "Overlap suppression threshold",
                                style = MaterialTheme.typography.bodySmall,
                                color = NightVisionColors.TextSecondary
                            )
                            Text(
                                text = String.format("%.2f", settings.iouThreshold),
                                style = MaterialTheme.typography.bodySmall,
                                fontWeight = FontWeight.Bold,
                                color = NightVisionColors.Accent,
                                fontFamily = FontFamily.Monospace
                            )
                        }
                        Spacer(modifier = Modifier.height(4.dp))
                        Slider(
                            value = settings.iouThreshold,
                            onValueChange = {
                                onSettingsChanged(settings.copy(iouThreshold = it))
                            },
                            valueRange = 0.1f..0.9f,
                            steps = 7,
                            colors = SliderDefaults.colors(
                                thumbColor      = NightVisionColors.Accent,
                                activeTrackColor  = NightVisionColors.Accent,
                                inactiveTrackColor = NightVisionColors.Border,
                                activeTickColor    = NightVisionColors.Accent.copy(alpha = 0.5f),
                                inactiveTickColor  = NightVisionColors.TextMuted
                            )
                        )
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text("0.10", fontSize = 10.sp, color = NightVisionColors.TextMuted)
                            Text("0.90", fontSize = 10.sp, color = NightVisionColors.TextMuted)
                        }
                    }
                }

                // ── Detection Mode Selector ──
                SettingsSection(label = "DETECTION MODE") {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        DetectionMode.entries.forEach { mode ->
                            val isSelected = settings.detectionMode == mode
                            val chipColor = if (isSelected) {
                                NightVisionColors.Accent.copy(alpha = 0.15f)
                            } else {
                                NightVisionColors.Background
                            }
                            val chipBorder = if (isSelected) {
                                NightVisionColors.Accent
                            } else {
                                NightVisionColors.Border
                            }

                            Surface(
                                modifier = Modifier
                                    .weight(1f)
                                    .clickable {
                                        onSettingsChanged(settings.copy(detectionMode = mode))
                                    },
                                shape = RoundedCornerShape(8.dp),
                                color = chipColor,
                                border = BorderStroke(1.dp, chipBorder)
                            ) {
                                Column(
                                    modifier = Modifier.padding(vertical = 10.dp, horizontal = 8.dp),
                                    horizontalAlignment = Alignment.CenterHorizontally
                                ) {
                                    Text(
                                        text = mode.label,
                                        style = MaterialTheme.typography.labelMedium,
                                        fontWeight = FontWeight.Bold,
                                        color = if (isSelected) NightVisionColors.Accent
                                                else NightVisionColors.TextSecondary
                                    )
                                    Spacer(modifier = Modifier.height(2.dp))
                                    Text(
                                        text = mode.description,
                                        style = MaterialTheme.typography.labelSmall,
                                        fontSize = 9.sp,
                                        color = NightVisionColors.TextMuted,
                                        maxLines = 2
                                    )
                                }
                            }
                        }
                    }
                }

                // ── Frame Skip Control ──
                SettingsSection(label = "FRAME SKIP") {
                    Column {
                        Text(
                            text = "Process every Nth frame (higher = longer battery, fewer detections)",
                            style = MaterialTheme.typography.bodySmall,
                            color = NightVisionColors.TextSecondary
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            listOf(1, 2, 3).forEach { skip ->
                                val isSelected = settings.frameSkip == skip
                                val chipColor = if (isSelected) {
                                    NightVisionColors.Accent.copy(alpha = 0.15f)
                                } else {
                                    NightVisionColors.Background
                                }
                                val chipBorder = if (isSelected) {
                                    NightVisionColors.Accent
                                } else {
                                    NightVisionColors.Border
                                }

                                Surface(
                                    modifier = Modifier
                                        .weight(1f)
                                        .clickable {
                                            onSettingsChanged(settings.copy(frameSkip = skip))
                                        },
                                    shape = RoundedCornerShape(8.dp),
                                    color = chipColor,
                                    border = BorderStroke(1.dp, chipBorder)
                                ) {
                                    Column(
                                        modifier = Modifier.padding(vertical = 10.dp),
                                        horizontalAlignment = Alignment.CenterHorizontally
                                    ) {
                                        Text(
                                            text = "1/$skip",
                                            style = MaterialTheme.typography.titleMedium,
                                            fontWeight = FontWeight.Bold,
                                            fontFamily = FontFamily.Monospace,
                                            color = if (isSelected) NightVisionColors.Accent
                                                    else NightVisionColors.TextSecondary
                                        )
                                        Text(
                                            text = when (skip) {
                                                1 -> "Every frame"
                                                2 -> "Every 2nd"
                                                3 -> "Every 3rd"
                                                else -> ""
                                            },
                                            fontSize = 10.sp,
                                            color = NightVisionColors.TextMuted
                                        )
                                    }
                                }
                            }
                        }
                    }
                }

                // ── Backend Selection ──
                SettingsSection(label = "BACKEND") {
                    Column {
                        Text(
                            text = "Inference delegate (Auto = GPU → NNAPI → CPU fallback)",
                            style = MaterialTheme.typography.bodySmall,
                            color = NightVisionColors.TextSecondary
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            BackendPreference.entries.forEach { backend ->
                                val isSelected = settings.backendPreference == backend
                                val chipColor = if (isSelected) {
                                    NightVisionColors.Accent.copy(alpha = 0.15f)
                                } else {
                                    NightVisionColors.Background
                                }
                                val chipBorder = if (isSelected) {
                                    NightVisionColors.Accent
                                } else {
                                    NightVisionColors.Border
                                }

                                Surface(
                                    modifier = Modifier
                                        .weight(1f)
                                        .clickable {
                                            onSettingsChanged(settings.copy(backendPreference = backend))
                                        },
                                    shape = RoundedCornerShape(8.dp),
                                    color = chipColor,
                                    border = BorderStroke(1.dp, chipBorder)
                                ) {
                                    Column(
                                        modifier = Modifier.padding(vertical = 10.dp, horizontal = 4.dp),
                                        horizontalAlignment = Alignment.CenterHorizontally
                                    ) {
                                        Text(
                                            text = backend.label,
                                            style = MaterialTheme.typography.labelMedium,
                                            fontWeight = FontWeight.Bold,
                                            color = if (isSelected) NightVisionColors.Accent
                                                    else NightVisionColors.TextSecondary
                                        )
                                    }
                                }
                            }
                        }
                    }
                }

                // ── GPU Precision ──
                SettingsSection(label = "GPU PRECISION") {
                    Column {
                        Text(
                            text = "FP16 is 2x faster on Adreno 750; FP32 is more accurate",
                            style = MaterialTheme.typography.bodySmall,
                            color = NightVisionColors.TextSecondary
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            GpuPrecision.entries.forEach { precision ->
                                val isSelected = settings.gpuPrecision == precision
                                val chipColor = if (isSelected) {
                                    NightVisionColors.Accent.copy(alpha = 0.15f)
                                } else {
                                    NightVisionColors.Background
                                }
                                val chipBorder = if (isSelected) {
                                    NightVisionColors.Accent
                                } else {
                                    NightVisionColors.Border
                                }

                                Surface(
                                    modifier = Modifier
                                        .weight(1f)
                                        .clickable {
                                            onSettingsChanged(settings.copy(gpuPrecision = precision))
                                        },
                                    shape = RoundedCornerShape(8.dp),
                                    color = chipColor,
                                    border = BorderStroke(1.dp, chipBorder)
                                ) {
                                    Column(
                                        modifier = Modifier.padding(vertical = 10.dp, horizontal = 4.dp),
                                        horizontalAlignment = Alignment.CenterHorizontally
                                    ) {
                                        Text(
                                            text = precision.label,
                                            style = MaterialTheme.typography.labelMedium,
                                            fontWeight = FontWeight.Bold,
                                            color = if (isSelected) NightVisionColors.Accent
                                                    else NightVisionColors.TextSecondary
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        confirmButton = {
            Button(
                onClick = onDismiss,
                colors = ButtonDefaults.buttonColors(
                    containerColor = NightVisionColors.Accent,
                    contentColor = NightVisionColors.Background
                ),
                shape = RoundedCornerShape(8.dp)
            ) {
                Text("DONE", fontWeight = FontWeight.Bold, letterSpacing = 1.sp)
            }
        }
    )
}

@Composable
private fun SettingsSection(
    label: String,
    content: @Composable ColumnScope.() -> Unit
) {
    Column {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            fontWeight = FontWeight.Bold,
            letterSpacing = 1.5.sp,
            color = NightVisionColors.TextMuted,
            modifier = Modifier.padding(bottom = 8.dp)
        )
        content()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
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
                text = "TELEMETRY",
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
                label = "MODEL",
                value = modelName,
                valueColor = NightVisionColors.Accent
            )

            // FPS
            MetricRow(
                label = "FPS",
                value = String.format("%.1f", metrics.fps),
                valueColor = when {
                    metrics.fps >= 24f -> NightVisionColors.Success
                    metrics.fps >= 15f -> NightVisionColors.Warning
                    else -> NightVisionColors.Danger
                }
            )

            // Inference latency
            MetricRow(
                label = "LATENCY",
                value = "${metrics.inferenceLatencyMs.toInt()} ms",
                valueColor = when {
                    metrics.inferenceLatencyMs <= 30f -> NightVisionColors.Success
                    metrics.inferenceLatencyMs <= 60f -> NightVisionColors.Warning
                    else -> NightVisionColors.Danger
                }
            )

            // Detection count
            MetricRow(
                label = "DETECTS",
                value = "${metrics.detectionCount}",
                valueColor = if (metrics.detectionCount > 0) NightVisionColors.Accent
                             else NightVisionColors.TextSecondary
            )

            // Backend
            MetricRow(
                label = "BACKEND",
                value = metrics.backend,
                valueColor = if (metrics.backend.contains("GPU")) NightVisionColors.Accent
                             else NightVisionColors.TextSecondary
            )

            // Thermal status
            MetricRow(
                label = "THERMAL",
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
        modifier = modifier.size(44.dp),
        shape = CircleShape,
        color = NightVisionColors.Background.copy(alpha = 0.85f),
        border = BorderStroke(1.dp, NightVisionColors.Border.copy(alpha = 0.6f))
    ) {
        Box(contentAlignment = Alignment.Center) {
            icon()
        }
    }
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
    viewModel: MainScreenViewModel = viewModel(),
    modifier: Modifier = Modifier
) {
    val lifecycleOwner = LocalLifecycleOwner.current

    // ViewModel state
    val detections by viewModel.detections.collectAsState()
    val metrics by viewModel.performanceMetrics.collectAsState()
    val settings by viewModel.settings.collectAsState()
    val currentModelName by viewModel.currentModelName.collectAsState()
    val errorMessage by viewModel.errorMessage.collectAsState()
    val activeDelegate by viewModel.activeDelegate.collectAsState()

    // Local UI state
    var showSettingsDialog by remember { mutableStateOf(false) }
    var showModelSelector by remember { mutableStateOf(false) }
    var showPerformancePanel by remember { mutableStateOf(true) }
    var flashlightOn by remember { mutableStateOf(false) }

    // Camera setup - runs once on first composition
    val cameraManager = remember {
        viewModel.createCameraManager(lifecycleOwner)
    }
    var cameraStarted by remember { mutableStateOf(false) }

    // Cleanup camera when composable leaves composition
    DisposableEffect(Unit) {
        onDispose {
            cameraManager.stopCamera()
        }
    }

    NightVisionTheme {
        Box(
            modifier = modifier
                .fillMaxSize()
                .background(NightVisionColors.Background)
        ) {
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
                    if (!cameraStarted) {
                        cameraStarted = true
                        cameraManager.startCamera(previewView.surfaceProvider)
                    }
                },
                modifier = Modifier.fillMaxSize()
            )

            // ── Detection overlay ──
            DetectionOverlay(
                detections = detections,
                cameraWidth = viewModel.getCameraFrameWidth(),
                cameraHeight = viewModel.getCameraFrameHeight(),
                sensorRotationDegrees = viewModel.getSensorRotationDegrees(),
                modifier = Modifier.fillMaxSize()
            )

            // ── Performance Panel (top-left) ──
            AnimatedVisibility(
                visible = showPerformancePanel,
                enter = fadeIn() + slideInHorizontally(initialOffsetX = { -it }),
                exit = fadeOut() + slideOutHorizontally(targetOffsetX = { -it }),
                modifier = Modifier
                    .statusBarsPadding()
                    .padding(start = 12.dp, top = 12.dp)
                    .align(Alignment.TopStart)
                    .widthIn(min = 160.dp, max = 200.dp)
            ) {
                PerformancePanel(
                    modelName = currentModelName,
                    metrics = metrics
                )
            }

            // ── Bottom toolbar ──
            Row(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .navigationBarsPadding()
                    .padding(bottom = 16.dp)
                    .clip(RoundedCornerShape(24.dp))
                    .background(NightVisionColors.Background.copy(alpha = 0.85f))
                    .border(1.dp, NightVisionColors.Border.copy(alpha = 0.6f), RoundedCornerShape(24.dp))
                    .padding(horizontal = 12.dp, vertical = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Flashlight toggle
                CircleIconButton(
                    onClick = {
                        flashlightOn = viewModel.toggleFlashlight()
                    },
                    icon = {
                        Text(
                            text = if (flashlightOn) "\u26A1" else "\u25CB",
                            fontSize = 18.sp,
                            color = if (flashlightOn) NightVisionColors.Warning
                                    else NightVisionColors.TextSecondary
                        )
                    }
                )

                // Telemetry toggle
                CircleIconButton(
                    onClick = { showPerformancePanel = !showPerformancePanel },
                    icon = {
                        Text(
                            text = "T",
                            color = if (showPerformancePanel) NightVisionColors.Accent
                                    else NightVisionColors.TextSecondary,
                            fontWeight = FontWeight.Bold,
                            fontSize = 14.sp,
                            fontFamily = FontFamily.Monospace
                        )
                    }
                )

                // Detection count badge
                Surface(
                    shape = RoundedCornerShape(12.dp),
                    color = if (metrics.detectionCount > 0) NightVisionColors.Accent.copy(alpha = 0.15f)
                            else NightVisionColors.SurfaceVariant
                ) {
                    Text(
                        text = "${metrics.detectionCount} obj",
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.Bold,
                        color = if (metrics.detectionCount > 0) NightVisionColors.Accent
                                else NightVisionColors.TextMuted,
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp)
                    )
                }

                // Settings button
                CircleIconButton(
                    onClick = { showSettingsDialog = true },
                    icon = {
                        Icon(
                            imageVector = Icons.Filled.Settings,
                            contentDescription = "Settings",
                            tint = NightVisionColors.TextSecondary,
                            modifier = Modifier.size(20.dp)
                        )
                    }
                )
            }

            // ── Error Snackbar ──
            errorMessage?.let { error ->
                Snackbar(
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .navigationBarsPadding()
                        .padding(bottom = 80.dp, start = 16.dp, end = 16.dp),
                    containerColor = NightVisionColors.Danger.copy(alpha = 0.9f),
                    contentColor = NightVisionColors.Text,
                    dismissAction = {
                        TextButton(onClick = { viewModel.clearError() }) {
                            Text("DISMISS", color = NightVisionColors.Accent, fontWeight = FontWeight.Bold)
                        }
                    },
                    action = {
                        TextButton(onClick = { viewModel.clearError() }) {
                            Text("OK", color = NightVisionColors.Accent, fontWeight = FontWeight.Bold)
                        }
                    }
                ) {
                    Text(text = error, style = MaterialTheme.typography.bodySmall)
                }
            }
        }

        // ── Settings Dialog ──
        if (showSettingsDialog) {
            SettingsDialog(
                settings = settings,
                currentModelName = currentModelName,
                onSettingsChanged = { newSettings ->
                    viewModel.updateSettings(newSettings)
                },
                onOpenModelSelector = {
                    showSettingsDialog = false
                    showModelSelector = true
                },
                onDismiss = { showSettingsDialog = false }
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
                    showSettingsDialog = true
                },
                onDismiss = { showModelSelector = false }
            )
        }
    }
}
