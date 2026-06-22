package com.nightroadvision.app.ui.overlay

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalDensity
import com.nightroadvision.app.inference.InferenceEngine
import java.util.Locale

/**
 * A point on the predicted driving path in vehicle frame.
 * vehicleX = forward distance (m), vehicleY = lateral offset (m).
 */
data class RoutePoint(
    val vehicleX: Float,
    val vehicleY: Float,
)

private enum class TargetProximity(val label: String) {
    FAR("远"),
    MEDIUM("中"),
    NEAR("近"),
}

/**
 * DetectionOverlay renders detection boxes on top of the camera preview.
 *
 * CameraManager rotates model input into the target display orientation. Detections
 * therefore already use upright frame coordinates; this layer only mirrors
 * PreviewView.ScaleType.FILL_CENTER scaling and cropping.
 *
 * @param detections List of detections in camera sensor coordinate space
 * @param cameraWidth Width of the upright model source frame
 * @param cameraHeight Height of the upright model source frame
 * @param modifier Modifier for the Canvas
 */
@Composable
fun DetectionOverlay(
    detections: List<InferenceEngine.Detection>,
    modifier: Modifier = Modifier,
    highlightedTrackId: Int? = null,
    cameraWidth: Int = InferenceEngine.DEFAULT_CAMERA_WIDTH,
    cameraHeight: Int = InferenceEngine.DEFAULT_CAMERA_HEIGHT,
    showLabels: Boolean = true,
    showConfidence: Boolean = true,
    showTrackIds: Boolean = false,
    routePoints: List<RoutePoint> = emptyList(),
) {
    val density = LocalDensity.current.density

    // Pre-create and remember Paint objects (keyed on density)
    val bgPaint = remember(density) {
        android.graphics.Paint().apply {
            color = android.graphics.Color.argb(165, 0, 0, 0)
            isAntiAlias = true
            textSize = 10f * density
            typeface = android.graphics.Typeface.DEFAULT_BOLD
        }
    }
    val textPaint = remember(density) {
        android.graphics.Paint().apply {
            color = android.graphics.Color.WHITE
            isAntiAlias = true
            textSize = 10f * density
        }
    }

    Canvas(modifier = modifier.fillMaxSize()) {
        val overlayWidth = size.width
        val overlayHeight = size.height

        if (overlayWidth <= 0f || overlayHeight <= 0f || cameraWidth <= 0 || cameraHeight <= 0) {
            return@Canvas
        }

        // FILL_CENTER logic: scale to fill, crop excess
        val cameraAspect = cameraWidth.toFloat() / cameraHeight.toFloat()
        val overlayAspect = overlayWidth / overlayHeight

        val scale: Float
        val offsetX: Float
        val offsetY: Float

        if (cameraAspect > overlayAspect) {
            scale = overlayHeight / cameraHeight
            offsetX = (overlayWidth - cameraWidth * scale) / 2f
            offsetY = 0f
        } else {
            scale = overlayWidth / cameraWidth
            offsetX = 0f
            offsetY = (overlayHeight - cameraHeight * scale) / 2f
        }

        // ── Draw predicted driving route ──
        if (routePoints.size >= 2) {
            drawRoutePath(
                routePoints = routePoints,
                overlayWidth = overlayWidth,
                overlayHeight = overlayHeight,
                density = density,
            )
        }

        for (detection in detections) {
            drawDetectionBox(
                detection = detection,
                highlighted = detection.trackId != null && detection.trackId == highlightedTrackId,
                cameraWidth = cameraWidth,
                cameraHeight = cameraHeight,
                scale = scale,
                offsetX = offsetX,
                offsetY = offsetY,
                overlayWidth = overlayWidth,
                overlayHeight = overlayHeight,
                density = density,
                bgPaint = bgPaint,
                textPaint = textPaint,
                showLabels = showLabels,
                showConfidence = showConfidence,
                showTrackIds = showTrackIds,
            )
        }
    }
}

/**
 * Draw a single detection box mapped from camera coordinates to screen coordinates.
 */
private fun DrawScope.drawDetectionBox(
    detection: InferenceEngine.Detection,
    highlighted: Boolean,
    cameraWidth: Int,
    cameraHeight: Int,
    scale: Float,
    offsetX: Float,
    offsetY: Float,
    overlayWidth: Float,
    overlayHeight: Float,
    density: Float,
    bgPaint: android.graphics.Paint,
    textPaint: android.graphics.Paint,
    showLabels: Boolean,
    showConfidence: Boolean,
    showTrackIds: Boolean,
) {
    // Map upright coordinates to screen/overlay coordinates. FILL_CENTER crops the
    // preview, so valid camera coordinates can legitimately land outside the canvas.
    val rawLeft = minOf(detection.x1, detection.x2) * scale + offsetX
    val rawTop = minOf(detection.y1, detection.y2) * scale + offsetY
    val rawRight = maxOf(detection.x1, detection.x2) * scale + offsetX
    val rawBottom = maxOf(detection.y1, detection.y2) * scale + offsetY

    // Do not draw targets that are completely outside the visible camera preview.
    if (rawRight <= 0f || rawBottom <= 0f ||
        rawLeft >= overlayWidth || rawTop >= overlayHeight
    ) return

    // Clip boxes to the same bounds as the preview. This prevents oversized model
    // predictions and preview cropping from producing labels or strokes off-screen.
    val screenX1 = rawLeft.coerceIn(0f, overlayWidth)
    val screenY1 = rawTop.coerceIn(0f, overlayHeight)
    val screenX2 = rawRight.coerceIn(0f, overlayWidth)
    val screenY2 = rawBottom.coerceIn(0f, overlayHeight)
    val boxWidth = screenX2 - screenX1
    val boxHeight = screenY2 - screenY1

    if (boxWidth < density || boxHeight < density) return

    val displayCameraWidth = cameraWidth.toFloat()
    val displayCameraHeight = cameraHeight.toFloat()
    val normalizedWidth = (kotlin.math.abs(detection.x2 - detection.x1) / displayCameraWidth).coerceAtLeast(0f)
    val normalizedHeight = (kotlin.math.abs(detection.y2 - detection.y1) / displayCameraHeight).coerceAtLeast(0f)
    val proximity = estimateProximity(normalizedWidth, normalizedHeight)
    val centerX = (detection.x1 + detection.x2) / 2f / displayCameraWidth
    val centerY = (detection.y1 + detection.y2) / 2f / displayCameraHeight
    val inRidingAttentionZone = centerX in 0.22f..0.78f && centerY in 0.35f..1f

    val boxColor = getClassColor(
        className = detection.className,
        proximity = proximity,
        inAttentionZone = inRidingAttentionZone,
        distanceMeters = detection.distanceMeters ?: detection.estimatedDistanceMeters,
    )

    if (highlighted) {
        drawRoundRect(
            color = boxColor.copy(alpha = 0.08f),
            topLeft = Offset(screenX1, screenY1),
            size = Size(boxWidth, boxHeight),
            cornerRadius = CornerRadius(5f * density),
        )
        drawRoundRect(
            color = boxColor.copy(alpha = 0.88f),
            topLeft = Offset(screenX1, screenY1),
            size = Size(boxWidth, boxHeight),
            cornerRadius = CornerRadius(5f * density),
            style = Stroke(width = 1.4f * density),
        )
    }

    // Openpilot-inspired corner brackets keep the target clear without covering it.
    val corner = minOf(22f * density, boxWidth * 0.24f, boxHeight * 0.24f)
    val cornerStroke = if (highlighted) 3.2f * density else 2.2f * density
    drawLine(boxColor, Offset(screenX1, screenY1), Offset(screenX1 + corner, screenY1), cornerStroke)
    drawLine(boxColor, Offset(screenX1, screenY1), Offset(screenX1, screenY1 + corner), cornerStroke)
    drawLine(boxColor, Offset(screenX2, screenY1), Offset(screenX2 - corner, screenY1), cornerStroke)
    drawLine(boxColor, Offset(screenX2, screenY1), Offset(screenX2, screenY1 + corner), cornerStroke)
    drawLine(boxColor, Offset(screenX1, screenY2), Offset(screenX1 + corner, screenY2), cornerStroke)
    drawLine(boxColor, Offset(screenX1, screenY2), Offset(screenX1, screenY2 - corner), cornerStroke)
    drawLine(boxColor, Offset(screenX2, screenY2), Offset(screenX2 - corner, screenY2), cornerStroke)
    drawLine(boxColor, Offset(screenX2, screenY2), Offset(screenX2, screenY2 - corner), cornerStroke)

    if (highlighted) {
        drawLeadChevron(
            centerX = (screenX1 + screenX2) / 2f,
            targetTop = screenY1,
            proximity = proximity,
            density = density,
        )
    }

    // Draw label background
    val shouldDrawLabel = highlighted || proximity != TargetProximity.FAR ||
        (boxWidth >= 56f * density && boxHeight >= 40f * density)
    if (!shouldDrawLabel || (!showLabels && !showConfidence && !showTrackIds)) return

    val distanceLabel = (detection.distanceMeters ?: detection.estimatedDistanceMeters)
        ?.let { "${String.format(Locale.US, "%.0f", it)}m" }
        .orEmpty()
    val label = buildList {
        if (showTrackIds) detection.trackId?.let { add("#$it") }
        if (showLabels) add(detection.className.uppercase(Locale.ROOT))
        if (showConfidence) add("${(detection.confidence * 100).toInt()}%")
        if (distanceLabel.isNotEmpty()) add(distanceLabel)
    }.joinToString("  ")
    val textWidth = bgPaint.measureText(label)
    val textHeight = bgPaint.textSize
    val labelPadding = 4f * density

    val labelWidth = (textWidth + 2f * labelPadding).coerceAtMost(overlayWidth)
    val labelHeight = textHeight + labelPadding
    val labelLeft = screenX1.coerceIn(0f, (overlayWidth - labelWidth).coerceAtLeast(0f))
    val preferredTop = when {
        highlighted && screenY2 + labelHeight <= overlayHeight -> screenY2 + 2f * density
        highlighted -> screenY1 + 2f * density
        screenY1 >= labelHeight -> screenY1 - labelHeight
        else -> screenY1
    }
    val labelTop = preferredTop.coerceIn(0f, (overlayHeight - labelHeight).coerceAtLeast(0f))
    val labelBottom = labelTop + labelHeight

    drawRoundRect(
        color = Color.Black.copy(alpha = 0.68f),
        topLeft = Offset(labelLeft, labelTop),
        size = Size(labelWidth, labelHeight),
        cornerRadius = CornerRadius(4f * density),
    )

    // Draw label text
    drawContext.canvas.nativeCanvas.drawText(
        label,
        labelLeft + labelPadding,
        labelBottom - labelPadding,
        textPaint
    )
}

/**
 * openpilot-inspired lead marker: a warm outer glow with a red/orange inner
 * chevron. Unlike openpilot, the anchor comes from a tracked YOLO box rather
 * than radar-relative distance.
 */
private fun DrawScope.drawLeadChevron(
    centerX: Float,
    targetTop: Float,
    proximity: TargetProximity,
    density: Float,
) {
    val size = when (proximity) {
        TargetProximity.NEAR -> 15f * density
        TargetProximity.MEDIUM -> 12f * density
        TargetProximity.FAR -> 10f * density
    }
    val tipY = (targetTop - 7f * density).coerceAtLeast(5f * density)

    val glow = Path().apply {
        moveTo(centerX, tipY - 2f * density)
        lineTo(centerX - size * 1.45f, tipY + size * 1.12f)
        lineTo(centerX + size * 1.45f, tipY + size * 1.12f)
        close()
    }
    drawPath(glow, Color(0xFFDACA25).copy(alpha = 0.72f))

    val chevron = Path().apply {
        moveTo(centerX, tipY)
        lineTo(centerX - size * 1.20f, tipY + size)
        lineTo(centerX + size * 1.20f, tipY + size)
        close()
    }
    val fillColor = if (proximity == TargetProximity.NEAR) {
        Color(0xFFC92231)
    } else {
        Color(0xFFDA6F25)
    }
    drawPath(chevron, fillColor.copy(alpha = if (proximity == TargetProximity.FAR) 0.48f else 0.92f))
}

/**
 * Draw the predicted driving path as a glowing curve.
 *
 * Vehicle frame → screen mapping:
 *   vehicleX (forward, m) → moves up on screen (toward horizon)
 *   vehicleY (lateral, m) → moves left/right on screen
 *
 * The path starts at the bottom-center of the screen and curves upward,
 * simulating the driver's forward perspective.
 */
private fun DrawScope.drawRoutePath(
    routePoints: List<RoutePoint>,
    overlayWidth: Float,
    overlayHeight: Float,
    density: Float,
) {
    // Approximate pinhole projection. openpilot road coordinates use x forward
    // and y left, while screen x grows to the right.
    val startX = overlayWidth / 2f
    val startY = overlayHeight * 0.92f
    val horizonY = overlayHeight * 0.42f
    val focalXPixels = overlayWidth * 0.46f
    val focalYPixels = overlayHeight * 0.82f
    val cameraHeightMeters = 1.20f

    val path = Path()
    path.moveTo(startX, startY)

    for (pt in routePoints) {
        val forwardMeters = pt.vehicleX
        if (!forwardMeters.isFinite() || !pt.vehicleY.isFinite() ||
            forwardMeters !in 0.75f..80f
        ) continue

        // u = cx - fx * lateral / depth. Positive road-frame y is left.
        val screenX = startX - pt.vehicleY * focalXPixels / forwardMeters
        // Ground points approach the horizon asymptotically instead of moving
        // linearly past it as distance grows.
        val screenY = (horizonY + cameraHeightMeters * focalYPixels / forwardMeters)
            .coerceIn(horizonY, startY)

        path.lineTo(screenX, screenY)
    }

    // Outer glow (wider, semi-transparent)
    drawPath(
        path = path,
        color = Color(0xFF00FF41).copy(alpha = 0.18f),
        style = Stroke(width = 8f * density, cap = StrokeCap.Round, join = StrokeJoin.Round),
    )
    // Inner bright line
    drawPath(
        path = path,
        color = Color(0xFF00FF41).copy(alpha = 0.65f),
        style = Stroke(width = 2.5f * density, cap = StrokeCap.Round, join = StrokeJoin.Round),
    )
    // Center bright core
    drawPath(
        path = path,
        color = Color(0xFFAAFFCC).copy(alpha = 0.90f),
        style = Stroke(width = 1.2f * density, cap = StrokeCap.Round, join = StrokeJoin.Round),
    )
}

private fun estimateProximity(widthRatio: Float, heightRatio: Float): TargetProximity {
    val areaRatio = widthRatio * heightRatio
    return when {
        heightRatio >= 0.42f || areaRatio >= 0.16f -> TargetProximity.NEAR
        heightRatio >= 0.20f || areaRatio >= 0.04f -> TargetProximity.MEDIUM
        else -> TargetProximity.FAR
    }
}

private fun getClassColor(
    className: String,
    proximity: TargetProximity,
    inAttentionZone: Boolean,
    distanceMeters: Float? = null,
): Color {
    // If we have real distance from supercombo, use it for color coding
    if (distanceMeters != null) {
        return when {
            distanceMeters < 10f -> Color(0xFFFF4D5E)   // red: very close
            distanceMeters < 30f -> Color(0xFFFFC857)   // yellow: moderate
            else -> Color(0xFF53FF8A)                     // green: far
        }
    }

    if (inAttentionZone && proximity == TargetProximity.NEAR) return Color(0xFFFF4D5E)
    if (inAttentionZone && proximity == TargetProximity.MEDIUM) return Color(0xFFFFC857)

    return when (className.lowercase()) {
        "person", "rider" -> Color(0xFF36D9FF)
        "bicycle", "motorcycle", "car", "bus", "truck" -> Color(0xFF53FF8A)
        else -> Color(0xFFD8E2EA)
    }
}
