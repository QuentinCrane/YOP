package com.nightroadvision.app.ui.overlay

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalDensity
import com.nightroadvision.app.FileLogger
import com.nightroadvision.app.inference.InferenceEngine

private enum class TargetProximity(val label: String) {
    FAR("远"),
    MEDIUM("中"),
    NEAR("近"),
}

/**
 * DetectionOverlay renders detection boxes on top of the camera preview.
 *
 * Coordinate mapping strategy:
 *   - Detections from InferenceEngine are in CAMERA SENSOR coordinate space.
 *   - The camera sensor is typically landscape (e.g. 1280x720 or 2448x2448).
 *   - PreviewView rotates the preview by [sensorRotationDegrees] for portrait display.
 *   - This overlay applies the SAME rotation to detection coordinates,
 *     then uses FILL_CENTER to match the visible preview content.
 *
 * @param detections List of detections in camera sensor coordinate space
 * @param cameraWidth Width of the camera sensor frame (before rotation)
 * @param cameraHeight Height of the camera sensor frame (before rotation)
 * @param sensorRotationDegrees Camera sensor rotation (0, 90, 180, or 270)
 * @param modifier Modifier for the Canvas
 */
@Composable
fun DetectionOverlay(
    detections: List<InferenceEngine.Detection>,
    highlightedTrackId: Int? = null,
    cameraWidth: Int = InferenceEngine.DEFAULT_CAMERA_WIDTH,
    cameraHeight: Int = InferenceEngine.DEFAULT_CAMERA_HEIGHT,
    sensorRotationDegrees: Int = 90,
    modifier: Modifier = Modifier
) {
    val density = LocalDensity.current.density

    // Pre-create and remember Paint objects (keyed on density)
    val bgPaint = remember(density) {
        android.graphics.Paint().apply {
            color = android.graphics.Color.argb(165, 0, 0, 0)
            isAntiAlias = true
            textSize = 10f * density
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

        // Guard: skip if canvas has zero size (first layout pass)
        if (overlayWidth <= 0f || overlayHeight <= 0f) {
            FileLogger.w("DetectionOverlay", "Canvas has zero size, skipping")
            return@Canvas
        }

        // After rotation, the effective camera dimensions change:
        // 90° or 270°: width and height swap
        // 0° or 180°: same
        val isRotated = (sensorRotationDegrees == 90 || sensorRotationDegrees == 270)
        val effectiveCamWidth = if (isRotated) cameraHeight else cameraWidth
        val effectiveCamHeight = if (isRotated) cameraWidth else cameraHeight

        // FILL_CENTER logic: scale to fill, crop excess
        val cameraAspect = effectiveCamWidth.toFloat() / effectiveCamHeight.toFloat()
        val overlayAspect = overlayWidth / overlayHeight

        val scale: Float
        val offsetX: Float
        val offsetY: Float

        if (cameraAspect > overlayAspect) {
            scale = overlayHeight / effectiveCamHeight
            offsetX = (overlayWidth - effectiveCamWidth * scale) / 2f
            offsetY = 0f
        } else {
            scale = overlayWidth / effectiveCamWidth
            offsetX = 0f
            offsetY = (overlayHeight - effectiveCamHeight * scale) / 2f
        }

        // Log pipeline state every time (throttled by frame rate anyway)
        FileLogger.d("DetectionOverlay", "Pipeline: detections=${detections.size}, " +
            "canvas=${overlayWidth}x${overlayHeight}, " +
            "sensor=${cameraWidth}x${cameraHeight}@${sensorRotationDegrees}°, " +
            "effective=${effectiveCamWidth}x${effectiveCamHeight}, " +
            "scale=$scale, offset=($offsetX, $offsetY), " +
            "cameraAspect=$cameraAspect, overlayAspect=$overlayAspect")

        if (detections.isNotEmpty()) {
            val d = detections[0]
            // Apply rotation to first detection for logging
            val (rx1, ry1, rx2, ry2) = rotateCoords(
                d.x1, d.y1, d.x2, d.y2,
                sensorRotationDegrees, cameraWidth, cameraHeight
            )
            val sx1 = rx1 * scale + offsetX
            val sy1 = ry1 * scale + offsetY
            val sx2 = rx2 * scale + offsetX
            val sy2 = ry2 * scale + offsetY
            FileLogger.d("DetectionOverlay", "First detection: " +
                "cam=(${d.x1.toInt()},${d.y1.toInt()})-(${d.x2.toInt()},${d.y2.toInt()}) " +
                "rotated=(${rx1.toInt()},${ry1.toInt()})-(${rx2.toInt()},${ry2.toInt()}) " +
                "screen=(${sx1.toInt()},${sy1.toInt()})-(${sx2.toInt()},${sy2.toInt()})")
        }

        for (detection in detections) {
            drawDetectionBox(
                detection = detection,
                highlighted = detection.trackId != null && detection.trackId == highlightedTrackId,
                sensorRotationDegrees = sensorRotationDegrees,
                cameraWidth = cameraWidth,
                cameraHeight = cameraHeight,
                scale = scale,
                offsetX = offsetX,
                offsetY = offsetY,
                overlayWidth = overlayWidth,
                overlayHeight = overlayHeight,
                density = density,
                bgPaint = bgPaint,
                textPaint = textPaint
            )
        }
    }
}

/**
 * Rotate camera sensor coordinates to display coordinates.
 *
 * Camera sensor is typically landscape. PreviewView rotates the preview
 * by [degrees] for portrait display. We must apply the same rotation
 * to detection coordinates so boxes align with the preview.
 *
 * @return Rotated (x1, y1, x2, y2) in display coordinate space
 */
private fun rotateCoords(
    x1: Float, y1: Float, x2: Float, y2: Float,
    degrees: Int, camWidth: Int, camHeight: Int
): FloatArray {
    return when (degrees) {
        90 -> {
            // 90° CW: (x,y) → (y, W-x), effective space is (H, W)
            floatArrayOf(y1, camWidth - x2, y2, camWidth - x1)
        }
        180 -> {
            // 180°: (x,y) → (W-x, H-y)
            floatArrayOf(camWidth - x2, camHeight - y2, camWidth - x1, camHeight - y1)
        }
        270 -> {
            // 270° CW = 90° CCW: (x,y) → (H-y, x), effective space is (H, W)
            floatArrayOf(camHeight - y2, x1, camHeight - y1, x2)
        }
        else -> {
            // 0°: no rotation
            floatArrayOf(x1, y1, x2, y2)
        }
    }
}

/**
 * Draw a single detection box mapped from camera coordinates to screen coordinates.
 */
private fun DrawScope.drawDetectionBox(
    detection: InferenceEngine.Detection,
    highlighted: Boolean,
    sensorRotationDegrees: Int,
    cameraWidth: Int,
    cameraHeight: Int,
    scale: Float,
    offsetX: Float,
    offsetY: Float,
    overlayWidth: Float,
    overlayHeight: Float,
    density: Float,
    bgPaint: android.graphics.Paint,
    textPaint: android.graphics.Paint
) {
    // Rotate camera coordinates to display coordinates
    val (rx1, ry1, rx2, ry2) = rotateCoords(
        detection.x1, detection.y1, detection.x2, detection.y2,
        sensorRotationDegrees, cameraWidth, cameraHeight
    )

    // Map rotated coordinates to screen/overlay coordinates. FILL_CENTER crops the
    // preview, so valid camera coordinates can legitimately land outside the canvas.
    val rawLeft = minOf(rx1, rx2) * scale + offsetX
    val rawTop = minOf(ry1, ry2) * scale + offsetY
    val rawRight = maxOf(rx1, rx2) * scale + offsetX
    val rawBottom = maxOf(ry1, ry2) * scale + offsetY

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

    val isRotated = sensorRotationDegrees == 90 || sensorRotationDegrees == 270
    val displayCameraWidth = if (isRotated) cameraHeight.toFloat() else cameraWidth.toFloat()
    val displayCameraHeight = if (isRotated) cameraWidth.toFloat() else cameraHeight.toFloat()
    val normalizedWidth = ((maxOf(rx1, rx2) - minOf(rx1, rx2)) / displayCameraWidth).coerceAtLeast(0f)
    val normalizedHeight = ((maxOf(ry1, ry2) - minOf(ry1, ry2)) / displayCameraHeight).coerceAtLeast(0f)
    val proximity = estimateProximity(normalizedWidth, normalizedHeight)
    val centerX = (rx1 + rx2) / 2f / displayCameraWidth
    val centerY = (ry1 + ry2) / 2f / displayCameraHeight
    val inRidingAttentionZone = centerX in 0.22f..0.78f && centerY in 0.35f..1f

    val boxColor = getClassColor(
        className = detection.className,
        proximity = proximity,
        inAttentionZone = inRidingAttentionZone,
    )

    // A low-opacity interior and thin outline are easier to read while driving than
    // the previous heavy multi-colour rectangles.
    drawRect(
        color = boxColor.copy(alpha = if (highlighted) 0.09f else 0.025f),
        topLeft = Offset(screenX1, screenY1),
        size = Size(boxWidth, boxHeight),
    )
    drawRect(
        color = boxColor.copy(alpha = if (highlighted) 0.96f else 0.62f),
        topLeft = Offset(screenX1, screenY1),
        size = Size(boxWidth, boxHeight),
        style = Stroke(width = if (highlighted) 1.8f * density else 0.8f * density),
    )

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
    val trackLabel = detection.trackId?.let { "#$it " }.orEmpty()
    val label = "$trackLabel${detection.className} ${(detection.confidence * 100).toInt()}% · ${proximity.label}"
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

    drawContext.canvas.nativeCanvas.drawRect(
        labelLeft,
        labelTop,
        labelLeft + labelWidth,
        labelBottom,
        bgPaint
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
): Color {
    if (inAttentionZone && proximity == TargetProximity.NEAR) return Color(0xFFFF4D5E)
    if (inAttentionZone && proximity == TargetProximity.MEDIUM) return Color(0xFFFFC857)

    return when (className.lowercase()) {
        "person", "rider" -> Color(0xFF36D9FF)
        "bicycle", "motorcycle", "car", "bus", "truck" -> Color(0xFF53FF8A)
        else -> Color(0xFFD8E2EA)
    }
}
