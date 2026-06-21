package com.nightroadvision.app.ui.overlay

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalDensity
import com.nightroadvision.app.FileLogger
import com.nightroadvision.app.inference.InferenceEngine

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
    cameraWidth: Int = InferenceEngine.DEFAULT_CAMERA_WIDTH,
    cameraHeight: Int = InferenceEngine.DEFAULT_CAMERA_HEIGHT,
    sensorRotationDegrees: Int = 90,
    modifier: Modifier = Modifier
) {
    val density = LocalDensity.current.density

    // Pre-create and remember Paint objects (keyed on density)
    val bgPaint = remember(density) {
        android.graphics.Paint().apply {
            color = android.graphics.Color.argb(200, 0, 0, 0)
            isAntiAlias = true
            textSize = 11f * density // 11sp in pixels
        }
    }
    val textPaint = remember(density) {
        android.graphics.Paint().apply {
            color = android.graphics.Color.WHITE
            isAntiAlias = true
            textSize = 11f * density // 11sp in pixels
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

    // Map rotated coordinates to screen/overlay coordinates
    val screenX1 = rx1 * scale + offsetX
    val screenY1 = ry1 * scale + offsetY
    val screenX2 = rx2 * scale + offsetX
    val screenY2 = ry2 * scale + offsetY

    val boxWidth = screenX2 - screenX1
    val boxHeight = screenY2 - screenY1

    // Skip degenerate boxes
    if (boxWidth <= 0f || boxHeight <= 0f) return

    // Choose color based on class ID
    val boxColor = getClassColor(detection.classId)

    // Draw bounding box outline
    drawRect(
        color = boxColor,
        topLeft = Offset(screenX1, screenY1),
        size = Size(boxWidth, boxHeight),
        style = Stroke(width = 3f * density)
    )

    // Draw label background
    val label = "${detection.className} ${(detection.confidence * 100).toInt()}%"
    val textWidth = bgPaint.measureText(label)
    val textHeight = bgPaint.textSize
    val labelPadding = 4f * density

    // Clamp label Y so it doesn't go above the top edge
    val labelTop = (screenY1 - textHeight - labelPadding).coerceAtLeast(0f)
    val labelBottom = labelTop + textHeight + labelPadding

    drawContext.canvas.nativeCanvas.drawRect(
        screenX1,
        labelTop,
        (screenX1 + textWidth + 2f * labelPadding).coerceAtMost(overlayWidth),
        labelBottom,
        bgPaint
    )

    // Draw label text
    drawContext.canvas.nativeCanvas.drawText(
        label,
        screenX1 + labelPadding,
        labelBottom - labelPadding,
        textPaint
    )
}

private fun getClassColor(classId: Int): Color {
    val colors = listOf(
        Color(0xFFFF0000), // Red
        Color(0xFF00FF00), // Green
        Color(0xFF0000FF), // Blue
        Color(0xFFFFFF00), // Yellow
        Color(0xFFFF00FF), // Magenta
        Color(0xFF00FFFF), // Cyan
        Color(0xFFFF8800), // Orange
        Color(0xFF8800FF), // Purple
    )
    val index = Math.floorMod(classId, colors.size)
    return colors[index]
}
