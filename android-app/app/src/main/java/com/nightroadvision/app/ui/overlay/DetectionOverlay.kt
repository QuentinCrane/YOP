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
import com.nightroadvision.app.inference.InferenceEngine

/**
 * DetectionOverlay renders detection boxes on top of the camera preview.
 *
 * Coordinate mapping strategy:
 *   - Detections from InferenceEngine are in CAMERA coordinate space (1280x720).
 *   - This composable maps them to SCREEN coordinates, accounting for the fact
 *     that the camera preview (16:9) may be letterboxed or pillarboxed to fit
 *     the actual screen aspect ratio.
 *
 * @param detections List of detections in camera coordinate space
 * @param cameraWidth Width of the camera frame (default 1280)
 * @param cameraHeight Height of the camera frame (default 720)
 * @param modifier Modifier for the Canvas
 */
@Composable
fun DetectionOverlay(
    detections: List<InferenceEngine.Detection>,
    cameraWidth: Int = InferenceEngine.CAMERA_WIDTH,
    cameraHeight: Int = InferenceEngine.CAMERA_HEIGHT,
    modifier: Modifier = Modifier
) {
    val density = LocalDensity.current.density

    // Pre-create and remember Paint objects to avoid allocation per draw call
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

        // Calculate the scale and offset to map camera coords -> overlay coords
        // while preserving aspect ratio (same logic as the camera preview scaling).
        val cameraAspect = cameraWidth.toFloat() / cameraHeight.toFloat()
        val overlayAspect = overlayWidth / overlayHeight

        val scale: Float
        val offsetX: Float
        val offsetY: Float

        if (cameraAspect > overlayAspect) {
            // Camera is wider than overlay: fit by width, letterbox top/bottom
            scale = overlayWidth / cameraWidth
            offsetX = 0f
            offsetY = (overlayHeight - cameraHeight * scale) / 2f
        } else {
            // Camera is taller than overlay: fit by height, pillarbox left/right
            scale = overlayHeight / cameraHeight
            offsetX = (overlayWidth - cameraWidth * scale) / 2f
            offsetY = 0f
        }

        for (detection in detections) {
            drawDetectionBox(
                detection = detection,
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
 * Draw a single detection box mapped from camera coordinates to screen coordinates.
 *
 * Mapping formula (aspect-ratio-preserving):
 *   screenX = cameraX * scale + offsetX
 *   screenY = cameraY * scale + offsetY
 */
private fun DrawScope.drawDetectionBox(
    detection: InferenceEngine.Detection,
    scale: Float,
    offsetX: Float,
    offsetY: Float,
    overlayWidth: Float,
    overlayHeight: Float,
    density: Float,
    bgPaint: android.graphics.Paint,
    textPaint: android.graphics.Paint
) {
    // Map camera coordinates to screen/overlay coordinates
    val screenX1 = detection.x1 * scale + offsetX
    val screenY1 = detection.y1 * scale + offsetY
    val screenX2 = detection.x2 * scale + offsetX
    val screenY2 = detection.y2 * scale + offsetY

    val boxWidth = screenX2 - screenX1
    val boxHeight = screenY2 - screenY1

    // Skip degenerate boxes
    if (boxWidth <= 0f || boxHeight <= 0f) return

    // Choose color based on class ID (use floorMod to handle negative classId)
    val boxColor = getClassColor(detection.classId)

    // Draw bounding box outline
    drawRect(
        color = boxColor,
        topLeft = Offset(screenX1, screenY1),
        size = Size(boxWidth, boxHeight),
        style = Stroke(width = 3f * density)
    )

    // Draw label background -- clamp to ensure label stays on screen
    val label = "${getClassName(detection)} ${(detection.confidence * 100).toInt()}%"
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
    // Use floorMod to ensure non-negative index
    val index = Math.floorMod(classId, colors.size)
    return colors[index]
}

private fun getClassName(detection: InferenceEngine.Detection): String {
    return detection.className.ifEmpty { "cls_${detection.classId}" }
}
