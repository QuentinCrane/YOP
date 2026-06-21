package com.nightroadvision.app.alert

import com.nightroadvision.app.inference.InferenceEngine
import kotlin.math.abs

enum class RiskSeverity {
    NONE,
    CAUTION,
    CRITICAL,
}

enum class RiskReason {
    NONE,
    IN_ATTENTION_ZONE,
    VERY_NEAR,
    APPROACHING_QUICKLY,
}

data class RidingRisk(
    val severity: RiskSeverity = RiskSeverity.NONE,
    val reason: RiskReason = RiskReason.NONE,
    val trackId: Int? = null,
    val className: String = "",
)

/**
 * Evaluates tracked people and vehicles in display space.
 *
 * This intentionally produces qualitative risk only. Bounding-box size is not an
 * accurate physical distance, but its position and growth are useful for selecting
 * a target that deserves the rider's attention.
 */
class RidingRiskEvaluator {
    companion object {
        private const val CLEAR_CONFIRMATION_FRAMES = 3
    }

    private val previousAreaByTrack = mutableMapOf<Int, Float>()
    private var lastRisk = RidingRisk()
    private var consecutiveClearFrames = 0

    fun evaluate(
        detections: List<InferenceEngine.Detection>,
        rotationDegrees: Int,
        cameraWidth: Int,
        cameraHeight: Int,
    ): RidingRisk {
        val liveTrackIds = detections.mapNotNull { it.trackId }.toSet()
        previousAreaByTrack.keys.retainAll(liveTrackIds)

        var selected = RidingRisk()
        var selectedScore = 0f

        for (detection in detections) {
            val trackId = detection.trackId ?: continue
            val geometry = displayGeometry(detection, rotationDegrees, cameraWidth, cameraHeight)
            val previousArea = previousAreaByTrack[trackId]
            val growthRatio = if (previousArea != null && previousArea > 0.002f) {
                geometry.areaRatio / previousArea
            } else {
                1f
            }
            previousAreaByTrack[trackId] = geometry.areaRatio

            val inAttentionZone = geometry.centerX in 0.22f..0.78f &&
                geometry.centerY in 0.34f..1f
            if (!inAttentionZone) continue

            val severity: RiskSeverity
            val reason: RiskReason

            // Use real distance from supercombo if available
            val dist = detection.distanceMeters
            if (dist != null) {
                when {
                    dist < 5f -> {
                        severity = RiskSeverity.CRITICAL
                        reason = RiskReason.VERY_NEAR
                    }
                    dist < 15f -> {
                        severity = RiskSeverity.CAUTION
                        reason = RiskReason.IN_ATTENTION_ZONE
                    }
                    else -> continue
                }
            } else {
                // Fallback to bbox-based heuristics
                val isVeryNear = geometry.maxDimension >= 0.42f || geometry.areaRatio >= 0.14f
                val isApproachingQuickly = growthRatio >= 1.32f && geometry.areaRatio >= 0.025f
                val isCaution = geometry.maxDimension >= 0.20f || geometry.areaRatio >= 0.035f

                when {
                    isVeryNear -> {
                        severity = RiskSeverity.CRITICAL
                        reason = RiskReason.VERY_NEAR
                    }
                    isApproachingQuickly -> {
                        severity = RiskSeverity.CRITICAL
                        reason = RiskReason.APPROACHING_QUICKLY
                    }
                    isCaution -> {
                        severity = RiskSeverity.CAUTION
                        reason = RiskReason.IN_ATTENTION_ZONE
                    }
                    else -> continue
                }
            }

            val score = severity.ordinal * 10f + geometry.maxDimension + geometry.areaRatio
            if (score > selectedScore) {
                selectedScore = score
                selected = RidingRisk(
                    severity = severity,
                    reason = reason,
                    trackId = trackId,
                    className = detection.className,
                )
            }
        }

        if (selected.severity == RiskSeverity.NONE && lastRisk.severity != RiskSeverity.NONE) {
            consecutiveClearFrames++
            if (consecutiveClearFrames < CLEAR_CONFIRMATION_FRAMES) return lastRisk
        } else {
            consecutiveClearFrames = 0
        }

        lastRisk = selected
        return selected
    }

    fun reset() {
        previousAreaByTrack.clear()
        lastRisk = RidingRisk()
        consecutiveClearFrames = 0
    }

    private fun displayGeometry(
        detection: InferenceEngine.Detection,
        rotationDegrees: Int,
        cameraWidth: Int,
        cameraHeight: Int,
    ): DisplayGeometry {
        val points = arrayOf(
            rotate(detection.x1, detection.y1, rotationDegrees, cameraWidth, cameraHeight),
            rotate(detection.x2, detection.y1, rotationDegrees, cameraWidth, cameraHeight),
            rotate(detection.x1, detection.y2, rotationDegrees, cameraWidth, cameraHeight),
            rotate(detection.x2, detection.y2, rotationDegrees, cameraWidth, cameraHeight),
        )
        val displayWidth = if (rotationDegrees == 90 || rotationDegrees == 270) {
            cameraHeight.toFloat()
        } else {
            cameraWidth.toFloat()
        }.coerceAtLeast(1f)
        val displayHeight = if (rotationDegrees == 90 || rotationDegrees == 270) {
            cameraWidth.toFloat()
        } else {
            cameraHeight.toFloat()
        }.coerceAtLeast(1f)

        val minX = points.minOf { it.first }
        val maxX = points.maxOf { it.first }
        val minY = points.minOf { it.second }
        val maxY = points.maxOf { it.second }
        val widthRatio = abs(maxX - minX) / displayWidth
        val heightRatio = abs(maxY - minY) / displayHeight

        return DisplayGeometry(
            centerX = ((minX + maxX) / 2f / displayWidth).coerceIn(0f, 1f),
            centerY = ((minY + maxY) / 2f / displayHeight).coerceIn(0f, 1f),
            areaRatio = widthRatio * heightRatio,
            maxDimension = maxOf(widthRatio, heightRatio),
        )
    }

    private fun rotate(
        x: Float,
        y: Float,
        degrees: Int,
        cameraWidth: Int,
        cameraHeight: Int,
    ): Pair<Float, Float> = when (degrees) {
        90 -> y to cameraWidth - x
        180 -> cameraWidth - x to cameraHeight - y
        270 -> cameraHeight - y to x
        else -> x to y
    }

    private data class DisplayGeometry(
        val centerX: Float,
        val centerY: Float,
        val areaRatio: Float,
        val maxDimension: Float,
    )
}
