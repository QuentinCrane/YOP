package com.nightroadvision.app.alert

import com.nightroadvision.app.inference.InferenceEngine
import kotlin.math.abs

enum class RiskSeverity {
    NONE,
    CAUTION,   // 注意：目标进入注意区域
    URGENT,    // 紧急：目标快速接近或较近
    DANGER,    // 危险：目标非常近，需要立即反应
}

enum class RiskReason {
    NONE,
    IN_ATTENTION_ZONE,
    APPROACHING_QUICKLY,
    VERY_NEAR,
    IMMINENT,
}

data class RidingRisk(
    val severity: RiskSeverity = RiskSeverity.NONE,
    val reason: RiskReason = RiskReason.NONE,
    val trackId: Int? = null,
    val className: String = "",
    val estimatedDistanceM: Float? = null,  // 估算距离（米）
)

/**
 * Evaluates tracked people and vehicles in display space.
 *
 * Produces three risk levels:
 * - CAUTION: object entered the attention zone
 * - URGENT: object is approaching quickly or moderately close
 * - DANGER: object is very close, immediate reaction needed
 *
 * Includes basic distance estimation based on bounding box size
 * and known object class heights.
 */
class RidingRiskEvaluator {
    companion object {
        private const val CLEAR_CONFIRMATION_FRAMES = 3

        // Typical visible heights (meters). Names work for both COCO and custom
        // night-road class layouts, whose numeric vehicle IDs are different.
        private val CLASS_HEIGHTS = mapOf(
            "person" to 1.7f,
            "rider" to 1.7f,
            "bicycle" to 1.5f,
            "car" to 1.5f,
            "motorcycle" to 1.4f,
            "bus" to 1.5f,
            "truck" to 1.8f,
        )

        // Default focal length factor (tunable via calibration)
        // Calibrated for typical phone camera (~66° vertical FOV, 720px height)
        // distance = (real_height * focal_factor) / pixel_height_in_camera_space
        private const val DEFAULT_FOCAL_FACTOR = 588f
    }

    private val previousAreaByTrack = mutableMapOf<Int, Float>()
    private val previousDistanceByTrack = mutableMapOf<Int, Float>()
    private var lastRisk = RidingRisk()
    private var consecutiveClearFrames = 0

    // Calibration: focal length factor (adjustable)
    private var focalFactor = DEFAULT_FOCAL_FACTOR

    // Configurable distance thresholds (meters)
    private var dangerThreshold = 8f
    private var urgentThreshold = 18f
    private var cautionThreshold = 35f

    @Synchronized
    fun setFocalFactor(factor: Float) {
        focalFactor = factor.coerceIn(50f, 2000f)
    }

    @Synchronized
    fun getFocalFactor(): Float = focalFactor

    @Synchronized
    fun setDistanceThresholds(dangerM: Float, urgentM: Float, cautionM: Float) {
        dangerThreshold = dangerM.coerceIn(1f, 50f)
        urgentThreshold = urgentM.coerceIn(3f, 80f).coerceAtLeast(dangerThreshold)
        cautionThreshold = cautionM.coerceIn(5f, 150f).coerceAtLeast(urgentThreshold)
    }

    /**
     * Estimate distance to a detected object based on its bounding box size
     * and known real-world class height.
     *
     * Uses the pinhole camera model: distance = (real_height * focal_factor) / pixel_height
     */
    private fun estimateDistance(
        detection: InferenceEngine.Detection,
        cameraHeight: Int,
    ): Float? {
        val classHeight = CLASS_HEIGHTS[detection.className.lowercase()] ?: return null
        val pixelHeight = abs(detection.y2 - detection.y1)
        if (pixelHeight < 1f) return null

        // Pinhole camera model: distance = (real_height * focal_factor) / pixel_height
        // focalFactor is calibrated at 720px; scale it with the active analysis height.
        val resolutionAdjustedFocal = focalFactor * cameraHeight.coerceAtLeast(1) / 720f
        return (classHeight * resolutionAdjustedFocal) / pixelHeight
    }

    @Synchronized
    fun evaluate(
        detections: List<InferenceEngine.Detection>,
        rotationDegrees: Int,
        cameraWidth: Int,
        cameraHeight: Int,
    ): RidingRisk {
        val liveTrackIds = detections.mapNotNull { it.trackId }.toSet()
        previousAreaByTrack.keys.retainAll(liveTrackIds)
        previousDistanceByTrack.keys.retainAll(liveTrackIds)

        var selected = RidingRisk()
        var selectedScore = 0f

        for (detection in detections) {
            val trackId = detection.trackId
            val geometry = displayGeometry(detection, rotationDegrees, cameraWidth, cameraHeight)
            val previousArea = trackId?.let(previousAreaByTrack::get)
            val growthRatio = if (previousArea != null && previousArea > 0.002f) {
                geometry.areaRatio / previousArea
            } else {
                1f
            }
            if (trackId != null) previousAreaByTrack[trackId] = geometry.areaRatio

            val inAttentionZone = geometry.centerX in 0.22f..0.78f &&
                geometry.centerY in 0.34f..1f
            if (!inAttentionZone) continue

            // Estimate distance
            val estimatedDist = estimateDistance(detection, cameraHeight)

            // Smooth distance with previous estimate
            val prevDist = trackId?.let(previousDistanceByTrack::get)
            val smoothedDist = if (prevDist != null && estimatedDist != null) {
                prevDist * 0.3f + estimatedDist * 0.7f
            } else {
                estimatedDist
            }
            if (trackId != null && smoothedDist != null) {
                previousDistanceByTrack[trackId] = smoothedDist
            }

            val severity: RiskSeverity
            val reason: RiskReason

            // Use real distance from supercombo if available, otherwise use estimated
            val dist = detection.distanceMeters ?: smoothedDist

            if (dist != null) {
                when {
                    dist < dangerThreshold -> {
                        severity = RiskSeverity.DANGER
                        reason = RiskReason.IMMINENT
                    }
                    dist < urgentThreshold -> {
                        severity = RiskSeverity.URGENT
                        reason = RiskReason.VERY_NEAR
                    }
                    dist < cautionThreshold -> {
                        severity = RiskSeverity.CAUTION
                        reason = RiskReason.IN_ATTENTION_ZONE
                    }
                    else -> continue
                }
            } else {
                // Fallback to bbox-based heuristics (no distance available)
                val isImminent = geometry.maxDimension >= 0.50f || geometry.areaRatio >= 0.18f
                val isVeryNear = geometry.maxDimension >= 0.38f || geometry.areaRatio >= 0.12f
                val isApproachingQuickly = growthRatio >= 1.28f && geometry.areaRatio >= 0.02f
                val isCaution = geometry.maxDimension >= 0.18f || geometry.areaRatio >= 0.03f

                when {
                    isImminent -> {
                        severity = RiskSeverity.DANGER
                        reason = RiskReason.IMMINENT
                    }
                    isVeryNear -> {
                        severity = RiskSeverity.URGENT
                        reason = RiskReason.VERY_NEAR
                    }
                    isApproachingQuickly -> {
                        severity = RiskSeverity.URGENT
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
                    estimatedDistanceM = dist,
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

    @Synchronized
    fun reset() {
        previousAreaByTrack.clear()
        previousDistanceByTrack.clear()
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
