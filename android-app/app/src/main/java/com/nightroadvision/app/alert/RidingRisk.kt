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
    val distanceSource: String = "",        // 距离来源：supercombo / pinhole / bbox
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

        // Typical visible widths (meters) for width-based distance estimation
        private val CLASS_WIDTHS = mapOf(
            "person" to 0.5f,
            "rider" to 0.6f,
            "bicycle" to 0.6f,
            "car" to 1.8f,
            "motorcycle" to 0.8f,
            "bus" to 2.5f,
            "truck" to 2.4f,
        )

        // Typical bbox aspect ratios (width/height) for confidence scoring
        private val CLASS_ASPECTS = mapOf(
            "person" to 0.4f,
            "rider" to 0.5f,
            "bicycle" to 0.6f,
            "car" to 1.2f,
            "motorcycle" to 0.7f,
            "bus" to 1.5f,
            "truck" to 1.4f,
        )

        // Default focal length factor (tunable via calibration)
        // Calibrated for typical phone camera (~66° vertical FOV, 720px height)
        // distance = (real_height * focal_factor) / pixel_height_in_camera_space
        private const val DEFAULT_FOCAL_FACTOR = 588f
    }

    private val previousAreaByTrack = mutableMapOf<Int, Float>()
    private val previousDistanceByTrack = mutableMapOf<Int, Float>()
    private val previousConfidenceByTrack = mutableMapOf<Int, Float>()
    private val previousDistanceSourceByTrack = mutableMapOf<Int, String>()
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
     * Estimate distance using the pinhole camera model.
     *
     * Uses both height and width of the bounding box and averages them for robustness.
     * At long range, small pixel errors cause large distance errors, so we also
     * compute a confidence value based on how large the bbox is.
     *
     * @return Pair(distance meters, confidence 0..1) or null if class unknown
     */
    private fun estimateDistance(
        detection: InferenceEngine.Detection,
        cameraHeight: Int,
    ): Pair<Float, Float>? {
        val className = detection.className.lowercase()
        val classHeight = CLASS_HEIGHTS[className] ?: return null
        val pixelHeight = abs(detection.y2 - detection.y1).coerceAtLeast(1f)
        val pixelWidth = abs(detection.x2 - detection.x1).coerceAtLeast(1f)

        // Typical aspect ratios (width/height) for each class at ~30m
        val classAspect = CLASS_ASPECTS[className] ?: 0.5f

        val resolutionAdjustedFocal = focalFactor * cameraHeight.coerceAtLeast(1) / 720f

        // Height-based estimate
        val distFromHeight = (classHeight * resolutionAdjustedFocal) / pixelHeight

        // With square camera pixels, horizontal and vertical focal lengths are
        // approximately equal in pixel units. Scaling by frame width here would
        // over-estimate distance on a 16:9 frame.
        val classWidth = CLASS_WIDTHS[className] ?: 1.8f
        val widthFocal = resolutionAdjustedFocal
        val distFromWidth = (classWidth * widthFocal) / pixelWidth

        // Weighted average: height is more reliable for upright objects (person, car),
        // width helps when object is partially occluded vertically.
        val dist = when {
            pixelHeight > 40f && pixelWidth > 20f -> distFromHeight * 0.6f + distFromWidth * 0.4f
            pixelHeight > 20f -> distFromHeight  // small bbox, trust height more
            else -> distFromHeight * 0.7f + distFromWidth * 0.3f
        }

        // Confidence: higher for closer (larger) objects, lower for distant (tiny) ones
        // At 5m: pixelHeight ~170px → confidence ~0.95
        // At 15m: pixelHeight ~57px → confidence ~0.8
        // At 30m: pixelHeight ~29px → confidence ~0.5
        // At 50m: pixelHeight ~17px → confidence ~0.3
        val sizeConfidence = (pixelHeight / 180f).coerceIn(0.15f, 1f)

        // Aspect ratio plausibility check: if bbox is very elongated or squished,
        // the detection may be partial/occluded → lower confidence
        val actualAspect = pixelWidth / pixelHeight
        val aspectDeviation = abs(actualAspect - classAspect) / classAspect
        val aspectConfidence = (1f - aspectDeviation * 0.5f).coerceIn(0.4f, 1f)

        val confidence = (sizeConfidence * aspectConfidence).coerceIn(0.1f, 1f)

        return Pair(dist.coerceIn(1f, 200f), confidence)
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
        previousConfidenceByTrack.keys.retainAll(liveTrackIds)
        previousDistanceSourceByTrack.keys.retainAll(liveTrackIds)

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

            val severity: RiskSeverity
            val reason: RiskReason
            val supercomboDistance = detection.distanceMeters
                ?.takeIf { it.isFinite() && it > 0.5f }
            val pinholeResult = when {
                supercomboDistance != null -> null
                detection.estimatedDistanceMeters != null -> {
                    detection.estimatedDistanceMeters
                        .takeIf { it.isFinite() && it > 0.5f }
                        ?.let { Pair(it, 0.7f) }
                }
                else -> estimateDistance(detection, cameraHeight)
            }
            val rawDistance = supercomboDistance ?: pinholeResult?.first
            val estimateConfidence = if (supercomboDistance != null) 1f else pinholeResult?.second ?: 0f
            val distanceSource = when {
                supercomboDistance != null -> "supercombo"
                rawDistance != null -> "pinhole"
                else -> "bbox"
            }

            // Smooth exactly one source per frame. When the source switches, start
            // from the new measurement instead of blending metric model output with
            // a class-size estimate from the previous frame.
            val previousDistance = trackId?.let(previousDistanceByTrack::get)
            val previousSource = trackId?.let(previousDistanceSourceByTrack::get)
            val alpha = if (distanceSource == "supercombo") {
                0.68f
            } else {
                (0.3f + estimateConfidence * 0.5f).coerceIn(0.3f, 0.8f)
            }
            val smoothedDistance = when {
                rawDistance == null -> null
                previousDistance != null && previousSource == distanceSource ->
                    previousDistance * (1f - alpha) + rawDistance * alpha
                else -> rawDistance
            }
            if (trackId != null && smoothedDistance != null) {
                val previousConfidence = previousConfidenceByTrack[trackId] ?: 0f
                previousDistanceByTrack[trackId] = smoothedDistance
                previousConfidenceByTrack[trackId] =
                    maxOf(estimateConfidence, previousConfidence * 0.9f)
                previousDistanceSourceByTrack[trackId] = distanceSource
            }

            if (smoothedDistance != null) {
                when {
                    smoothedDistance < dangerThreshold -> {
                        severity = RiskSeverity.DANGER
                        reason = RiskReason.IMMINENT
                    }
                    smoothedDistance < urgentThreshold -> {
                        severity = RiskSeverity.URGENT
                        reason = RiskReason.VERY_NEAR
                    }
                    smoothedDistance < cautionThreshold -> {
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
                    estimatedDistanceM = smoothedDistance,
                    distanceSource = distanceSource,
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
        previousConfidenceByTrack.clear()
        previousDistanceSourceByTrack.clear()
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
