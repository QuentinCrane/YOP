package com.nightroadvision.app.motion

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

/** A sampled cross-section of the short-horizon area the rider is likely to occupy. */
data class CorridorPoint(
    val forwardMeters: Float,
    val lateralMeters: Float,
    val halfWidthMeters: Float,
)

data class NormalizedScreenPoint(
    val x: Float,
    val y: Float,
)

data class RoadPoint(
    val forwardMeters: Float,
    val lateralMeters: Float,
)

/**
 * Lightweight alternative to a learned path-planning model.
 *
 * The same geometry is consumed by the overlay and the risk evaluator, keeping the
 * visible ribbon and the active warning area aligned.
 */
data class DrivingCorridor(
    val points: List<CorridorPoint> = emptyList(),
    val motionSensorAvailable: Boolean = false,
) {
    fun containsScreenPoint(
        normalizedX: Float,
        normalizedY: Float,
        extraMarginMeters: Float = 0f,
    ): Boolean {
        val roadPoint = CorridorProjection.unproject(normalizedX, normalizedY) ?: return false
        return containsRoadPoint(
            forwardMeters = roadPoint.forwardMeters,
            lateralMeters = roadPoint.lateralMeters,
            extraMarginMeters = extraMarginMeters,
        )
    }

    fun containsRoadPoint(
        forwardMeters: Float,
        lateralMeters: Float,
        extraMarginMeters: Float = 0f,
    ): Boolean {
        if (points.size < 2 || forwardMeters !in points.first().forwardMeters..points.last().forwardMeters) {
            return false
        }

        val upperIndex = points.indexOfFirst { it.forwardMeters >= forwardMeters }
            .takeIf { it > 0 }
            ?: return false
        val lower = points[upperIndex - 1]
        val upper = points[upperIndex]
        val span = (upper.forwardMeters - lower.forwardMeters).coerceAtLeast(0.001f)
        val fraction = ((forwardMeters - lower.forwardMeters) / span).coerceIn(0f, 1f)
        val center = lerp(lower.lateralMeters, upper.lateralMeters, fraction)
        val halfWidth = lerp(lower.halfWidthMeters, upper.halfWidthMeters, fraction)

        return abs(lateralMeters - center) <= halfWidth + extraMarginMeters.coerceAtLeast(0f)
    }

    companion object {
        val EMPTY = DrivingCorridor()

        private fun lerp(start: Float, end: Float, fraction: Float): Float =
            start + (end - start) * fraction
    }
}

/** Shared approximate camera projection used for both drawing and collision-zone tests. */
object CorridorProjection {
    const val HORIZON_Y_NORMALIZED = 0.42f
    const val FOCAL_X_NORMALIZED = 0.46f
    const val FOCAL_Y_NORMALIZED = 0.82f
    const val CAMERA_HEIGHT_METERS = 1.20f
    const val START_Y_NORMALIZED = 0.92f

    fun project(forwardMeters: Float, lateralMeters: Float): NormalizedScreenPoint? {
        if (!forwardMeters.isFinite() || !lateralMeters.isFinite() || forwardMeters < 0.75f) return null
        return NormalizedScreenPoint(
            x = 0.5f - lateralMeters * FOCAL_X_NORMALIZED / forwardMeters,
            y = (HORIZON_Y_NORMALIZED + CAMERA_HEIGHT_METERS * FOCAL_Y_NORMALIZED / forwardMeters)
                .coerceIn(HORIZON_Y_NORMALIZED, START_Y_NORMALIZED),
        )
    }

    fun unproject(normalizedX: Float, normalizedY: Float): RoadPoint? {
        if (!normalizedX.isFinite() || !normalizedY.isFinite()) return null
        val vertical = normalizedY - HORIZON_Y_NORMALIZED
        if (vertical <= 0.015f) return null
        val forward = CAMERA_HEIGHT_METERS * FOCAL_Y_NORMALIZED / vertical
        if (!forward.isFinite() || forward < 0.75f) return null
        val lateral = (0.5f - normalizedX) * forward / FOCAL_X_NORMALIZED
        return RoadPoint(forwardMeters = forward, lateralMeters = lateral)
    }
}

/** Configuration for the driving corridor estimator. */
data class CorridorConfig(
    val baseHalfWidthMeters: Float = 0.85f,
    val maxHalfWidthMeters: Float = 1.55f,
    val horizonSeconds: Float = 2.0f,
    val minDistanceMeters: Float = 8f,
    val maxDistanceMeters: Float = 32f,
    val widthGrowthPerMeter: Float = 0.018f,
    val widthGrowthPerYawRate: Float = 0.18f,
    /** Speed (km/h) at which the corridor starts expanding beyond minimum. */
    val speedRangeLowKmh: Float = 10f,
    /** Speed (km/h) at which the corridor reaches its maximum expansion. */
    val speedRangeHighKmh: Float = 60f,
) {
    companion object {
        val DEFAULT = CorridorConfig()
    }

    /**
     * Returns a 0..1 factor representing how much the corridor should expand
     * based on current speed. 0 = minimum size, 1 = maximum size.
     */
    fun speedScaleFactor(speedKmh: Float): Float {
        if (speedRangeHighKmh <= speedRangeLowKmh) return 1f
        return ((speedKmh - speedRangeLowKmh) / (speedRangeHighKmh - speedRangeLowKmh))
            .coerceIn(0f, 1f)
    }
}

/** Constant-curvature short-horizon estimator driven by speed and measured yaw rate. */
class RiskCorridorEstimator {
    companion object {
        private const val POINT_COUNT = 24
        private const val MIN_CURVE_SPEED_MPS = 1.2f
        private const val CURVATURE_SPEED_FLOOR_MPS = 2.5f
        private const val MAX_TOTAL_HEADING_RADIANS = 0.85f
        private const val MAX_CURVATURE_PER_METER = 0.12f
    }

    fun estimate(
        speedKmh: Float,
        yawRateRadPerSecond: Float,
        motionSensorAvailable: Boolean,
        config: CorridorConfig = CorridorConfig.DEFAULT,
    ): DrivingCorridor {
        val speedMps = (speedKmh.coerceAtLeast(0f) / 3.6f)
        val scale = config.speedScaleFactor(speedKmh)
        // Distance scales from min to max based on speed
        val distance = config.minDistanceMeters +
            (config.maxDistanceMeters - config.minDistanceMeters) * scale
        val requestedCurvature = if (motionSensorAvailable && speedMps >= MIN_CURVE_SPEED_MPS) {
            yawRateRadPerSecond / speedMps.coerceAtLeast(CURVATURE_SPEED_FLOOR_MPS)
        } else {
            0f
        }
        val headingLimitedCurvature = MAX_TOTAL_HEADING_RADIANS / distance
        val curvatureLimit = minOf(MAX_CURVATURE_PER_METER, headingLimitedCurvature)
        val curvature = requestedCurvature.coerceIn(-curvatureLimit, curvatureLimit)

        val points = ArrayList<CorridorPoint>(POINT_COUNT)
        for (index in 1..POINT_COUNT) {
            val travelled = distance * index / POINT_COUNT
            val heading = curvature * travelled
            val forward = if (abs(curvature) < 0.0001f) {
                travelled
            } else {
                sin(heading) / curvature
            }
            val lateral = if (abs(curvature) < 0.0001f) {
                0f
            } else {
                (1f - cos(heading)) / curvature
            }
            if (forward < 0.75f) continue

            val uncertainty = travelled * config.widthGrowthPerMeter +
                abs(yawRateRadPerSecond) * config.widthGrowthPerYawRate
            // Base width scales with speed: at low speed use 60% of base, at high speed use 100%
            val effectiveBaseWidth = config.baseHalfWidthMeters * (0.6f + 0.4f * scale)
            points += CorridorPoint(
                forwardMeters = forward,
                lateralMeters = lateral,
                halfWidthMeters = (effectiveBaseWidth + uncertainty)
                    .coerceAtMost(config.maxHalfWidthMeters),
            )
        }

        return DrivingCorridor(
            points = points,
            motionSensorAvailable = motionSensorAvailable,
        )
    }
}
