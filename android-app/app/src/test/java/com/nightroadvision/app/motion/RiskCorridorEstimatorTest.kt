package com.nightroadvision.app.motion

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class RiskCorridorEstimatorTest {
    private val estimator = RiskCorridorEstimator()

    @Test
    fun stationaryCorridorIsStraightAndShort() {
        val corridor = estimator.estimate(
            speedKmh = 0f,
            yawRateRadPerSecond = 0.8f,
            motionSensorAvailable = true,
        )

        assertTrue(corridor.points.isNotEmpty())
        assertTrue(corridor.points.all { kotlin.math.abs(it.lateralMeters) < 0.001f })
        assertTrue(corridor.points.last().forwardMeters <= 8.01f)
    }

    @Test
    fun speedExtendsCorridorAndYawControlsTurnDirection() {
        val slow = estimator.estimate(10f, 0f, motionSensorAvailable = true)
        val left = estimator.estimate(45f, 0.25f, motionSensorAvailable = true)
        val right = estimator.estimate(45f, -0.25f, motionSensorAvailable = true)

        assertTrue(left.points.last().forwardMeters > slow.points.last().forwardMeters)
        assertTrue(left.points.last().lateralMeters > 0f)
        assertTrue(right.points.last().lateralMeters < 0f)
    }

    @Test
    fun projectedCorridorCenterMatchesWarningHitTest() {
        val corridor = estimator.estimate(30f, 0.18f, motionSensorAvailable = true)
        val section = corridor.points[corridor.points.size / 2]
        val screenPoint = requireNotNull(
            CorridorProjection.project(section.forwardMeters, section.lateralMeters)
        )

        assertTrue(corridor.containsScreenPoint(screenPoint.x, screenPoint.y))
        assertFalse(corridor.containsScreenPoint(0.02f, screenPoint.y))
    }
}
