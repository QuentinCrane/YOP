package com.nightroadvision.app.alert

import com.nightroadvision.app.inference.InferenceEngine
import com.nightroadvision.app.motion.RiskCorridorEstimator
import org.junit.Assert.assertEquals
import org.junit.Test

class RidingRiskEvaluatorTest {
    private val sideObject = InferenceEngine.Detection(
        x1 = 0f,
        y1 = 350f,
        x2 = 100f,
        y2 = 700f,
        confidence = 0.9f,
        classId = 2,
        className = "car",
        cameraWidth = 1000,
        cameraHeight = 1000,
        estimatedDistanceMeters = 6f,
    )
    private val straightCorridor = RiskCorridorEstimator().estimate(
        speedKmh = 30f,
        yawRateRadPerSecond = 0f,
        motionSensorAvailable = true,
    )

    @Test
    fun corridorEnabledIgnoresCloseObjectOutsideCorridor() {
        val risk = RidingRiskEvaluator().evaluate(
            detections = listOf(sideObject),
            rotationDegrees = 0,
            cameraWidth = 1000,
            cameraHeight = 1000,
            drivingCorridor = straightCorridor,
            corridorFilteringEnabled = true,
        )

        assertEquals(RiskSeverity.NONE, risk.severity)
    }

    @Test
    fun corridorDisabledWarnsForCloseObjectAnywhereInFrame() {
        val risk = RidingRiskEvaluator().evaluate(
            detections = listOf(sideObject),
            rotationDegrees = 0,
            cameraWidth = 1000,
            cameraHeight = 1000,
            drivingCorridor = straightCorridor,
            corridorFilteringEnabled = false,
        )

        assertEquals(RiskSeverity.DANGER, risk.severity)
    }
}
