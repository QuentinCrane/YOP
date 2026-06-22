package com.nightroadvision.app.motion

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import com.nightroadvision.app.FileLogger
import kotlin.math.sqrt

data class VehicleMotionSample(
    val yawRateRadPerSecond: Float = 0f,
    val sensorAvailable: Boolean = false,
)

/**
 * Estimates rotation around gravity without assuming portrait or landscape mounting.
 * Sampling at 20 Hz and publishing at 10 Hz is sufficient for a visual risk corridor
 * while remaining much cheaper than another camera model or optical-flow pass.
 */
class VehicleMotionEstimator(
    context: Context,
    private val onMotionUpdate: (VehicleMotionSample) -> Unit,
) : SensorEventListener {
    companion object {
        private const val TAG = "VehicleMotion"
        private const val SENSOR_PERIOD_US = 50_000
        private const val PUBLISH_INTERVAL_NS = 100_000_000L
        private const val FILTER_ALPHA = 0.18f
        private const val GRAVITY_FILTER_ALPHA = 0.10f
        private const val YAW_DEAD_ZONE_RAD_PER_SECOND = 0.015f
        private const val MAX_YAW_RATE_RAD_PER_SECOND = 1.2f
    }

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val gravitySensor = sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY)
    private val accelerometer = if (gravitySensor == null) {
        sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    } else {
        null
    }
    private val gravity = FloatArray(3)

    @Volatile
    private var running = false
    private var gravityReady = false
    private var filteredYawRate = 0f
    private var lastPublishTimestampNs = 0L

    val isAvailable: Boolean
        get() = gyroscope != null && (gravitySensor != null || accelerometer != null)

    @Synchronized
    fun start(): Boolean {
        if (running) return isAvailable
        if (!isAvailable) {
            onMotionUpdate(VehicleMotionSample())
            FileLogger.w(TAG, "Gyroscope or gravity source unavailable; using straight corridor")
            return false
        }

        val gyroRegistered = sensorManager.registerListener(
            this,
            gyroscope,
            SENSOR_PERIOD_US,
        )
        val gravityRegistered = sensorManager.registerListener(
            this,
            gravitySensor ?: accelerometer,
            SENSOR_PERIOD_US,
        )
        running = gyroRegistered && gravityRegistered
        if (!running) sensorManager.unregisterListener(this)
        onMotionUpdate(VehicleMotionSample(sensorAvailable = running))
        FileLogger.i(TAG, "Motion estimation ${if (running) "started at 20Hz" else "unavailable"}")
        return running
    }

    @Synchronized
    fun stop() {
        if (running) sensorManager.unregisterListener(this)
        running = false
        gravityReady = false
        filteredYawRate = 0f
        lastPublishTimestampNs = 0L
        onMotionUpdate(VehicleMotionSample())
        FileLogger.i(TAG, "Motion estimation stopped")
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (!running) return
        when (event.sensor.type) {
            Sensor.TYPE_GRAVITY -> {
                gravity[0] = event.values[0]
                gravity[1] = event.values[1]
                gravity[2] = event.values[2]
                gravityReady = true
            }

            Sensor.TYPE_ACCELEROMETER -> {
                if (!gravityReady) {
                    gravity[0] = event.values[0]
                    gravity[1] = event.values[1]
                    gravity[2] = event.values[2]
                    gravityReady = true
                } else {
                    for (axis in 0..2) {
                        gravity[axis] += GRAVITY_FILTER_ALPHA * (event.values[axis] - gravity[axis])
                    }
                }
            }

            Sensor.TYPE_GYROSCOPE -> updateYawRate(event)
        }
    }

    private fun updateYawRate(event: SensorEvent) {
        if (!gravityReady) return
        val magnitude = sqrt(
            gravity[0] * gravity[0] + gravity[1] * gravity[1] + gravity[2] * gravity[2]
        )
        if (magnitude < 1f) return

        // Project angular velocity onto the gravity/up axis. This automatically
        // follows portrait, landscape, and small mounting-angle changes.
        var yawRate = (
            event.values[0] * gravity[0] +
                event.values[1] * gravity[1] +
                event.values[2] * gravity[2]
            ) / magnitude
        if (kotlin.math.abs(yawRate) < YAW_DEAD_ZONE_RAD_PER_SECOND) yawRate = 0f
        yawRate = yawRate.coerceIn(-MAX_YAW_RATE_RAD_PER_SECOND, MAX_YAW_RATE_RAD_PER_SECOND)
        filteredYawRate += FILTER_ALPHA * (yawRate - filteredYawRate)

        if (event.timestamp - lastPublishTimestampNs >= PUBLISH_INTERVAL_NS) {
            lastPublishTimestampNs = event.timestamp
            onMotionUpdate(
                VehicleMotionSample(
                    yawRateRadPerSecond = filteredYawRate,
                    sensorAvailable = true,
                )
            )
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
}
