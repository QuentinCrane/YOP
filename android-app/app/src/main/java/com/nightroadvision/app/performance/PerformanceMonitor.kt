package com.nightroadvision.app.performance

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.os.PowerManager

/**
 * Monitors device performance and thermal state to help adjust inference parameters.
 *
 * Provides thermal throttling status, available memory, and recommendations
 * for adjusting detection mode based on current device conditions.
 */
class PerformanceMonitor(private val context: Context) {

    /**
     * Thermal throttling severity levels.
     */
    enum class ThermalLevel {
        NONE,
        LIGHT,
        MODERATE,
        SEVERE,
        CRITICAL,
        SHUTDOWN
    }

    /**
     * Snapshot of current device performance state.
     */
    data class PerformanceState(
        val thermalLevel: ThermalLevel = ThermalLevel.NONE,
        val thermalStatus: String = "Normal",
        val availableMemoryMb: Long = 0L,
        val totalMemoryMb: Long = 0L,
        val memoryPressure: Boolean = false
    )

    private val powerManager = context.getSystemService(Context.POWER_SERVICE) as? PowerManager
    private val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager

    /**
     * Get the current thermal throttling status.
     * Requires API 29+ for PowerManager.getThermalStatus(); falls back to NONE on older APIs.
     * Uses reflection since the method may not be available at compile time.
     */
    fun getThermalLevel(): ThermalLevel {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val status = try {
                powerManager?.javaClass?.getMethod("getThermalStatus")?.invoke(powerManager) as? Int
                    ?: 0
            } catch (_: Exception) {
                0
            }
            return when (status) {
                0 -> ThermalLevel.NONE   // THERMAL_STATUS_NONE
                1 -> ThermalLevel.LIGHT   // THERMAL_STATUS_LIGHT
                2 -> ThermalLevel.MODERATE // THERMAL_STATUS_MODERATE
                3 -> ThermalLevel.SEVERE   // THERMAL_STATUS_SEVERE
                4 -> ThermalLevel.CRITICAL // THERMAL_STATUS_CRITICAL
                5 -> ThermalLevel.SHUTDOWN // THERMAL_STATUS_SHUTDOWN
                else -> ThermalLevel.NONE
            }
        }
        return ThermalLevel.NONE
    }

    /**
     * Get a human-readable label for the current thermal state.
     */
    fun getThermalStatusLabel(): String {
        return getThermalLabel(getThermalLevel())
    }

    private fun getThermalLabel(level: ThermalLevel): String {
        return when (level) {
            ThermalLevel.NONE -> "Normal"
            ThermalLevel.LIGHT -> "Light"
            ThermalLevel.MODERATE -> "Moderate"
            ThermalLevel.SEVERE -> "Severe"
            ThermalLevel.CRITICAL -> "Critical"
            ThermalLevel.SHUTDOWN -> "Shutdown"
        }
    }

    /**
     * Get current memory info.
     */
    fun getMemoryInfo(): Pair<Long, Long> {
        val memInfo = ActivityManager.MemoryInfo()
        activityManager?.getMemoryInfo(memInfo)
        val availMb = memInfo.availMem / (1024 * 1024)
        val totalMb = memInfo.totalMem / (1024 * 1024)
        return Pair(availMb, totalMb)
    }

    /**
     * Get a full performance state snapshot.
     */
    fun getState(): PerformanceState {
        val (availMb, totalMb) = getMemoryInfo()
        val thermal = getThermalLevel()
        val memInfo = ActivityManager.MemoryInfo()
        activityManager?.getMemoryInfo(memInfo)

        return PerformanceState(
            thermalLevel = thermal,
            thermalStatus = getThermalLabel(thermal),
            availableMemoryMb = availMb,
            totalMemoryMb = totalMb,
            memoryPressure = memInfo.lowMemory
        )
    }

    /**
     * Recommend a detection mode based on current thermal and memory conditions.
     * Returns null if no change is recommended.
     */
    fun recommendDetectionMode(): String? {
        return when (getThermalLevel()) {
            ThermalLevel.NONE, ThermalLevel.LIGHT -> null // no change needed
            ThermalLevel.MODERATE -> "BALANCED"
            ThermalLevel.SEVERE, ThermalLevel.CRITICAL, ThermalLevel.SHUTDOWN -> "ECO"
        }
    }
}
