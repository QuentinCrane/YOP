package com.nightroadvision.app.alert

import android.content.Context
import android.os.Build
import android.os.SystemClock
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import com.nightroadvision.app.FileLogger

/**
 * Three-level haptic feedback controller for riding risk alerts.
 *
 * - CAUTION (注意): single soft pulse
 * - URGENT (紧急): double pulse with medium intensity
 * - DANGER (危险): triple strong pulse pattern
 */
class RidingVibrationController(context: Context) {
    companion object {
        private const val TAG = "RidingVibration"
        private const val MIN_GLOBAL_INTERVAL_MS = 1_200L
        private const val CAUTION_COOLDOWN_MS = 5_000L
        private const val URGENT_COOLDOWN_MS = 3_000L
        private const val DANGER_COOLDOWN_MS = 1_800L
    }

    private val vibrator: Vibrator? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        context.getSystemService(VibratorManager::class.java)?.defaultVibrator
    } else {
        @Suppress("DEPRECATION")
        context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
    }

    private var lastAlertAtMs = 0L
    private var lastTrackId: Int? = null
    private var lastSeverity = RiskSeverity.NONE

    fun update(risk: RidingRisk, enabled: Boolean) {
        if (!enabled || risk.severity == RiskSeverity.NONE) {
            if (risk.severity == RiskSeverity.NONE) lastSeverity = RiskSeverity.NONE
            return
        }
        val deviceVibrator = vibrator ?: return
        if (!deviceVibrator.hasVibrator()) return

        val now = SystemClock.elapsedRealtime()
        val cooldown = when (risk.severity) {
            RiskSeverity.DANGER -> DANGER_COOLDOWN_MS
            RiskSeverity.URGENT -> URGENT_COOLDOWN_MS
            else -> CAUTION_COOLDOWN_MS
        }
        val newTarget = risk.trackId != lastTrackId
        val escalated = risk.severity.ordinal > lastSeverity.ordinal
        if (now - lastAlertAtMs < MIN_GLOBAL_INTERVAL_MS) return
        if (!newTarget && !escalated && now - lastAlertAtMs < cooldown) return

        val pattern = when (risk.severity) {
            RiskSeverity.DANGER -> longArrayOf(0L, 150L, 60L, 200L, 60L, 150L)  // 三段脉冲
            RiskSeverity.URGENT -> longArrayOf(0L, 130L, 70L, 200L)             // 双脉冲
            else -> longArrayOf(0L, 100L)                                         // 单脉冲
        }

        val amplitude = when (risk.severity) {
            RiskSeverity.DANGER -> 200  // 强
            RiskSeverity.URGENT -> 140  // 中
            else -> 80                   // 轻
        }

        // Build amplitude array matching pattern length (even indices = gap/silence, odd = vibrate)
        val amplitudes = IntArray(pattern.size) { i ->
            if (i % 2 == 0) 0 else amplitude
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            deviceVibrator.vibrate(VibrationEffect.createWaveform(pattern, amplitudes, -1))
        } else {
            deviceVibrator.vibrate(VibrationEffect.createWaveform(pattern, -1))
        }

        lastAlertAtMs = now
        lastTrackId = risk.trackId
        lastSeverity = risk.severity
        FileLogger.i(TAG, "Haptic ${risk.severity}: track=${risk.trackId}, reason=${risk.reason}")
    }

    fun reset() {
        lastAlertAtMs = 0L
        lastTrackId = null
        lastSeverity = RiskSeverity.NONE
        vibrator?.cancel()
    }
}
