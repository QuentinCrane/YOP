package com.nightroadvision.app.alert

import android.content.Context
import android.os.Build
import android.os.SystemClock
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import com.nightroadvision.app.FileLogger

/** Provides rate-limited haptic alerts for confirmed riding risks. */
class RidingVibrationController(context: Context) {
    companion object {
        private const val TAG = "RidingVibration"
        private const val MIN_GLOBAL_INTERVAL_MS = 1_500L
        private const val CAUTION_COOLDOWN_MS = 5_000L
        private const val CRITICAL_COOLDOWN_MS = 2_500L
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
        val cooldown = if (risk.severity == RiskSeverity.CRITICAL) {
            CRITICAL_COOLDOWN_MS
        } else {
            CAUTION_COOLDOWN_MS
        }
        val newTarget = risk.trackId != lastTrackId
        val escalated = risk.severity.ordinal > lastSeverity.ordinal
        if (now - lastAlertAtMs < MIN_GLOBAL_INTERVAL_MS) return
        if (!newTarget && !escalated && now - lastAlertAtMs < cooldown) return

        val pattern = if (risk.severity == RiskSeverity.CRITICAL) {
            longArrayOf(0L, 130L, 90L, 220L)
        } else {
            longArrayOf(0L, 110L)
        }
        deviceVibrator.vibrate(VibrationEffect.createWaveform(pattern, -1))

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
