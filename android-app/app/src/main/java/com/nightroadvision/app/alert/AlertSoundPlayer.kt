package com.nightroadvision.app.alert

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Log
import kotlin.math.PI
import kotlin.math.sin

/**
 * Tesla-style audio alert player for riding risk warnings.
 *
 * Synthesizes Tesla-inspired chime sounds using AudioTrack:
 * - CAUTION: ascending two-tone sweep (Tesla attention chime style)
 * - CRITICAL: rapid repeating urgent tone (Tesla collision warning style)
 */
class AlertSoundPlayer {

    companion object {
        private const val TAG = "AlertSoundPlayer"
        private const val SAMPLE_RATE = 22050

        // CAUTION: Tesla-style ascending sweep chime
        private const val CAUTION_FREQ_START = 880f   // A5
        private const val CAUTION_FREQ_END = 1320f     // E6
        private const val CAUTION_DURATION_MS = 220L
        private const val CAUTION_VOLUME = 0.55f

        // CRITICAL: Tesla-style urgent repeating tone
        private const val CRITICAL_FREQ = 1047f        // C6
        private const val CRITICAL_BEEP_MS = 80L
        private const val CRITICAL_GAP_MS = 60L
        private const val CRITICAL_BEEP_COUNT = 4
        private const val CRITICAL_VOLUME = 0.7f

        private const val MIN_ALERT_INTERVAL_MS = 2000L
    }

    private var lastAlertTime = 0L
    private var lastRiskSeverity = RiskSeverity.NONE
    private var isPlaying = false
    private var enabled = true

    // Pre-generated PCM samples
    private val cautionPcm: ShortArray by lazy { generateCautionChime() }
    private val criticalPcm: ShortArray by lazy { generateCriticalBeep() }

    fun setEnabled(value: Boolean) {
        enabled = value
        if (!enabled) stopAlerts()
    }

    fun isEnabled(): Boolean = enabled

    fun onRiskChanged(risk: RidingRisk) {
        if (!enabled) return

        val now = System.currentTimeMillis()
        val elapsed = now - lastAlertTime

        when (risk.severity) {
            RiskSeverity.NONE -> {
                lastRiskSeverity = RiskSeverity.NONE
                return
            }
            RiskSeverity.CAUTION -> {
                if (lastRiskSeverity != RiskSeverity.CAUTION || elapsed > MIN_ALERT_INTERVAL_MS * 2) {
                    playCautionAlert()
                    lastAlertTime = now
                }
            }
            RiskSeverity.CRITICAL -> {
                if (elapsed > MIN_ALERT_INTERVAL_MS) {
                    playCriticalAlert()
                    lastAlertTime = now
                }
            }
        }
        lastRiskSeverity = risk.severity
    }

    /**
     * Tesla attention chime: ascending sine sweep from A5 to E6.
     * A clean, pleasant two-tone "bloop" that's instantly recognizable.
     */
    private fun generateCautionChime(): ShortArray {
        val durationSec = CAUTION_DURATION_MS / 1000.0
        val numSamples = (SAMPLE_RATE * durationSec).toInt()
        val pcm = ShortArray(numSamples)

        for (i in 0 until numSamples) {
            val t = i.toDouble() / SAMPLE_RATE
            val progress = t / durationSec

            // Frequency sweeps linearly from start to end
            val freq = CAUTION_FREQ_START.toDouble() +
                (CAUTION_FREQ_END - CAUTION_FREQ_START).toDouble() * progress
            val phase = 2.0 * PI * freq * t

            // Smooth envelope: quick attack, sustained, smooth release
            val envelope = when {
                progress < 0.05 -> progress / 0.05                          // attack 5%
                progress < 0.7 -> 1.0                                        // sustain
                else -> ((1.0 - progress) / 0.3).coerceIn(0.0, 1.0)        // release 30%
            }

            val sample = sin(phase).toFloat() * CAUTION_VOLUME * envelope.toFloat()
            pcm[i] = (sample * Short.MAX_VALUE).toInt().coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort()
        }
        return pcm
    }

    /**
     * Tesla collision warning: urgent repeating tone at C6.
     * Rapid on/off pulses that convey immediate danger.
     */
    private fun generateCriticalBeep(): ShortArray {
        val oneBeepSamples = (SAMPLE_RATE * CRITICAL_BEEP_MS / 1000.0).toInt()
        val pcm = ShortArray(oneBeepSamples)

        for (i in 0 until oneBeepSamples) {
            val t = i.toDouble() / SAMPLE_RATE
            val progress = i.toDouble() / oneBeepSamples

            val phase = 2.0 * PI * CRITICAL_FREQ.toDouble() * t

            // Sharp envelope: instant attack, quick decay
            val envelope = when {
                progress < 0.1 -> progress / 0.1                // attack 10%
                progress < 0.7 -> 1.0                            // sustain
                else -> ((1.0 - progress) / 0.3)                // release 30%
            }

            val sample = sin(phase).toFloat() * CRITICAL_VOLUME * envelope.toFloat()
            pcm[i] = (sample * Short.MAX_VALUE).toInt().coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort()
        }
        return pcm
    }

    private fun playCautionAlert() {
        Thread {
            try {
                playPcm(cautionPcm)
            } catch (e: Exception) {
                Log.e(TAG, "Caution alert failed", e)
            }
        }.start()
    }

    private fun playCriticalAlert() {
        if (isPlaying) return
        isPlaying = true

        Thread {
            try {
                for (i in 0 until CRITICAL_BEEP_COUNT) {
                    if (!isPlaying) break
                    playPcm(criticalPcm)
                    if (i < CRITICAL_BEEP_COUNT - 1) {
                        Thread.sleep(CRITICAL_GAP_MS)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Critical alert failed", e)
            } finally {
                isPlaying = false
            }
        }.start()
    }

    private fun playPcm(pcm: ShortArray) {
        val bufferSize = AudioTrack.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )

        val track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ASSISTANCE_SONIFICATION)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(SAMPLE_RATE)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .build()
            )
            .setBufferSizeInBytes(maxOf(bufferSize, pcm.size * 2))
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()

        track.write(pcm, 0, pcm.size)
        track.play()

        // Wait for playback to complete
        val durationMs = (pcm.size.toLong() * 1000) / SAMPLE_RATE
        Thread.sleep(durationMs + 20)

        track.stop()
        track.release()
    }

    private fun stopAlerts() {
        isPlaying = false
    }

    fun release() {
        stopAlerts()
    }
}
