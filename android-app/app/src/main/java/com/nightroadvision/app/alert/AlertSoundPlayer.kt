package com.nightroadvision.app.alert

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Log
import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.pow
import kotlin.math.sin

/**
 * Three-level Tesla-style audio alert player.
 *
 * - CAUTION (注意): warm two-tone ascending bell chime
 * - URGENT (紧急): three-note descending chime, faster and brighter
 * - DANGER (危险): rapid alternating high-low alarm pulses
 *
 * Uses multi-layered harmonic synthesis with bell-like inharmonic partials
 * and proper ADSR envelopes for a clean, modern chime sound.
 */
class AlertSoundPlayer {

    companion object {
        private const val TAG = "AlertSoundPlayer"
        private const val SAMPLE_RATE = 44100

        // ── CAUTION: warm ascending bell chime ──
        private const val CAUTION_FREQ_1 = 587.33f   // D5
        private const val CAUTION_FREQ_2 = 783.99f   // G5
        private const val CAUTION_TONE_MS = 160L
        private const val CAUTION_GAP_MS = 30L
        private const val CAUTION_VOLUME = 0.50f

        // ── URGENT: brighter descending three-note ──
        private const val URGENT_FREQ_1 = 880.00f    // A5
        private const val URGENT_FREQ_2 = 740.00f    // F#5
        private const val URGENT_FREQ_3 = 587.33f    // D5
        private const val URGENT_TONE_MS = 110L
        private const val URGENT_GAP_MS = 25L
        private const val URGENT_VOLUME = 0.58f

        // ── DANGER: alternating alarm pulses ──
        private const val DANGER_FREQ_HIGH = 1174.66f  // D6
        private const val DANGER_FREQ_LOW = 783.99f    // G5
        private const val DANGER_PULSE_MS = 75L
        private const val DANGER_GAP_MS = 20L
        private const val DANGER_PULSE_COUNT = 7
        private const val DANGER_VOLUME = 0.68f

        // Bell-like harmonic structure: fundamental + inharmonic partials
        // Each entry: [frequency multiplier, amplitude, decay rate multiplier]
        private val BELL_HARMONICS = arrayOf(
            floatArrayOf(1.000f, 1.000f, 1.0f),    // fundamental
            floatArrayOf(2.000f, 0.500f, 1.4f),    // 2nd harmonic (octave)
            floatArrayOf(3.000f, 0.200f, 2.0f),    // 3rd harmonic
            floatArrayOf(4.000f, 0.080f, 2.8f),    // 4th harmonic
            floatArrayOf(5.040f, 0.040f, 3.5f),    // slightly detuned 5th (bell character)
            floatArrayOf(6.100f, 0.020f, 4.0f),    // inharmonic upper partial
        )

        // DANGER uses a brighter, more cutting harmonic set
        private val ALARM_HARMONICS = arrayOf(
            floatArrayOf(1.000f, 1.000f, 1.0f),
            floatArrayOf(2.000f, 0.600f, 1.2f),
            floatArrayOf(3.000f, 0.300f, 1.6f),
            floatArrayOf(4.000f, 0.150f, 2.2f),
            floatArrayOf(5.000f, 0.080f, 3.0f),
        )

        private const val MIN_ALERT_INTERVAL_MS = 1800L
    }

    private var lastAlertTime = 0L
    private var lastRiskSeverity = RiskSeverity.NONE
    private var isPlaying = false
    private var enabled = true

    // Pre-generated PCM samples
    private val cautionPcm: ShortArray by lazy { generateCautionChime() }
    private val urgentPcm: ShortArray by lazy { generateUrgentSequence() }
    private val dangerPcm: ShortArray by lazy { generateDangerAlarm() }

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
            RiskSeverity.URGENT -> {
                val cooldown = (MIN_ALERT_INTERVAL_MS * 1.2).toLong()
                if (elapsed > cooldown || risk.severity.ordinal > lastRiskSeverity.ordinal) {
                    playUrgentAlert()
                    lastAlertTime = now
                }
            }
            RiskSeverity.DANGER -> {
                if (elapsed > MIN_ALERT_INTERVAL_MS || risk.severity.ordinal > lastRiskSeverity.ordinal) {
                    playDangerAlert()
                    lastAlertTime = now
                }
            }
        }
        lastRiskSeverity = risk.severity
    }

    /**
     * Generate a bell-like tone with inharmonic partials and proper envelope.
     *
     * Uses a multi-stage envelope:
     * - Fast attack (3ms) for clean onset
     * - Brief initial decay (brightness loss)
     * - Longer sustain decay (body of the tone)
     * Each harmonic decays at its own rate (higher partials die faster).
     */
    private fun generateBellTone(
        freq: Float,
        durationMs: Long,
        volume: Float,
        harmonics: Array<FloatArray> = BELL_HARMONICS,
    ): ShortArray {
        val durationSec = durationMs / 1000.0
        val numSamples = (SAMPLE_RATE * durationSec).toInt()
        val pcm = ShortArray(numSamples)

        val attackSamples = (SAMPLE_RATE * 0.003).toInt().coerceAtLeast(1) // 3ms attack
        val baseDecayRate = 3.5 / durationSec

        for (i in 0 until numSamples) {
            val t = i.toDouble() / SAMPLE_RATE
            val progress = i.toDouble() / numSamples

            var sample = 0.0
            for (h in harmonics) {
                val harmonicFreq = (freq * h[0]).toDouble()
                if (harmonicFreq > SAMPLE_RATE / 2.0) break
                val amplitude = h[1].toDouble()
                val harmonicDecay = baseDecayRate * h[2].toDouble()

                // Each harmonic has its own decay envelope
                val harmonicEnvelope = exp(-harmonicDecay * t)
                sample += sin(2.0 * PI * harmonicFreq * t) * amplitude * harmonicEnvelope
            }

            // Global envelope: fast attack + smooth decay
            val attack = if (i < attackSamples) {
                // Smooth cubic attack (no click)
                val p = i.toDouble() / attackSamples
                p * p * (3.0 - 2.0 * p)
            } else {
                1.0
            }

            // Slight sustain drop in the first 15% then gradual fade
            val bodyEnvelope = if (progress < 0.15) {
                1.0
            } else {
                exp(-1.5 * (progress - 0.15))
            }

            val value = sample * volume * attack * bodyEnvelope
            pcm[i] = (value * Short.MAX_VALUE).toInt()
                .coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort()
        }
        return pcm
    }

    /**
     * CAUTION: warm two-tone ascending bell chime (D5 -> G5).
     */
    private fun generateCautionChime(): ShortArray {
        val tone1 = generateBellTone(CAUTION_FREQ_1, CAUTION_TONE_MS, CAUTION_VOLUME)
        val gap = ShortArray((SAMPLE_RATE * CAUTION_GAP_MS / 1000.0).toInt())
        val tone2 = generateBellTone(CAUTION_FREQ_2, CAUTION_TONE_MS, CAUTION_VOLUME * 0.90f)

        val combined = ShortArray(tone1.size + gap.size + tone2.size)
        tone1.copyInto(combined, 0)
        gap.copyInto(combined, tone1.size)
        tone2.copyInto(combined, tone1.size + gap.size)
        return combined
    }

    /**
     * URGENT: brighter three-note descending (A5 -> F#5 -> D5).
     */
    private fun generateUrgentSequence(): ShortArray {
        val tone1 = generateBellTone(URGENT_FREQ_1, URGENT_TONE_MS, URGENT_VOLUME)
        val gap = ShortArray((SAMPLE_RATE * URGENT_GAP_MS / 1000.0).toInt())
        val tone2 = generateBellTone(URGENT_FREQ_2, URGENT_TONE_MS, URGENT_VOLUME * 0.95f)
        val tone3 = generateBellTone(URGENT_FREQ_3, URGENT_TONE_MS, URGENT_VOLUME * 1.05f)

        val combined = ShortArray(tone1.size + gap.size + tone2.size + gap.size + tone3.size)
        var offset = 0
        tone1.copyInto(combined, offset); offset += tone1.size
        gap.copyInto(combined, offset); offset += gap.size
        tone2.copyInto(combined, offset); offset += tone2.size
        gap.copyInto(combined, offset); offset += gap.size
        tone3.copyInto(combined, offset)
        return combined
    }

    /**
     * DANGER: rapid alternating high-low alarm pulses.
     * Each pulse uses the alarm harmonic set for a cutting, urgent sound.
     */
    private fun generateDangerAlarm(): ShortArray {
        val pulseSamples = (SAMPLE_RATE * DANGER_PULSE_MS / 1000.0).toInt()
        val gapSamples = (SAMPLE_RATE * DANGER_GAP_MS / 1000.0).toInt()
        val totalPulses = DANGER_PULSE_COUNT
        val totalSamples = totalPulses * (pulseSamples + gapSamples)
        val pcm = ShortArray(totalSamples)

        var offset = 0
        for (p in 0 until totalPulses) {
            val freq = if (p % 2 == 0) DANGER_FREQ_HIGH else DANGER_FREQ_LOW
            // Last two pulses are slightly louder for emphasis
            val volume = DANGER_VOLUME * if (p >= totalPulses - 2) 1.10f else 1.0f

            val tone = generateBellTone(freq, DANGER_PULSE_MS, volume, ALARM_HARMONICS)
            tone.copyInto(pcm, offset)
            offset += pulseSamples

            // Gap (silence)
            for (i in 0 until gapSamples) {
                if (offset < pcm.size) pcm[offset] = 0
                offset++
            }
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

    private fun playUrgentAlert() {
        if (isPlaying) return
        isPlaying = true

        Thread {
            try {
                playPcm(urgentPcm)
            } catch (e: Exception) {
                Log.e(TAG, "Urgent alert failed", e)
            } finally {
                isPlaying = false
            }
        }.start()
    }

    private fun playDangerAlert() {
        if (isPlaying) return
        isPlaying = true

        Thread {
            try {
                // Play alarm twice for maximum attention
                for (i in 0 until 2) {
                    if (!isPlaying) break
                    playPcm(dangerPcm)
                    if (i < 1) {
                        Thread.sleep(60L)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Danger alert failed", e)
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
