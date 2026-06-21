package com.nightroadvision.app.alert

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.SystemClock
import android.util.Log
import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.sin

/**
 * Sound style presets for the three-level alert system.
 * Each preset defines a distinct sonic character while maintaining
 * the Tesla-inspired design language of clarity and urgency.
 */
enum class AlertSoundStyle(val label: String, val description: String) {
    TESLA("Tesla", "温暖钟声，渐升音调"),
    OPENPILOT("Openpilot", "柔和双音，温和提醒"),
    RADAR("雷达", "电子脉冲，清晰精准"),
    CHIME("旋律", "悦耳三音，音乐感"),
    BUZZER("蜂鸣", "急促方波，强烈警示"),
}

/**
 * Parameters for a single alert level tone.
 */
data class ToneParams(
    val frequencies: FloatArray,       // one or more frequencies (Hz)
    val durationMs: Long,              // tone duration per note
    val gapMs: Long,                   // gap between notes
    val volume: Float,                 // 0..1
    val repeat: Int = 1,               // how many times to repeat the sequence
    val harmonicType: HarmonicType = HarmonicType.BELL,
)

enum class HarmonicType { BELL, ELECTRONIC, PURE, MUSICAL, SQUARE }

/**
 * Complete sound preset: one ToneParams per alert level.
 */
data class SoundPreset(
    val caution: ToneParams,
    val urgent: ToneParams,
    val danger: ToneParams,
)

/**
 * Three-level audio alert player with switchable sound presets.
 *
 * Presets:
 * - TESLA: warm bell chimes with inharmonic partials
 * - OPENPILOT: softer, lower-pitched gentle chimes
 * - RADAR: short electronic pings with fast decay
 * - CHIME: pleasant musical triad notes
 * - BUZZER: harsh square-wave pulses
 */
class AlertSoundPlayer {

    companion object {
        private const val TAG = "AlertSoundPlayer"
        private const val SAMPLE_RATE = 44100
        private const val MIN_ALERT_INTERVAL_MS = 800L

        // ── Harmonic structures ──

        // Bell-like: inharmonic partials for warm, complex tone
        private val BELL_HARMONICS = arrayOf(
            floatArrayOf(1.000f, 1.000f, 1.0f),
            floatArrayOf(2.000f, 0.500f, 1.4f),
            floatArrayOf(3.000f, 0.200f, 2.0f),
            floatArrayOf(4.000f, 0.080f, 2.8f),
            floatArrayOf(5.040f, 0.040f, 3.5f),
            floatArrayOf(6.100f, 0.020f, 4.0f),
        )

        // Electronic: cleaner, fewer harmonics for a clinical sound
        private val ELECTRONIC_HARMONICS = arrayOf(
            floatArrayOf(1.000f, 1.000f, 1.0f),
            floatArrayOf(2.000f, 0.350f, 1.8f),
            floatArrayOf(3.000f, 0.100f, 3.0f),
            floatArrayOf(4.000f, 0.030f, 4.5f),
        )

        // Pure: just fundamental + slight overtone
        private val PURE_HARMONICS = arrayOf(
            floatArrayOf(1.000f, 1.000f, 1.0f),
            floatArrayOf(2.000f, 0.150f, 2.0f),
        )

        // Musical: rich but harmonious, consonant intervals
        private val MUSICAL_HARMONICS = arrayOf(
            floatArrayOf(1.000f, 1.000f, 1.0f),
            floatArrayOf(2.000f, 0.450f, 1.3f),
            floatArrayOf(3.000f, 0.250f, 1.8f),
            floatArrayOf(4.000f, 0.120f, 2.4f),
            floatArrayOf(5.000f, 0.050f, 3.2f),
        )

        // Square-wave-like: odd harmonics dominant for buzzer character
        private val SQUARE_HARMONICS = arrayOf(
            floatArrayOf(1.000f, 1.000f, 1.0f),
            floatArrayOf(3.000f, 0.333f, 1.0f),
            floatArrayOf(5.000f, 0.200f, 1.0f),
            floatArrayOf(7.000f, 0.143f, 1.0f),
            floatArrayOf(9.000f, 0.111f, 1.0f),
        )

        // ── Preset definitions ──

        private val PRESETS = mapOf(
            AlertSoundStyle.TESLA to SoundPreset(
                caution = ToneParams(
                    frequencies = floatArrayOf(587.33f, 783.99f),  // D5→G5 ascending
                    durationMs = 160, gapMs = 30, volume = 0.50f,
                    harmonicType = HarmonicType.BELL,
                ),
                urgent = ToneParams(
                    frequencies = floatArrayOf(880.00f, 740.00f, 587.33f),  // A5→F#5→D5
                    durationMs = 110, gapMs = 25, volume = 0.58f,
                    harmonicType = HarmonicType.BELL,
                ),
                danger = ToneParams(
                    frequencies = floatArrayOf(1174.66f, 783.99f),  // D6/G5 alternating
                    durationMs = 75, gapMs = 20, volume = 0.68f, repeat = 2,
                    harmonicType = HarmonicType.ELECTRONIC,
                ),
            ),
            AlertSoundStyle.OPENPILOT to SoundPreset(
                caution = ToneParams(
                    frequencies = floatArrayOf(440.00f, 554.37f),  // A4→C#5 soft ascending
                    durationMs = 200, gapMs = 40, volume = 0.40f,
                    harmonicType = HarmonicType.MUSICAL,
                ),
                urgent = ToneParams(
                    frequencies = floatArrayOf(523.25f, 440.00f, 349.23f),  // C5→A4→F4 gentle desc
                    durationMs = 140, gapMs = 30, volume = 0.48f,
                    harmonicType = HarmonicType.MUSICAL,
                ),
                danger = ToneParams(
                    frequencies = floatArrayOf(880.00f, 659.25f),  // A5/E5 alternating
                    durationMs = 90, gapMs = 25, volume = 0.58f, repeat = 2,
                    harmonicType = HarmonicType.BELL,
                ),
            ),
            AlertSoundStyle.RADAR to SoundPreset(
                caution = ToneParams(
                    frequencies = floatArrayOf(1200.00f, 1400.00f),  // high pings
                    durationMs = 60, gapMs = 15, volume = 0.45f,
                    harmonicType = HarmonicType.ELECTRONIC,
                ),
                urgent = ToneParams(
                    frequencies = floatArrayOf(1600.00f, 1400.00f, 1200.00f),  // descending pings
                    durationMs = 50, gapMs = 12, volume = 0.52f,
                    harmonicType = HarmonicType.ELECTRONIC,
                ),
                danger = ToneParams(
                    frequencies = floatArrayOf(2000.00f, 1400.00f),  // rapid high-low
                    durationMs = 40, gapMs = 10, volume = 0.62f, repeat = 3,
                    harmonicType = HarmonicType.ELECTRONIC,
                ),
            ),
            AlertSoundStyle.CHIME to SoundPreset(
                caution = ToneParams(
                    frequencies = floatArrayOf(523.25f, 659.25f),  // C5→E5 major third
                    durationMs = 180, gapMs = 35, volume = 0.45f,
                    harmonicType = HarmonicType.MUSICAL,
                ),
                urgent = ToneParams(
                    frequencies = floatArrayOf(523.25f, 659.25f, 783.99f),  // C5→E5→G5 triad
                    durationMs = 130, gapMs = 25, volume = 0.52f,
                    harmonicType = HarmonicType.MUSICAL,
                ),
                danger = ToneParams(
                    frequencies = floatArrayOf(1046.50f, 783.99f),  // C6/G5 alternating
                    durationMs = 80, gapMs = 18, volume = 0.62f, repeat = 2,
                    harmonicType = HarmonicType.MUSICAL,
                ),
            ),
            AlertSoundStyle.BUZZER to SoundPreset(
                caution = ToneParams(
                    frequencies = floatArrayOf(400.00f, 500.00f),  // low buzz ascending
                    durationMs = 100, gapMs = 20, volume = 0.42f,
                    harmonicType = HarmonicType.SQUARE,
                ),
                urgent = ToneParams(
                    frequencies = floatArrayOf(600.00f, 500.00f, 400.00f),  // descending buzz
                    durationMs = 80, gapMs = 15, volume = 0.52f,
                    harmonicType = HarmonicType.SQUARE,
                ),
                danger = ToneParams(
                    frequencies = floatArrayOf(800.00f, 500.00f),  // rapid alternating buzz
                    durationMs = 60, gapMs = 12, volume = 0.65f, repeat = 3,
                    harmonicType = HarmonicType.SQUARE,
                ),
            ),
        )
    }

    private var lastAlertTime = 0L
    private var lastRiskSeverity = RiskSeverity.NONE
    @Volatile private var isPlaying = false
    @Volatile private var enabled = true
    @Volatile private var currentStyle = AlertSoundStyle.TESLA
    @Volatile private var released = false
    private val stateLock = Any()

    // PCM buffers — regenerated when style changes
    @Volatile private var cautionPcm: ShortArray? = null
    @Volatile private var urgentPcm: ShortArray? = null
    @Volatile private var dangerPcm: ShortArray? = null
    @Volatile private var prewarmed = false

    private val playbackQueue = java.util.concurrent.LinkedBlockingDeque<() -> Unit>()
    @Volatile private var activeTrack: AudioTrack? = null

    // Dedicated playback thread
    private val playbackThread = Thread {
        while (!Thread.currentThread().isInterrupted) {
            try {
                val task = playbackQueue.takeFirst()
                task()
            } catch (_: InterruptedException) {
                break
            } catch (e: Exception) {
                Log.e(TAG, "Playback error", e)
            }
        }
    }.apply { isDaemon = true; start() }

    fun setEnabled(value: Boolean) {
        synchronized(stateLock) {
            if (released) return
            enabled = value
            if (!enabled) stopAlerts()
        }
    }

    fun isEnabled(): Boolean = enabled

    /**
     * Switch the active sound style. Regenerates all PCM buffers.
     */
    fun setStyle(style: AlertSoundStyle) {
        synchronized(stateLock) {
            if (released || style == currentStyle && prewarmed) return
            currentStyle = style
            prewarmed = false
            cautionPcm = null
            urgentPcm = null
            dangerPcm = null
        }
        prewarm()
    }

    fun getStyle(): AlertSoundStyle = currentStyle

    /**
     * Pre-generate PCM buffers for the current style.
     */
    fun prewarm() {
        val styleToGenerate = synchronized(stateLock) {
            if (released || prewarmed) return
            prewarmed = true
            currentStyle
        }
        playbackQueue.offerFirst {
            try {
                val preset = checkNotNull(PRESETS[styleToGenerate])
                val caution = generateSequence(preset.caution)
                val urgent = generateSequence(preset.urgent)
                val danger = generateSequence(preset.danger)
                synchronized(stateLock) {
                    if (!released && currentStyle == styleToGenerate) {
                        cautionPcm = caution
                        urgentPcm = urgent
                        dangerPcm = danger
                    }
                }
                Log.d(TAG, "Pre-warmed PCM for style: ${styleToGenerate.label}")
            } catch (e: Exception) {
                synchronized(stateLock) {
                    if (currentStyle == styleToGenerate) prewarmed = false
                }
                Log.e(TAG, "PCM pre-warm failed", e)
            }
        }
    }

    /**
     * Preview play a single alert level (for settings UI).
     */
    fun previewPlay(severity: RiskSeverity) {
        if (released) return
        val preset = PRESETS[currentStyle] ?: return
        val params = when (severity) {
            RiskSeverity.CAUTION -> preset.caution
            RiskSeverity.URGENT -> preset.urgent
            RiskSeverity.DANGER -> preset.danger
            RiskSeverity.NONE -> return
        }
        playbackQueue.offerLast {
            runPlayback("Preview") {
                val pcm = generateSequence(params)
                playPcm(pcm)
            }
        }
    }

    fun onRiskChanged(risk: RidingRisk) {
        synchronized(stateLock) {
          if (!enabled || released) return

          val now = SystemClock.elapsedRealtime()
          val elapsed = now - lastAlertTime

          when (risk.severity) {
            RiskSeverity.NONE -> {
                lastRiskSeverity = RiskSeverity.NONE
                return
            }
            RiskSeverity.CAUTION -> {
                if (lastRiskSeverity != RiskSeverity.CAUTION || elapsed > MIN_ALERT_INTERVAL_MS * 2) {
                    if (playCautionAlert()) lastAlertTime = now
                }
            }
            RiskSeverity.URGENT -> {
                if (elapsed > MIN_ALERT_INTERVAL_MS || risk.severity.ordinal > lastRiskSeverity.ordinal) {
                    if (playUrgentAlert()) lastAlertTime = now
                }
            }
            RiskSeverity.DANGER -> {
                if (elapsed > MIN_ALERT_INTERVAL_MS / 2 || risk.severity.ordinal > lastRiskSeverity.ordinal) {
                    if (playDangerAlert()) lastAlertTime = now
                }
            }
          }
          lastRiskSeverity = risk.severity
        }
    }

    // ── PCM Generation ──

    private fun getHarmonics(type: HarmonicType): Array<FloatArray> = when (type) {
        HarmonicType.BELL -> BELL_HARMONICS
        HarmonicType.ELECTRONIC -> ELECTRONIC_HARMONICS
        HarmonicType.PURE -> PURE_HARMONICS
        HarmonicType.MUSICAL -> MUSICAL_HARMONICS
        HarmonicType.SQUARE -> SQUARE_HARMONICS
    }

    /**
     * Generate a complete alert sequence from ToneParams.
     * Plays all frequencies in order, repeated `repeat` times.
     */
    private fun generateSequence(params: ToneParams): ShortArray {
        val harmonics = getHarmonics(params.harmonicType)
        val tones = mutableListOf<ShortArray>()
        val gap = ShortArray((SAMPLE_RATE * params.gapMs / 1000.0).toInt())

        for (r in 0 until params.repeat) {
            for ((i, freq) in params.frequencies.withIndex()) {
                // Slight volume variation for character
                val vol = params.volume * when {
                    params.frequencies.size == 1 -> 1.0f
                    i == params.frequencies.size - 1 -> 0.92f  // last note slightly softer
                    else -> 1.0f
                }
                tones.add(generateTone(freq, params.durationMs, vol, harmonics))
                if (i < params.frequencies.size - 1 || r < params.repeat - 1) {
                    tones.add(gap)
                }
            }
        }

        val totalSize = tones.sumOf { it.size }
        val combined = ShortArray(totalSize)
        var offset = 0
        for (tone in tones) {
            tone.copyInto(combined, offset)
            offset += tone.size
        }
        return combined
    }

    /**
     * Generate a single tone with harmonics and envelope.
     */
    private fun generateTone(
        freq: Float,
        durationMs: Long,
        volume: Float,
        harmonics: Array<FloatArray>,
    ): ShortArray {
        val durationSec = durationMs / 1000.0
        val numSamples = (SAMPLE_RATE * durationSec).toInt()
        val pcm = ShortArray(numSamples)

        val attackSamples = (SAMPLE_RATE * 0.003).toInt().coerceAtLeast(1)
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
                val harmonicEnvelope = exp(-harmonicDecay * t)
                sample += sin(2.0 * PI * harmonicFreq * t) * amplitude * harmonicEnvelope
            }

            // Smooth cubic attack
            val attack = if (i < attackSamples) {
                val p = i.toDouble() / attackSamples
                p * p * (3.0 - 2.0 * p)
            } else {
                1.0
            }

            // Body envelope: sustain then fade
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

    // ── Playback ──

    private fun playCautionAlert(): Boolean {
        val pcm = cautionPcm ?: return false
        return synchronized(stateLock) {
            if (released || !enabled) return false
            playbackQueue.offerLast {
                runPlayback("Caution alert") { playPcm(pcm) }
            }
        }
    }

    private fun playUrgentAlert(): Boolean {
        val pcm = urgentPcm ?: return false
        synchronized(stateLock) {
            if (released || !enabled || isPlaying) return false
            isPlaying = true
        }
        val queued = playbackQueue.offerFirst {
            try {
                runPlayback("Urgent alert") { playPcm(pcm) }
            } finally {
                isPlaying = false
            }
        }
        if (!queued) isPlaying = false
        return queued
    }

    private fun playDangerAlert(): Boolean {
        val pcm = dangerPcm ?: return false
        synchronized(stateLock) {
            if (released || !enabled || isPlaying) return false
            isPlaying = true
        }
        val queued = playbackQueue.offerFirst {
            try {
                runPlayback("Danger alert") {
                    for (i in 0 until 2) {
                        if (!isPlaying) break
                        playPcm(pcm)
                        if (i < 1) Thread.sleep(40L)
                    }
                }
            } finally {
                isPlaying = false
            }
        }
        if (!queued) isPlaying = false
        return queued
    }

    private inline fun runPlayback(label: String, block: () -> Unit) {
        try {
            block()
        } catch (_: InterruptedException) {
            Thread.currentThread().interrupt()
        } catch (e: Exception) {
            Log.e(TAG, "$label failed", e)
        }
    }

    private fun playPcm(pcm: ShortArray) {
        val bufferSize = AudioTrack.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        var track: AudioTrack? = null
        try {
            val audioTrack = AudioTrack.Builder()
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
            track = audioTrack
            check(audioTrack.state == AudioTrack.STATE_INITIALIZED) { "AudioTrack initialization failed" }
            activeTrack = audioTrack
            val written = audioTrack.write(pcm, 0, pcm.size)
            check(written >= 0) { "AudioTrack write failed: $written" }
            audioTrack.play()
            val durationMs = (pcm.size.toLong() * 1000) / SAMPLE_RATE
            Thread.sleep(durationMs + 20)
        } finally {
            if (activeTrack === track) activeTrack = null
            track?.let { audioTrack ->
                runCatching { audioTrack.stop() }
                runCatching { audioTrack.release() }
            }
        }
    }

    private fun stopAlerts() {
        synchronized(stateLock) {
            isPlaying = false
            lastRiskSeverity = RiskSeverity.NONE
            lastAlertTime = 0L
            playbackQueue.clear()
        }
        activeTrack?.let { track ->
            runCatching { track.stop() }
        }
    }

    fun release() {
        synchronized(stateLock) {
            if (released) return
            released = true
        }
        stopAlerts()
        playbackQueue.clear()
        playbackThread.interrupt()
    }
}
