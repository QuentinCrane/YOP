package com.nightroadvision.app.tracking

import android.graphics.RectF

// ---------------------------------------------------------------------------
// Detection -- input from the inference / YOLO pipeline
// ---------------------------------------------------------------------------

/**
 * A single object detection produced by the inference engine on one frame.
 *
 * @param classId  COCO (or custom) class index.
 * @param confidence  Model confidence in [0, 1].
 * @param box  Bounding box in absolute pixel coordinates.
 */
data class Detection(
    val classId: Int,
    val confidence: Float,
    val box: RectF
)

// ---------------------------------------------------------------------------
// Track -- represents one tracked object across frames
// ---------------------------------------------------------------------------

/**
 * A persistent track that is maintained across frames by [ObjectTracker].
 *
 * @param id  Unique, monotonically increasing identifier assigned at creation.
 * @param classId  Class label inherited from the originating detection.
 * @param confidence  Latest (smoothed) confidence score.
 * @param box  Raw bounding box from the most recent matched detection.
 * @param smoothedBox  Exponentially smoothed bounding box for display.
 * @param ageFrames  Total number of frames this track has existed.
 * @param missedFrames  Consecutive frames without a matched detection.
 * @param confirmed  True once the track has been matched at least
 *                   [CONFIRM_THRESHOLD] times consecutively.
 */
data class Track(
    val id: Int,
    val classId: Int,
    val confidence: Float,
    val box: RectF,
    val smoothedBox: RectF,
    val ageFrames: Int,
    val missedFrames: Int,
    val confirmed: Boolean
)

// ---------------------------------------------------------------------------
// ObjectTracker -- ByteTrack-inspired multi-object tracker
// ---------------------------------------------------------------------------

/**
 * A lightweight, ByteTrack-inspired multi-object tracker suitable for
 * real-time on-device use.
 *
 * ### Matching strategy
 * 1. Compute pairwise IoU between every existing track and every new
 *    detection whose classId matches the track's classId.
 * 2. Greedily assign detections to tracks in descending IoU order,
 *    requiring IoU >= [IOU_THRESHOLD].
 * 3. Unmatched detections spawn new tracks; unmatched tracks have their
 *    [Track.missedFrames] counter incremented.
 * 4. Tracks whose missedFrames exceeds [MAX_MISSED] are removed.
 *
 * ### Smoothing
 * - Bounding-box coordinates are smoothed with an exponential moving
 *   average (EMA) using alpha = [BOX_SMOOTH_ALPHA].
 * - Confidence is smoothed separately with alpha = [CONF_SMOOTH_ALPHA].
 */
class ObjectTracker(initialConfig: Config = Config()) {

    data class Config(
        val iouThreshold: Float = 0.15f,
        val confirmFrames: Int = 1,
        val maxMissedFrames: Int = 15,
        val boxSmoothing: Float = 0.50f,
        val confidenceSmoothing: Float = 0.40f,
        val maxVisibleMissedFrames: Int = 4,
    ) {
        fun sanitized(): Config = copy(
            iouThreshold = iouThreshold.coerceIn(0.05f, 0.80f),
            confirmFrames = confirmFrames.coerceIn(1, 5),
            maxMissedFrames = maxMissedFrames.coerceIn(1, 30),
            boxSmoothing = boxSmoothing.coerceIn(0.05f, 1.0f),
            confidenceSmoothing = confidenceSmoothing.coerceIn(0.05f, 1.0f),
            maxVisibleMissedFrames = maxVisibleMissedFrames.coerceIn(0, 5),
        )
    }

    // ---- constants ----------------------------------------------------------

    companion object {
        /**
         * Compute Intersection-over-Union of two axis-aligned rectangles.
         * Returns 0 when either rectangle is empty or they do not overlap.
         */
        fun computeIoU(a: RectF, b: RectF): Float {
            val interLeft = maxOf(a.left, b.left)
            val interTop = maxOf(a.top, b.top)
            val interRight = minOf(a.right, b.right)
            val interBottom = minOf(a.bottom, b.bottom)

            if (interRight <= interLeft || interBottom <= interTop) return 0f

            val interArea = (interRight - interLeft) * (interBottom - interTop)
            val aArea = (a.right - a.left) * (a.bottom - a.top)
            val bArea = (b.right - b.left) * (b.bottom - b.top)
            val unionArea = aArea + bArea - interArea

            return if (unionArea > 0f) interArea / unionArea else 0f
        }

        /** Linear interpolation between [start] and [end] by factor [alpha]. */
        internal fun lerp(start: Float, end: Float, alpha: Float): Float =
            start + alpha * (end - start)
    }

    // ---- mutable state ------------------------------------------------------

    /** All live tracks (confirmed and tentative). */
    private val tracks = mutableListOf<InternalTrack>()

    @Volatile
    private var config = initialConfig.sanitized()

    /** Monotonically increasing ID counter. */
    private var nextId = 1

    // ---- public API ---------------------------------------------------------

    /**
     * Feed a new frame's detections into the tracker and return the current
     * set of **confirmed** tracks.
     *
     * @param detections  Raw detections from the inference engine for this frame.
     * @return  Snapshot list of all confirmed [Track] objects after the update.
     */
    fun update(detections: List<Detection>): List<Track> {
        // -- 1. Build IoU cost matrix (class-matched) -------------------------
        // Use parallel arrays instead of data class to reduce allocation
        val candTrackIdx = mutableListOf<Int>()
        val candDetIdx = mutableListOf<Int>()
        val candIou = mutableListOf<Float>()

        for (ti in tracks.indices) {
            val track = tracks[ti]
            for (di in detections.indices) {
                val det = detections[di]
                if (det.classId != track.classId) continue
                val iou = computeIoU(track.box, det.box)
                if (iou >= config.iouThreshold) {
                    candTrackIdx.add(ti)
                    candDetIdx.add(di)
                    candIou.add(iou)
                }
            }
        }

        // -- 2. Greedy matching in descending IoU order -----------------------
        val indices = (0 until candIou.size).sortedByDescending { candIou[it] }

        val matchedTracks = BooleanArray(tracks.size)
        val matchedDets = BooleanArray(detections.size)
        // trackIdx -> detectionIdx (use IntArray with -1 sentinel)
        val assignment = IntArray(tracks.size) { -1 }

        for (idx in indices) {
            val ti = candTrackIdx[idx]
            val di = candDetIdx[idx]
            if (matchedTracks[ti] || matchedDets[di]) continue
            matchedTracks[ti] = true
            matchedDets[di] = true
            assignment[ti] = di
        }

        // -- 3. Update matched tracks -----------------------------------------
        for (ti in tracks.indices) {
            val track = tracks[ti]
            val di = assignment[ti]
            if (di >= 0) {
                tracks[ti] = track.applyDetection(detections[di], config)
            } else {
                // Unmatched -- increment missed counter, age the track
                tracks[ti] = track.copy(
                    missedFrames = track.missedFrames + 1,
                    ageFrames = track.ageFrames + 1
                )
            }
        }

        // -- 4. Remove dead tracks --------------------------------------------
        tracks.removeIf { it.missedFrames > config.maxMissedFrames }

        // -- 5. Create new tracks for unmatched detections --------------------
        for (di in detections.indices) {
            if (matchedDets[di]) continue
            val det = detections[di]
            tracks.add(
                InternalTrack(
                    id = nextId++,
                    classId = det.classId,
                    confidence = det.confidence,
                    box = RectF(det.box),
                    smoothedBox = RectF(det.box),
                    ageFrames = 1,
                    missedFrames = 0,
                    consecutiveMatches = 1
                )
            )
        }

        // -- 6. Return confirmed tracks as public snapshot --------------------
        return tracks
            .filter {
                it.consecutiveMatches >= config.confirmFrames &&
                    it.missedFrames <= config.maxVisibleMissedFrames
            }
            .map { it.toPublicTrack(config.confirmFrames) }
    }

    /** Applies tracking tuning live without discarding active IDs. */
    fun updateConfig(value: Config) {
        config = value.sanitized()
    }

    /**
     * Reset the tracker, discarding all tracks and resetting the ID counter.
     */
    fun reset() {
        tracks.clear()
        nextId = 1
    }

    /** Total number of live tracks (confirmed + tentative). Exposed for diagnostics. */
    val trackCount: Int get() = tracks.size

    /** Public snapshot of every live track. Exposed for diagnostics. */
    val allTracks: List<Track> get() = tracks.map { it.toPublicTrack(config.confirmFrames) }

    // ---- internal track representation --------------------------------------

    /**
     * Internal mutable representation that additionally tracks
     * [consecutiveMatches], which is not part of the public [Track] data class.
     */
    private data class InternalTrack(
        val id: Int,
        val classId: Int,
        val confidence: Float,
        val box: RectF,
        val smoothedBox: RectF,
        val ageFrames: Int,
        val missedFrames: Int,
        val consecutiveMatches: Int
    ) {
        /**
         * Produce a new [InternalTrack] that incorporates a freshly matched
         * detection, applying EMA smoothing to box and confidence.
         */
        fun applyDetection(det: Detection, config: Config): InternalTrack {
            // EMA-smooth the bounding box
            val newSmoothedBox = RectF(
                lerp(smoothedBox.left, det.box.left, config.boxSmoothing),
                lerp(smoothedBox.top, det.box.top, config.boxSmoothing),
                lerp(smoothedBox.right, det.box.right, config.boxSmoothing),
                lerp(smoothedBox.bottom, det.box.bottom, config.boxSmoothing)
            )

            // EMA-smooth confidence
            val newConfidence = lerp(confidence, det.confidence, config.confidenceSmoothing)

            return InternalTrack(
                id = id,
                classId = classId,
                confidence = newConfidence,
                box = RectF(det.box),
                smoothedBox = newSmoothedBox,
                ageFrames = ageFrames + 1,
                missedFrames = 0,
                consecutiveMatches = consecutiveMatches + 1
            )
        }

        /** Convert to the public immutable [Track] snapshot. */
        fun toPublicTrack(confirmFrames: Int): Track = Track(
            id = id,
            classId = classId,
            confidence = confidence,
            box = RectF(box),
            smoothedBox = RectF(smoothedBox),
            ageFrames = ageFrames,
            missedFrames = missedFrames,
            confirmed = consecutiveMatches >= confirmFrames
        )
    }
}
