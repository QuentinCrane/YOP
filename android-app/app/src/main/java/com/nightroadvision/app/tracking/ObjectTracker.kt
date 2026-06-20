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
class ObjectTracker {

    // ---- constants ----------------------------------------------------------

    companion object {
        /** Minimum IoU required to consider a detection-to-track match valid. */
        private const val IOU_THRESHOLD = 0.3f

        /** Number of consecutive matches before a track is promoted to confirmed. */
        private const val CONFIRM_THRESHOLD = 2

        /** A track is dropped after this many consecutive unmatched frames. */
        private const val MAX_MISSED = 10

        /** EMA smoothing factor for bounding box (higher = more responsive). */
        private const val BOX_SMOOTH_ALPHA = 0.6f

        /** EMA smoothing factor for confidence score. */
        private const val CONF_SMOOTH_ALPHA = 0.4f

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
        data class CandidatePair(val trackIdx: Int, val detIdx: Int, val iou: Float)

        val candidates = mutableListOf<CandidatePair>()
        for (ti in tracks.indices) {
            val track = tracks[ti]
            for (di in detections.indices) {
                val det = detections[di]
                if (det.classId != track.classId) continue
                val iou = computeIoU(track.box, det.box)
                if (iou >= IOU_THRESHOLD) {
                    candidates.add(CandidatePair(ti, di, iou))
                }
            }
        }

        // -- 2. Greedy matching in descending IoU order -----------------------
        candidates.sortByDescending { it.iou }

        val matchedTrackIndices = mutableSetOf<Int>()
        val matchedDetIndices = mutableSetOf<Int>()
        // trackIdx -> detectionIdx
        val assignment = mutableMapOf<Int, Int>()

        for (c in candidates) {
            if (c.trackIdx in matchedTrackIndices) continue
            if (c.detIdx in matchedDetIndices) continue
            matchedTrackIndices.add(c.trackIdx)
            matchedDetIndices.add(c.detIdx)
            assignment[c.trackIdx] = c.detIdx
        }

        // -- 3. Update matched tracks -----------------------------------------
        for (ti in tracks.indices) {
            val track = tracks[ti]
            if (ti in assignment) {
                val det = detections[assignment[ti]!!]
                tracks[ti] = track.applyDetection(det)
            } else {
                // Unmatched -- increment missed counter, age the track
                tracks[ti] = track.copy(
                    missedFrames = track.missedFrames + 1,
                    ageFrames = track.ageFrames + 1
                )
            }
        }

        // -- 4. Remove dead tracks --------------------------------------------
        tracks.removeIf { it.missedFrames > MAX_MISSED }

        // -- 5. Create new tracks for unmatched detections --------------------
        for (di in detections.indices) {
            if (di in matchedDetIndices) continue
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
            .filter { it.consecutiveMatches >= CONFIRM_THRESHOLD }
            .map { it.toPublicTrack() }
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
    val allTracks: List<Track> get() = tracks.map { it.toPublicTrack() }

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
        fun applyDetection(det: Detection): InternalTrack {
            // EMA-smooth the bounding box
            val newSmoothedBox = RectF(
                lerp(smoothedBox.left, det.box.left, BOX_SMOOTH_ALPHA),
                lerp(smoothedBox.top, det.box.top, BOX_SMOOTH_ALPHA),
                lerp(smoothedBox.right, det.box.right, BOX_SMOOTH_ALPHA),
                lerp(smoothedBox.bottom, det.box.bottom, BOX_SMOOTH_ALPHA)
            )

            // EMA-smooth confidence
            val newConfidence = lerp(confidence, det.confidence, CONF_SMOOTH_ALPHA)

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
        fun toPublicTrack(): Track = Track(
            id = id,
            classId = classId,
            confidence = confidence,
            box = RectF(box),
            smoothedBox = RectF(smoothedBox),
            ageFrames = ageFrames,
            missedFrames = missedFrames,
            confirmed = consecutiveMatches >= CONFIRM_THRESHOLD
        )
    }
}
