package com.nightroadvision.app.inference

/**
 * Merges YOLO detections with supercombo lead vehicle data.
 * Enriches YOLO detections with real distance/velocity from supercombo,
 * or creates new detections for leads not found in YOLO output.
 */
class LeadVehicleMerger {

    companion object {
        private const val CENTER_DIST_THRESHOLD = 0.25f  // max normalized distance for matching
    }

    /**
     * Merge YOLO detections with supercombo lead vehicles.
     *
     * @param yoloDetections detections from YOLO model (camera pixel coordinates)
     * @param supercomboOutput output from supercombo inference
     * @param cameraWidth camera frame width
     * @param cameraHeight camera frame height
     * @return merged list of detections with distance info where available
     */
    fun merge(
        yoloDetections: List<InferenceEngine.Detection>,
        supercomboOutput: SupercomboEngine.SupercomboOutput?,
        cameraWidth: Int,
        cameraHeight: Int,
        vehicleClassIds: Set<Int> = setOf(2, 5, 7),
        syntheticVehicleClassId: Int = 2,
    ): List<InferenceEngine.Detection> {
        if (supercomboOutput == null || supercomboOutput.leads.isEmpty()) {
            return yoloDetections
        }

        val merged = yoloDetections.toMutableList()
        val matchedLeadIndices = mutableSetOf<Int>()

        // For each YOLO detection, try to match with a supercombo lead
        for (i in merged.indices) {
            val det = merged[i]
            if (det.classId !in vehicleClassIds) continue
            val detCenterNorm = Pair(
                det.centerX / cameraWidth,
                det.centerY / cameraHeight
            )

            var bestMatchIdx = -1
            var bestDist = Float.MAX_VALUE

            for (leadIdx in supercomboOutput.leads.indices) {
                if (leadIdx in matchedLeadIndices) continue

                val lead = supercomboOutput.leads[leadIdx]

                // Check center distance in normalized coordinates
                val dx = detCenterNorm.first - lead.normalizedCameraX
                val dy = detCenterNorm.second - lead.normalizedCameraY
                val centerDist = Math.sqrt((dx * dx + dy * dy).toDouble()).toFloat()

                if (centerDist < CENTER_DIST_THRESHOLD && centerDist < bestDist) {
                    bestDist = centerDist
                    bestMatchIdx = leadIdx
                }
            }

            if (bestMatchIdx >= 0) {
                val lead = supercomboOutput.leads[bestMatchIdx]
                matchedLeadIndices.add(bestMatchIdx)

                // Enrich YOLO detection with supercombo distance/velocity
                merged[i] = det.copy(
                    distanceMeters = lead.distance,
                    velocityMps = lead.velocity,
                    isLeadVehicle = true
                )
            }
        }

        // Add unmatched leads as new detections (vehicles detected by supercombo but not YOLO)
        for (leadIdx in supercomboOutput.leads.indices) {
            if (leadIdx in matchedLeadIndices) continue

            val lead = supercomboOutput.leads[leadIdx]

            // Convert normalized camera coords to pixel coords
            val cx = lead.normalizedCameraX * cameraWidth
            val cy = lead.normalizedCameraY * cameraHeight
            val bw = lead.approxBoxWidth * cameraWidth
            val bh = lead.approxBoxHeight * cameraHeight

            merged.add(
                InferenceEngine.Detection(
                    x1 = (cx - bw / 2).coerceIn(0f, cameraWidth.toFloat()),
                    y1 = (cy - bh / 2).coerceIn(0f, cameraHeight.toFloat()),
                    x2 = (cx + bw / 2).coerceIn(0f, cameraWidth.toFloat()),
                    y2 = (cy + bh / 2).coerceIn(0f, cameraHeight.toFloat()),
                    confidence = lead.probability,
                    classId = syntheticVehicleClassId,
                    className = "car",
                    cameraWidth = cameraWidth,
                    cameraHeight = cameraHeight,
                    distanceMeters = lead.distance,
                    velocityMps = lead.velocity,
                    isLeadVehicle = true
                )
            )
        }

        return merged
    }
}
