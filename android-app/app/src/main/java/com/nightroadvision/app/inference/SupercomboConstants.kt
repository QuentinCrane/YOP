package com.nightroadvision.app.inference

/**
 * Constants for the openpilot supercombo driving model (v0.8.x).
 * Verified against actual ONNX model from MTammvee/openpilot-supercombo-model.
 */
object SupercomboConstants {

    // Model input dimensions
    const val MODEL_WIDTH = 256
    const val MODEL_HEIGHT = 128
    const val N_FRAMES = 2
    const val YUV_CHANNELS = 6  // Y + U + V per frame
    const val STACKED_CHANNELS = N_FRAMES * YUV_CHANNELS  // 12

    // Auxiliary input sizes
    const val FEATURE_LEN = 512       // GRU hidden state
    const val DESIRE_LEN = 8          // lane change intent
    const val TRAFFIC_CONVENTION_LEN = 2  // LHT/RHT

    // Input tensor names (verified from ONNX model)
    const val IN_IMGS = "input_imgs"
    const val IN_DESIRE = "desire"
    const val IN_TRAFFIC_CONVENTION = "traffic_convention"
    const val IN_INITIAL_STATE = "initial_state"

    // Output tensor name
    const val OUT_OUTPUTS = "outputs"
    const val OUTPUT_SIZE = 6409      // total flat output vector size

    // Output slice offsets — verified against MTammvee/openpilot-supercombo-model parsing code
    // and commaai/openpilot constants.py (v0.8.x era)
    //
    // Layout (each section has MHP mean + std + weights):
    //   plan:     33×15×5 mean + 33×15×5 std + 5 weights = 4955
    //   lanes:    4×33×2×5 mean+std+weights = 528
    //   road:     2×33×2×5 mean+std+weights = 264
    //   (gap):    55 (road extra / transition)
    //   lead:     3×6×4×3 mean + 3×6×4×3 std + 9 weights = 441
    //   lead_prob: 3
    //   ... then desire(72), meta(55), etc.
    const val PLAN_OFFSET = 0
    const val PLAN_SIZE = 4955        // 33*15*5 mean + 33*15*5 std + 5 weights

    const val LANE_LINES_OFFSET = 4955  // plan end
    const val LANE_LINES_SIZE = 528     // 4*33*2*5 (mean+std+weights for 5 MHP)

    const val ROAD_EDGES_OFFSET = 5483  // lanes end
    const val ROAD_EDGES_SIZE = 264     // 2*33*2*5 (mean+std+weights for 5 MHP)

    // Lead vehicle offsets — verified from MTammvee parsing code
    // lead data starts at 5810 (= 5755 road_end + 55 gap)
    const val LEAD_OFFSET = 5810
    const val LEAD_SIZE = 216         // 3 leads × 6 timesteps × 4 values × 3 MHP hypotheses
    const val LEAD_STD_SIZE = 216     // same size for standard deviations
    const val LEAD_WEIGHTS_SIZE = 9   // 3 leads × 3 MHP weights
    const val LEAD_PROB_OFFSET = LEAD_OFFSET + LEAD_SIZE + LEAD_STD_SIZE + LEAD_WEIGHTS_SIZE  // = 5963
    const val LEAD_PROB_SIZE = 3      // 3 lead detection probabilities

    // Lead dimensions
    const val LEAD_MAX = 3            // max 3 lead vehicles
    const val LEAD_TRAJ_LEN = 6      // 6 time steps
    const val LEAD_WIDTH = 4          // (x, y, velocity, acceleration)

    // Lead time indices
    val LEAD_T_IDXS = floatArrayOf(0f, 2f, 4f, 6f, 8f, 10f)
    val LEAD_T_OFFSETS = floatArrayOf(0f, 2f, 4f)

    // Detection probability threshold for leads
    const val LEAD_PROB_THRESHOLD = 0.3f

    // Inference frequency
    const val MODEL_RUN_FREQ_HZ = 20
}
