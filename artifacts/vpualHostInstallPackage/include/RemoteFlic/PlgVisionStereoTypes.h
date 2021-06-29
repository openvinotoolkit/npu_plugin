// {% copyright %}
///
/// @file
///

#ifndef PLG_VISIONSTEREO_TYPES_H
#define PLG_VISIONSTEREO_TYPES_H
#include <iostream>
#include <vector>
#include <swcFrameTypes.h>

#include "Flic.h"
#include "VpuData.h"

// Macros to align size according to the vpuip RgnAllocator adjustments of memory allocation
#define LEON_CACHE_ALIGNMENT 0x40          // RTEMS cache line aligned
#define DCALGN(x) (((x) + 0x3f) & (~0x3f)) // Macro to align size to 64Bytes (vpuip Leon L2-cache line)
#define ADD_VPUIP_RGNALLOC_METADATA_PADD(x) (DCALGN(x) + 0x40)
#define AJUST_SIZE_TO_VPUIP_RGNALLOC_CREATE(x) (ADD_VPUIP_RGNALLOC_METADATA_PADD(x) + LEON_CACHE_ALIGNMENT)

// Macros for the default configs of the stereo component
#define RATIO_THR (200)
#define DIV_FACTOR (1)
#define INVALID_THR (4)
#define INVALID_DSP (0)
#define OUT_REMOVE_THR (15)
#define OUT_CENSUS_THR (32)
#define OUT_DIFF_THR (4)
#define STEREO_NUM_SLICES (2)

// Macros for the default configs of the disp2depth sipp SW filter
#define STEREO_MV250_BASELINE_M (0.035)
#define STEREO_TG161B_HFOV_DEG (69.0)

// Macros for the default configs of the stereo plugin
#define WARP_NR_POOL_BUFFS (4)
#define DESC_NR_POOL_BUFFS (2)
#define BYPASS_STAGE (true)

/// @brief The struct types from this namespace was ported from the StereoSipp VPU class
namespace StereoSipp {

/// @brief Expose the possible values of the LA_MODE field from the CV_STEREO_CFG stereo HW register
enum class LAMode {
    AVG3x3 = 0,   // 3x3 Kernel Average
    CLAMP3x3 = 1, // 3x3 Raw Sum Clamped to U6
    PASS3x3 = 2,  // 3x3 Center pass-through (LA Bypass)
};

/// @brief Expose the possible values of the DSP_OUT_MODE field from the CV_STEREO_CFG stereo HW register
enum class OutputMode {
    OTYPE_U16 = 0,  // Fixed point 11 bit mantissa 5 bit exponent
    OTYPE_FP16 = 1, // Half precision floating point
};

/// @brief Expose the possible values of the DSP_SIZE field from the CV_STEREO_CFG stereo HW register
enum class DispRange {
    DISP_64 = 0, // 64 aggregated costs per pixel
    DISP_96 = 1, // 96 aggregated costs per pixel
};

/// @brief Expose the possible values of the CME field from the CV_STEREO_CFG stereo HW register
enum class CMEMode {
    CME_DISABLED = 0, // Companding disabled
    CME_ENABLED = 1,  // Companding enabled
};

/// @brief Expose the possible values of the LRC_EN field from the CV_STEREO_CFG stereo HW register
enum class LRCEn {
    LRC_DISABLED = 0, // LR Check disabled
    LRC_ENABLED = 1,  // LR Check enabled
};

/// @brief Expose the possible values of the INVALID_REP field from the CV_STEREO_PARAMS stereo HW register
enum class InvRep {
    INV_REP_DISABLED = 0, // Disabled replace of invalid disparity with computed spatial kernel average
    INV_REP_ENABLED = 1,  // Enabled replace of invalid disparity with computed spatial kernel average
};

/// @brief Structure of the main HW stereo configuration.
struct StereoCfg {
    OutputMode dspOutMode{StereoSipp::OutputMode::OTYPE_U16};
    LAMode localAggrMode{StereoSipp::LAMode::AVG3x3};
    CMEMode cme{StereoSipp::CMEMode::CME_DISABLED};
    LRCEn lrcEn{StereoSipp::LRCEn::LRC_DISABLED};
    DispRange dspSz{StereoSipp::DispRange::DISP_96};
};

/// @brief Structure of the generic HW stereo parameters.
struct StereoPrm {
    uint8_t ratioThr{RATIO_THR};                             // SGBM WTA RATIO invalidation threshold
    uint8_t divFactor{DIV_FACTOR};                           // SGBM Aggregation division factor
    InvRep invalidRep{StereoSipp::InvRep::INV_REP_DISABLED}; // Enable replace of invalid disparity with
                                                             // computed spatial kernel average
    uint8_t invalidThr{INVALID_THR};                         // LRC Conditional Disparity Invalidity Threshold value
    uint8_t invalidDsp{INVALID_DSP};                         // Value to define the invalid disparity (defaults to 0)
};

/// @brief Structure of the HW stereo output block.
struct StereoPrm2 {
    uint8_t outRemoveThr{OUT_REMOVE_THR}; // Outliers Remove Threshold
    uint8_t outCensusThr{OUT_CENSUS_THR}; // Census Threshold
    uint8_t outDiffThr{OUT_DIFF_THR};     // Kernel radius difference threshold
};

/// @brief Structure for Disp to depth configuration
struct StereoDispToDepth {
    uint8_t enabled{0};
    float baseline{STEREO_MV250_BASELINE_M};
    float fov{STEREO_TG161B_HFOV_DEG};
};

/// @brief Structure of the penalties LUT access
struct LutItem {
    uint8_t horzP1;
    uint8_t horzP2;
    uint8_t vertP1;
    uint8_t vertP2;
};

/// @brief Structure of the address to the penalties LUT access structure
struct LutConfig {
    uint32_t lut{}; // address
};

/// @brief Structure of the stereo resources used.
struct StereoResources {
    uint32_t numSlices{STEREO_NUM_SLICES};
};

/// @brief Structure of the input/output frame spec.
struct StereoFrameSpec {
    uint32_t frameWidth{};  // [MANDATORY] TO BE SET AT THE USER LEVEL
    uint32_t frameHeight{}; // [MANDATORY] TO BE SET AT THE USER LEVEL
};

/// @brief Structure of the stereo component descriptor.
struct StereoDesc {
    StereoCfg stereoCfg;
    StereoPrm stereoPrm;
    StereoPrm2 stereoPrm2;
    LutConfig lutCfg;
    StereoDispToDepth dispToDepth;
};
} // namespace StereoSipp

/// @brief The struct types from this namespace was ported from the WarpSipp VPU class.
namespace WarpSipp {
/// @brief Expose the possible configs for Warp filter mode.
enum class FilterMode {
    BILINEAR = 0x0, // Set the filter to be in Bilinear mode
    BICUBIC = 0x1,  // Set the filter to be in Bicubic mode
    BYPASS = 0x2,   // Set the filter to be in Bypass mode
};
} // namespace WarpSipp

/// @brief This config is used by the plgVisionStereo flic plugin on the warp stage
struct CfgWarpStage {
    uint32_t inWidth{};    // [MANDATORY] TO BE SET AT THE USER LEVEL
    uint32_t inHeight{};   // [MANDATORY] TO BE SET AT THE USER LEVEL
    uint32_t meshLeft{};   // left mesh address - [MANDATORY] TO BE SET AT THE USER LEVEL
    uint32_t meshRight{};  // right mesh address - [MANDATORY] TO BE SET AT THE USER LEVEL
    uint16_t meshWidth{};  // [MANDATORY] TO BE SET AT THE USER LEVEL
    uint16_t meshHeight{}; // [MANDATORY] TO BE SET AT THE USER LEVEL
    uint16_t startX{0x0};  // X co-ord of start location within the output image
    uint16_t startY{0x0};  // Y co-ord of start location within the output image
    uint8_t filterMode{static_cast<uint8_t>(WarpSipp::FilterMode::BICUBIC)};
    uint32_t nrPoolBuffs{WARP_NR_POOL_BUFFS};
    bool bypassStage{BYPASS_STAGE};
};

/// @brief This config is used by the plgVisionStereo flic plugin on the NN-descriptor stage
struct CfgNNStage {
    uint32_t graphBuf{}; // [MANDATORY] TO BE SET AT THE USER LEVEL
    uint32_t graphLen{}; // [MANDATORY] TO BE SET AT THE USER LEVEL
    uint32_t nrPoolBuffs{DESC_NR_POOL_BUFFS};
};

/// @brief This config is used by the plgVisionStereo flic plugin on the stereo stage
struct CfgStereoStage {
    StereoSipp::StereoResources stereoResources;
    StereoSipp::StereoFrameSpec stereoFrameSpec;
    StereoSipp::StereoDesc stereoDesc;
};

/// @brief This struct contain the address and the size informations of the memory region used for the plgVisionStereo
/// plugin internal pools
struct CfgScratchBuffer {
    uint32_t base{}; // [MANDATORY] TO BE SET AT THE USER LEVEL
    uint32_t size{}; // [MANDATORY] TO BE SET AT THE USER LEVEL
};

#endif // PLG_VISIONSTEREO_TYPES_H
