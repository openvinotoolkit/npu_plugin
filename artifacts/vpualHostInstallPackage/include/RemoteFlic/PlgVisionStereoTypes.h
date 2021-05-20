/*
 * {% copyright %}
 */
#ifndef __PLG_VISIONSTEREO_TYPES_H__
#define __PLG_VISIONSTEREO_TYPES_H__
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

// Macros for the default configs of the warp component
#define WARP_NUM_SLICES (0X2)
#define MESH_WIDTH (0X20)
#define MESH_HEIGHT (0X14)
#define MESH_TYPE (0X0)
#define MESH_FORMAT (0X3)
#define MESH_RELATIVE (0X0)
#define MESH_DECPOSN (0X0)
#define MESH_BYPASS (0X0)
#define BYPASS_TRANSFORM (0X1)
#define EDGE_MODE (0X1)
#define EDGE_COLOUR (0X0)
#define STARTX (0X0)
#define STARTY (0X0)
#define MAX_MESH_YPOSITIVE (0X16)
#define MAX_MESH_YNEGATIVE (0X16)
#define FILTER_MODE (0X1)
#define PFBC_REQ_MODE (0X1)
#define TILEX_PREF_LOG2 (0X0)
#define HFLIP (0X0)
#define IN_PX_WIDTH (0X8)
#define OUT_PX_WIDTH (0X8)

// Macros for the default configs of the stereo plugin
#define WARP_NR_POOL_BUFFS (4)
#define DESC_NR_POOL_BUFFS (2)
#define BYPASS_STAGE (true)

/// @brief The struct types from this namespace was ported from the StereoSipp VPU class
namespace StereoSipp {

/// @brief Expose the possible values of the PXL_SIZE field from the CV_STEREO_CFG stereo HW register
enum class StereoInputType {
    INTYPE_U8 = 0,  // 8 bpp
    INTYPE_U10 = 1, // 10 bpp
};

/// @brief Expose the possible values of the IN_MODE field from the CV_STEREO_CFG stereo HW register
enum class StereoInputMode {
    INMODE_DCS = 0, // Image descriptors
    INMODE_PCC = 1, // Pre-computed costs
};

/// @brief Expose the possible values of the LA_MODE field from the CV_STEREO_CFG stereo HW register
enum class StereoModeLA {
    AVG3x3 = 0,   // 3x3 Kernel Average
    CLAMP3x3 = 1, // 3x3 Raw Sum Clamped to U6
    PASS3x3 = 2,  // 3x3 Center pass-through (LA Bypass)
};

/// @brief Expose the possible values of the DSP_OUT_MODE field from the CV_STEREO_CFG stereo HW register
enum class StereoModeOutput {
    OTYPE_U16 = 0,  // Fixed point 11 bit mantissa 5 bit exponent
    OTYPE_FP16 = 1, // Half precision floating point
};

/// @brief Expose the possible values of the DSP_SIZE field from the CV_STEREO_CFG stereo HW register
enum class StereoDisp {
    DISP_64 = 0, // 64 aggregated costs per pixel
    DISP_96 = 1, // 96 aggregated costs per pixel
};

/// @brief Expose the possible values of the SEARCH_DIR field from the CV_STEREO_CFG stereo HW register
enum class StereoDirection {
    DIR_L2R = 0, // Left to right image search direction
    DIR_R2L = 1, // Right to left image search direction
};

/// @brief Expose the possible values of the AXI_MODE field from the CV_STEREO_CFG stereo HW register
enum class StereoModeAXI {
    AXI_DISABLED = 0, // AXI access disabled
    AXI_WO = 1,       // AXI Write only
    AXI_RO = 2,       // AXI Read only
    AXI_RW = 3,       // AXI Read and write
};

/// @brief Expose the possible values of the CME field from the CV_STEREO_CFG stereo HW register
enum class StereoModeCME {
    CME_DISABLED = 0, // Companding disabled
    CME_ENABLED = 1,  // Companding enabled
};

/// @brief Expose the possible values of the WTA_DUMP_EN field from the CV_STEREO_CFG stereo HW register
enum class StereoEnDumpWTA {
    WTA_DISABLED = 0, // WTA (SGBM) dump disabled
    WTA_ENABLED = 1,  // WTA (SGBM) dump enabled
};

/// @brief Expose the possible values of the DUMP_MODE field from the CV_STEREO_CFG stereo HW register
enum class StereoModeDump {
    DUMP_LA = 0,   // Dump mode LA cost volume
    DUMP_SGBM = 1, // Dump mode SGBM cost volume
};

/// @brief Expose the possible values of the DUMP_EN field from the CV_STEREO_CFG stereo HW register
enum class StereoEnDump {
    CM_DUMP_DISABLED = 0, // Cost map dump disabled
    CM_DUMP_ENABLED = 1,  // Cost map dump enabled
};

/// @brief Expose the possible values of the LRC_EN field from the CV_STEREO_CFG stereo HW register
enum class StereoEnLRC {
    LRC_DISABLED = 0, // LR Check disabled
    LRC_ENABLED = 1,  // LR Check enabled
};

/// @brief Expose the possible values of the DATA_FLOW field from the CV_STEREO_CFG stereo HW register
enum class StereoDataFlow {
    DF_TOP = 0,  // Stereo data flow control TOP(LA+SGBM+WTA)
    DF_FULL = 1, // Stereo data flow control FULL(LA+SUB+SGBM+OUTPUT)
};

/// @brief Expose the possible values of the INVALID_REP field from the CV_STEREO_PARAMS stereo HW register
enum class StereoInvRep {
    INV_REP_DISABLED = 0, // Disabled replace of invalid disparity with computed spatial kernel average
    INV_REP_ENABLED = 1,  // Enabled replace of invalid disparity with computed spatial kernel average
};

/// @brief Structure of the main HW stereo configuration.
struct StereoCfg {
    StereoModeAXI axiMode{StereoSipp::StereoModeAXI::AXI_DISABLED};
    StereoModeOutput dspOutMode{StereoSipp::StereoModeOutput::OTYPE_U16};
    StereoModeLA localAggrMode{StereoSipp::StereoModeLA::AVG3x3};
    StereoModeCME cme{StereoSipp::StereoModeCME::CME_DISABLED};
    StereoEnDumpWTA wtaDumpEn{StereoSipp::StereoEnDumpWTA::WTA_DISABLED};
    StereoModeDump costDumpMode{StereoSipp::StereoModeDump::DUMP_LA};
    StereoEnDump costDumpEn{StereoSipp::StereoEnDump::CM_DUMP_DISABLED};
    StereoEnLRC lrcEn{StereoSipp::StereoEnLRC::LRC_DISABLED};
    StereoDataFlow dataFlow{StereoSipp::StereoDataFlow::DF_FULL};
    StereoDirection searchDir{StereoSipp::StereoDirection::DIR_L2R};
    StereoDisp dspSz{StereoSipp::StereoDisp::DISP_96};
    StereoInputType inputType{StereoSipp::StereoInputType::INTYPE_U8};
    StereoInputMode inputMode{StereoSipp::StereoInputMode::INMODE_DCS};
};

/// @brief Structure of the generic HW stereo parameters.
struct StereoPrm {
    uint8_t ratioThr{RATIO_THR};                                         // SGBM WTA RATIO invalidation threshold
    uint8_t divFactor{DIV_FACTOR};                                       // SGBM Aggregation division factor
    StereoInvRep invalidRep{StereoSipp::StereoInvRep::INV_REP_DISABLED}; // Enable replace of invalid disparity with
                                                                         // computed spatial kernel average
    uint8_t invalidThr{INVALID_THR}; // LRC Conditional Disparity Invalidity Threshold value
    uint8_t invalidDsp{INVALID_DSP}; // Value to define the invalid disparity (defaults to 0)
};

/// @brief Structure of the HW stereo output block.
struct StereoPrm2 {
    uint8_t outRemoveThr{OUT_REMOVE_THR}; // Outliers Remove Threshold
    uint8_t outCensusThr{OUT_CENSUS_THR}; // Census Threshold
    uint8_t outDiffThr{OUT_DIFF_THR};     // Kernel radius difference threshold
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
};
} // namespace StereoSipp

/// @brief The struct types from this namespace was ported from the WarpSipp VPU class
namespace WarpSipp {

/// @brief This struct is used to the configuration of the warp sipp pipe
struct SippPipeCfg {
    uint32_t inWidth{};   // [MANDATORY] TO BE SET AT THE USER LEVEL
    uint32_t inHeight{};  // [MANDATORY] TO BE SET AT THE USER LEVEL
    uint32_t outWidth{};  // [MANDATORY] TO BE SET AT THE USER LEVEL
    uint32_t outHeight{}; // [MANDATORY] TO BE SET AT THE USER LEVEL
    uint8_t engineId{};
    uint8_t numSlices{WARP_NUM_SLICES};
};

/// @brief This struct is used to the configuration of a warp mesh
struct MeshCfg {
    uint16_t meshWidth{MESH_WIDTH};      // Width of the mesh
    uint16_t meshHeight{MESH_HEIGHT};    // Height of the mesh
    uint8_t meshType{MESH_TYPE};         // 0 : Sparse mesh, 1: Pre-expanded mesh
    uint8_t meshFormat{MESH_FORMAT};     // Meshpoint format for the sparse mesh: 2 - Mixed point 16 bit; 3 - FP32
    uint8_t meshRelative{MESH_RELATIVE}; // 1 : MPs are relative to the output pixel location, 0 relative to (0,0)
                                         // origin (when meshType = sparse)
    uint8_t meshDecPosn{MESH_DECPOSN}; // The decimal position of the mixed point mesh points, counting from the LSB - 0
                                       // means fully integer
    uint8_t meshBypass{MESH_BYPASS}; // Set to generate a bypass mesh (NOTE: Must have PREEXPANDED_MODE = 1, RELATIVE =
                                     // 1 and DEC_POSN = 4'b0)
    uint8_t bypassTransform{BYPASS_TRANSFORM};     // 0 : FALSE, 1 : TRUE
    uint8_t edgeMode{EDGE_MODE};                   // 0: Pixel replication, 1: Edge colour
    uint16_t edgeColour{EDGE_COLOUR};              // Value to use when edgeMode = 1
    uint16_t startX{STARTX};                       // X co-ord of start location within the output image
    uint16_t startY{STARTY};                       // Y co-ord of start location within the output image
    uint16_t maxMeshYPositive{MAX_MESH_YPOSITIVE}; // Maximum positive mesh point Y offset (used for DMA CB allocation)
    uint16_t maxMeshYNegative{MAX_MESH_YNEGATIVE}; // Maximum negative mesh point Y offset (used for DMA CB allocation)
};

/// @brief This struct is used to the configuration of the warp HW filter
struct WarpCfg {
    uint8_t filterMode{FILTER_MODE}; // 0 : bilinear, 1 : bicubic, 2 : Bypass
    uint8_t pfbcReqMode{
        PFBC_REQ_MODE}; // cache prefetch mode to be used by the warp filters (refer to warpCtx\pfbcReqMode)
    uint8_t tileXPrefLog2{TILEX_PREF_LOG2}; // application preferred tile X dimension - log base 2 (0 - leave value as
                                            // computed in SIPP FW, 3..7 - valid values)
    uint8_t hFlip{HFLIP};                   // horizontal flip so allow speculative cache fetching to the left
    uint16_t inPixWidth{IN_PX_WIDTH};       // The width of the input pixels in bits (set to 0 for fp16)
    uint16_t outPixWidth{OUT_PX_WIDTH};     // The width of the output pixels in bits (set to 0 for fp16)
};

/// @brief This config struct inglobate the configs for the warp sipp pipe, mesh and some HW registers fields
struct Cfg {
    SippPipeCfg sippCfg;
    MeshCfg meshCfg;
    WarpCfg warpCfg;
};
} // namespace WarpSipp

/// @brief This config is used by the plgVisionStereo flic plugin on the warp stage
struct CfgWarpStage {
    WarpSipp::Cfg component;
    uint32_t meshLeft{};  // left mesh address - [MANDATORY] TO BE SET AT THE USER LEVEL
    uint32_t meshRight{}; // right mesh address - [MANDATORY] TO BE SET AT THE USER LEVEL
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

#endif // __PLG_VISIONSTEREO_TYPES_H__
