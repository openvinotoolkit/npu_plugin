// {% copyright %}

#ifndef _MV_TENSOR_H_
#define _MV_TENSOR_H_

#include <mv_types.h>
#include <mvTensorConfig.h>
#include <sw_tensor_ref.h>
#include <mvSubspaces.h>
#include "include/software_generated.h"
#include <map>
#include <mvSubspaces8d.h>

/// version of mvtensor library
#define MVTENSOR_VERSION_MAJOR  5
#define MVTENSOR_VERSION_MINOR  0

using namespace subspace;

typedef enum {
    t_fp16 = NN_FP16, ///< Half precision floating point
    t_u8f = NN_U8,    ///< Unsigned byte
    t_int = NN_INT32, ///< Signed integer (4 byte)
    t_fp32 = NN_FP32, ///< Single precision floating point
    t_i8 = NN_I8,     ///< Signed byte
    t_i16 = NN_INT16,
} t_MvTensorDataType;

typedef t_D8StorageOrder t_MvTensorStorageOrder;

void mvTensorInit(unsigned int start_shave = 0, unsigned int number_of_shaves = MVTENSOR_MAX_SHAVES,
                  unsigned int number_of_cmxslices = MVTENSOR_MAX_CMX_SLICES,
                  unsigned int start_cnn = 0, unsigned int number_of_cnns = MVTENSOR_MAX_NCES,
                  bool needTurnOnShaves = true, bool isLowPower = false);

void mvTensorClose(unsigned int start_shave = 0, unsigned int number_of_shaves = MVTENSOR_MAX_SHAVES,
                   unsigned int start_cnn = 0, unsigned int number_of_cnns = MVTENSOR_MAX_NCES,
                   bool needTurnOffShaves = true, bool isLowPower = false);


#define L2C_LINE_SZ 64
#define MV_TENSOR_DMA_PRIORITY 25

/// Generic macro to embed a structure alignment into it's definition so the
/// optimal alignment is enforced for all instantiations and even for arrays.
#define ALIGNED_STRUCT_INNER(fields, align)                                                           \
    typedef struct                                                                      \
    {                                                                                   \
        union                                                                           \
        {                                                                               \
            struct                                                                      \
            {                                                                           \
                fields                                                                  \
            };                                                                          \
            u8 padding[align] __attribute__((aligned(align)));                          \
        };                                                                              \
    }

/// Enforce alignment to the Leon/Shave L2 cache line size which is the highest
/// alignment needed on MA2x5x for best performance and safe data sharing.
#define ALIGNED_STRUCT(fields) ALIGNED_STRUCT_INNER(fields, L2C_LINE_SZ)


/// Naming Convention: opt_stge_x_y_sX_sY_optName. Please add fields as needed (rather than reflecting this in the optname)

#define streamIdDeviceMonBufferSize 2000

/// MvTensor data storage order options

/// Filtering types

#define SUBOP_FAMILY_SIZE 30
typedef enum : int32_t
{
    postopFamily = 0 * SUBOP_FAMILY_SIZE,
    eltwiseFamily = 1 * SUBOP_FAMILY_SIZE,
} SubOpFamilyOffset;

const char *getEltwiseSubOpName(int subOp);
const char *getPostOpSubOpName(int subOp);

constexpr inline int32_t GENERATE_SUBOPCODE(SubOpFamilyOffset family, int32_t subOplocalCode) {
    return MVCNN::SoftwareLayerParams_MAX + family + 1 + subOplocalCode;
}

inline int32_t generatSubOpCode(SubOpFamilyOffset family, int32_t subOplocalCode) {
    return MVCNN::SoftwareLayerParams_MAX + 1 + family + subOplocalCode;
}

inline int32_t decodeLocalSubOpCode(SubOpFamilyOffset family, int32_t globalSubOpCode) {
    return globalSubOpCode - MVCNN::SoftwareLayerParams_MAX - 1 - family;
}

inline int32_t decodeLocalSubOpCode(int32_t globalSubOpCode) {
    int32_t ret = globalSubOpCode - MVCNN::SoftwareLayerParams_MAX - 1;
    return ret % SUBOP_FAMILY_SIZE;
}

inline int32_t decodeLocalFamily(int32_t globalSubOpCode) {
    int32_t ret = globalSubOpCode - MVCNN::SoftwareLayerParams_MAX - 1;
    return ret - ret % SUBOP_FAMILY_SIZE;
}

namespace eltwise{
typedef enum : int32_t
{
    sum = 0,
    prod = 1,
    max = 2,
    div = 3,
    min = 4,
    sqdiff = 5,
    compareeq = 6,
    comparene = 7,
    comparegt = 8,
    comparege = 9,
    comparelt = 10,
    comparele = 11,
    logicalnot = 12,
    logicaland = 13,
    logicalor = 14,
    logicalxor = 15,
    pow = 16,
    floormod = 17,
    select = 18,
} EltwiseSubOp;
}  // namespace eltwise

namespace postop{
typedef enum : int32_t
{
    bias = MVCNN::PostOpsNestedParams_BiasParams,
    scale = MVCNN::PostOpsNestedParams_ScaleParams,
    scaleshift = MVCNN::PostOpsNestedParams_ScaleShiftParams,
    clamp = MVCNN::PostOpsNestedParams_ClampParams,
    elu = MVCNN::PostOpsNestedParams_EluParams,
    power = MVCNN::PostOpsNestedParams_PowerParams,
    biasLeakyRelu = MVCNN::PostOpsNestedParams_BiasLeakyReluParams,
    biasRelu = MVCNN::PostOpsNestedParams_BiasReluParams,
    leakyRelu = MVCNN::PostOpsNestedParams_LeakyReluParams,
    relu = MVCNN::PostOpsNestedParams_ReluParams,
    prelu = MVCNN::PostOpsNestedParams_PReluParams,
    sigmoid = MVCNN::PostOpsNestedParams_SigmoidParams,
    tanh = MVCNN::PostOpsNestedParams_TanhParams,
    hswish = MVCNN::PostOpsNestedParams_HSwishParams,
    swish = MVCNN::PostOpsNestedParams_SwishParams,
    softplus = MVCNN::PostOpsNestedParams_SoftPlusParams,
    mish = MVCNN::PostOpsNestedParams_MishParams,
    floor = MVCNN::PostOpsNestedParams_FloorParams,
    round = MVCNN::PostOpsNestedParams_RoundParams,
    erf = MVCNN::PostOpsNestedParams_ErfParams,
    ceiling = MVCNN::PostOpsNestedParams_CeilingParams,
    gelu = MVCNN::PostOpsNestedParams_GeluParams,
    log = MVCNN::PostOpsNestedParams_LogParams,
    exp = MVCNN::PostOpsNestedParams_ExpParams,
} postopSubOp;
}  // namespace postop


typedef enum : int32_t
{
    kEmpty = -1,
    kConv = MVCNN::SoftwareLayerParams_ConvolutionParams,
    kPool = MVCNN::SoftwareLayerParams_PoolingParams,
    kSoftMax = MVCNN::SoftwareLayerParams_SoftmaxParams,
    kFC = MVCNN::SoftwareLayerParams_FullyConnectedParams,
    kNone0 = MVCNN::SoftwareLayerParams_NONE,
//    kRelu = 6,
//    kDepthConv = 8,
//    kPRelu = 10,
    kLRN = MVCNN::SoftwareLayerParams_NormParams,
    kSum = GENERATE_SUBOPCODE(eltwiseFamily, eltwise::sum),
    kProd = GENERATE_SUBOPCODE(eltwiseFamily, eltwise::prod),
    kMax = GENERATE_SUBOPCODE(eltwiseFamily, eltwise::max),
//    kScale = 15,
//    kRelayout = 16,
//    kInnerLRN = 18,
//    kCopy = 19,
//    kTanh = 21,
    kDeconvolution = MVCNN::SoftwareLayerParams_DeconvolutionParams,
//    kElu = 23,
//    kPower = 26,
    kTile = MVCNN::SoftwareLayerParams_TileParams,
    kRegionYolo = MVCNN::SoftwareLayerParams_RegionYOLOParams,
//    kReorgYolo = 30,
    kConvert = MVCNN::SoftwareLayerParams_ConvertParams,
    kPermute = MVCNN::SoftwareLayerParams_PermuteParams,
    kNormalize = MVCNN::SoftwareLayerParams_NormalizeParams,
    kDetectionOutput = MVCNN::SoftwareLayerParams_DetectionOutputParams,
//    kMyriadXHwOp = 38,
    kCTCDecoder = MVCNN::SoftwareLayerParams_CTCDecoderParams,
//    kLeakyRelu = 44,
//    kBiasRelu = 45,
//    kBiasLeakyRelu = 46,
//    kIm2ColConvolution = 49,
//    kHwFcRelayout = 56,
//    kClamp = 57,
//    kRefConvolution = 58,
//    kGlobalAvgPool = 59,
//    kGlobalMaxPool = 60,
    kGRN = MVCNN::SoftwareLayerParams_GRNParams,
    kMVN = MVCNN::SoftwareLayerParams_MVNParams,
    kDepthDeconv = MVCNN::SoftwareLayerParams_DeconvolutionParams,
    kProposal = MVCNN::SoftwareLayerParams_ProposalParams,
    kROIPooling = MVCNN::SoftwareLayerParams_ROIPoolingParams,
    kPSROIPooling = MVCNN::SoftwareLayerParams_PSROIPoolingParams,
    kInterp = MVCNN::SoftwareLayerParams_InterpParams,
//    kCustom = 68,
//    kMTCNN = 69,
    kLSTMCell = MVCNN::SoftwareLayerParams_LSTMCellParams,
//    kPad = 71,
    kResample = MVCNN::SoftwareLayerParams_ResampleParams,
//    kUpsampling = 73,
//    kArgMax = 74,
    kDiv = GENERATE_SUBOPCODE(eltwiseFamily, eltwise::div),
    kMin = GENERATE_SUBOPCODE(eltwiseFamily, eltwise::min),
    kSqdiff = GENERATE_SUBOPCODE(eltwiseFamily, eltwise::sqdiff),
    kCompareEQ = GENERATE_SUBOPCODE(eltwiseFamily, eltwise::compareeq),
    kCompareNE = GENERATE_SUBOPCODE(eltwiseFamily, eltwise::comparene),
    kCompareGT = GENERATE_SUBOPCODE(eltwiseFamily, eltwise::comparegt),
    kCompareGE = GENERATE_SUBOPCODE(eltwiseFamily, eltwise::comparege),
    kCompareLT = GENERATE_SUBOPCODE(eltwiseFamily, eltwise::comparelt),
    kCompareLE = GENERATE_SUBOPCODE(eltwiseFamily, eltwise::comparele),
    kLogicalNOT = GENERATE_SUBOPCODE(eltwiseFamily, eltwise::logicalnot),
    kLogicalAND = GENERATE_SUBOPCODE(eltwiseFamily, eltwise::logicaland),
    kLogicalOR = GENERATE_SUBOPCODE(eltwiseFamily, eltwise::logicalor),
    kLogicalXOR = GENERATE_SUBOPCODE(eltwiseFamily, eltwise::logicalxor),
    kPow = GENERATE_SUBOPCODE(eltwiseFamily, eltwise::pow),
    kFloorMod = GENERATE_SUBOPCODE(eltwiseFamily, eltwise::floormod),
    kSelect = GENERATE_SUBOPCODE(eltwiseFamily, eltwise::select),
//    kGEMM = 91,
//    kLog = 92,
//    kReduceAnd = 93,
//    kReverseSequence = 94,
    kGather = MVCNN::SoftwareLayerParams_GatherParams,
//    kExp = 101,
//    kFloor = 102,
//    kScatter = 103,
//    kTopK = 104,
//    kReduceMin = 105,
//    kExpDetectionOutput = 106, // ExperimentalDetectronDetectionOutput
    kNMS =  MVCNN::SoftwareLayerParams_NMSParams,
//    kROIFeatureExtractor = 108,
//    kSCRelu = 109,
//    kErf = 110,
//    kReduceMax = 112,
//    kReduceSum = 113,
//    kReduceMean = 114,
//    kCustomDMA = 115,
//    kConvND = 116,
//    kOneHot = 117,
//    kPoolND = 118,
//    kLoopStart = 119,
//    kLoopEnd = 120,
//    kExpPriorGridGenerator = 121,
//    SOFTWARE_OP_TYPE_COUNT_ = MVCNN::SoftwareLayerParams_MAX + 1,
    kBias = GENERATE_SUBOPCODE(postopFamily, postop::bias),
    kScale = GENERATE_SUBOPCODE(postopFamily, postop::scale),
    kScaleShift = GENERATE_SUBOPCODE(postopFamily, postop::scaleshift),
    kClamp = GENERATE_SUBOPCODE(postopFamily, postop::clamp),
    kElu = GENERATE_SUBOPCODE(postopFamily, postop::elu),
    kPower = GENERATE_SUBOPCODE(postopFamily, postop::power),
    kBiasLeakyRelu = GENERATE_SUBOPCODE(postopFamily, postop::biasLeakyRelu),
    kBiasRelu = GENERATE_SUBOPCODE(postopFamily, postop::biasRelu),
    kLeakyRelu = GENERATE_SUBOPCODE(postopFamily, postop::leakyRelu),
    kRelu = GENERATE_SUBOPCODE(postopFamily, postop::relu),
    kPRelu = GENERATE_SUBOPCODE(postopFamily, postop::prelu),
    kSigmoidPostop = GENERATE_SUBOPCODE(postopFamily, postop::sigmoid),
    kTanh = GENERATE_SUBOPCODE(postopFamily, postop::tanh),
    kHSwish = GENERATE_SUBOPCODE(postopFamily, postop::hswish),
    kSwish = GENERATE_SUBOPCODE(postopFamily, postop::swish),
    kMish = GENERATE_SUBOPCODE(postopFamily, postop::mish),
    kFloor = GENERATE_SUBOPCODE(postopFamily, postop::floor),
    kCeiling = GENERATE_SUBOPCODE(postopFamily, postop::ceiling),
    kRound = GENERATE_SUBOPCODE(postopFamily, postop::round),
    kErf = GENERATE_SUBOPCODE(postopFamily, postop::erf),
    kGelu = GENERATE_SUBOPCODE(postopFamily, postop::gelu),
    kLog = GENERATE_SUBOPCODE(postopFamily, postop::log),
    kExp = GENERATE_SUBOPCODE(postopFamily, postop::exp),
    kFakeQuantize = MVCNN::SoftwareLayerParams_FakeQuantizeParams,
    kSoftPlus = GENERATE_SUBOPCODE(postopFamily, postop::softplus),
    kPad = MVCNN::SoftwareLayerParams_PadParams,
    kInterpolate = MVCNN::SoftwareLayerParams_InterpolateParams,
    kCTCGreedyDecoderSeqLen = MVCNN::SoftwareLayerParams_CTCGreedyDecoderSeqLenParams,
    kSpaceToDepth = MVCNN::SoftwareLayerParams_SpaceToDepthParams,
    kDummy = MVCNN::SoftwareLayerParams_DummyParams,
    kGatherElements = MVCNN::SoftwareLayerParams_GatherElementsParams,
    kDepthToSpace = MVCNN::SoftwareLayerParams_DepthToSpaceParams,
    kReverseSequence = MVCNN::SoftwareLayerParams_ReversesequenceParams,
    kCustomOcl = MVCNN::SoftwareLayerParams_CustomLayerOclParams,
    kCustomCpp = MVCNN::SoftwareLayerParams_CustomLayerCppParams,
    kStridedSlice = MVCNN::SoftwareLayerParams_StridedSliceParams,
    kGatherND = MVCNN::SoftwareLayerParams_GatherNDParams,
    kSWConvolution = MVCNN::SoftwareLayerParams_SWConvolutionParams,
    kScatterElementsUpdate = MVCNN::SoftwareLayerParams_ScatterElementsUpdateParams,
    kScatterUpdate = MVCNN::SoftwareLayerParams_ScatterUpdateParams,
    kOutShapeOfReshape = MVCNN::SoftwareLayerParams_OutShapeOfReshapeParams,
    kUpsampling = MVCNN::SoftwareLayerParams_UpsamplingParams,
    kNonZero = MVCNN::SoftwareLayerParams_NonZeroParams,
    kBroadcast = MVCNN::SoftwareLayerParams_BroadcastParams,
    kInnerLRN = MVCNN::SoftwareLayerParams_InnerLRNParams,
    kExpDetectionOutput = MVCNN::SoftwareLayerParams_ExpDetectionOutputParams,
    kExpGenerateProposals = MVCNN::SoftwareLayerParams_ExpGenerateProposalsParams,
    kExpPriorGridGenerator = MVCNN::SoftwareLayerParams_ExpPriorGridGeneratorParams,
    kExpTopKROIs = MVCNN::SoftwareLayerParams_ExpTopKROIsParams,
}t_MvTensorOpType;

/// Myriad resources structure

struct t_MvTensorMyriadResources {
    int dataPartitionNo;          ///< Number of shave L2 cache data partition use by MvTensor
    int instrPartitionNo;         ///< Number of shave L2 cache instruction partition use by MvTensor
    int bypassPartitionNo;        ///< Number of shave L2 cache bypass partition use by MvTensor
    // MvTensorDmaDescriptor* dmaTransactions;
    int32_t firstShave;
    int32_t lastShave;
    unsigned int shaveNum;
    uint32_t dmaLinkAgent = 0;
#if defined(MA2480)
    const int *hwBlocks;
#endif
};

struct t_MvTensorDebugInfo
{
    double ms;                               ///< Duration of the mvTensor call (in ms)
    char * debugMsg;   ///< Debug messages of size MV_TENSOR_DBG_MSG_SIZE
};

const char* getOpName(t_MvTensorOpType op);

#endif // _MV_TENSOR_H_
