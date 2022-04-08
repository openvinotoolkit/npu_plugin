// {% copyright %}

#ifndef _MV_TENSOR_H_
#define _MV_TENSOR_H_

#include <mv_types.h>
#include <mvTensorConfig.h>
#include <sw_tensor_ref.h>
#include <mvSubspaces.h>
#include <map>
#include <mvSubspaces8d.h>

/// version of mvtensor library
#define MVTENSOR_VERSION_MAJOR  5
#define MVTENSOR_VERSION_MINOR  0

using namespace subspace;

// the following 'flatbuffers' and 'MVCNN' definitions are copied 
// just to use codes of (sub)operations in testing system as previously

namespace flatbuffers {

// Check 'v' is out of closed range [low; high].
// Workaround for GCC warning [-Werror=type-limits]:
// comparison is always true due to limited range of data type.
template<typename T>
inline bool IsOutRange(const T &v, const T &low, const T &high) {
  return (v < low) || (high < v);
}

}  // namespace flatbuffers

namespace MVCNN {

enum DataType {
  DataType_UNKNOWN = 0,
  DataType_INT1 = 1,
  DataType_INT8 = 2,
  DataType_INT16 = 3,
  DataType_INT32 = 4,
  DataType_INT64 = 5,
  DataType_UINT8 = 6,
  DataType_UINT16 = 7,
  DataType_UINT32 = 8,
  DataType_UINT64 = 9,
  DataType_FLOAT16 = 10,
  DataType_FLOAT32 = 11,
  DataType_FLOAT64 = 12,
  DataType_MIN = DataType_UNKNOWN,
  DataType_MAX = DataType_FLOAT64
};

inline const DataType (&EnumValuesDataType())[13] {
  static const DataType values[] = {
    DataType_UNKNOWN,
    DataType_INT1,
    DataType_INT8,
    DataType_INT16,
    DataType_INT32,
    DataType_INT64,
    DataType_UINT8,
    DataType_UINT16,
    DataType_UINT32,
    DataType_UINT64,
    DataType_FLOAT16,
    DataType_FLOAT32,
    DataType_FLOAT64
  };
  return values;
}

inline const char * const *EnumNamesDataType() {
  static const char * const names[14] = {
    "UNKNOWN",
    "INT1",
    "INT8",
    "INT16",
    "INT32",
    "INT64",
    "UINT8",
    "UINT16",
    "UINT32",
    "UINT64",
    "FLOAT16",
    "FLOAT32",
    "FLOAT64",
    nullptr
  };
  return names;
}

inline const char *EnumNameDataType(DataType e) {
  if (flatbuffers::IsOutRange(e, DataType_UNKNOWN, DataType_FLOAT64)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesDataType()[index];
}

enum EltwiseParam {
  EltwiseParam_NONE = 0,
  EltwiseParam_i32 = 1,
  EltwiseParam_fp = 2,
  EltwiseParam_MIN = EltwiseParam_NONE,
  EltwiseParam_MAX = EltwiseParam_fp
};

inline const EltwiseParam (&EnumValuesEltwiseParam())[3] {
  static const EltwiseParam values[] = {
    EltwiseParam_NONE,
    EltwiseParam_i32,
    EltwiseParam_fp
  };
  return values;
}

inline const char * const *EnumNamesEltwiseParam() {
  static const char * const names[4] = {
    "NONE",
    "i32",
    "fp",
    nullptr
  };
  return names;
}

inline const char *EnumNameEltwiseParam(EltwiseParam e) {
  if (flatbuffers::IsOutRange(e, EltwiseParam_NONE, EltwiseParam_fp)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesEltwiseParam()[index];
}

enum EltwisePostOpsNestedParams {
  EltwisePostOpsNestedParams_NONE = 0,
  EltwisePostOpsNestedParams_EltwisePostOpEmpty = 1,
  EltwisePostOpsNestedParams_EltwisePostOpPReLU = 2,
  EltwisePostOpsNestedParams_MIN = EltwisePostOpsNestedParams_NONE,
  EltwisePostOpsNestedParams_MAX = EltwisePostOpsNestedParams_EltwisePostOpPReLU
};

inline const EltwisePostOpsNestedParams (&EnumValuesEltwisePostOpsNestedParams())[3] {
  static const EltwisePostOpsNestedParams values[] = {
    EltwisePostOpsNestedParams_NONE,
    EltwisePostOpsNestedParams_EltwisePostOpEmpty,
    EltwisePostOpsNestedParams_EltwisePostOpPReLU
  };
  return values;
}

inline const char * const *EnumNamesEltwisePostOpsNestedParams() {
  static const char * const names[4] = {
    "NONE",
    "EltwisePostOpEmpty",
    "EltwisePostOpPReLU",
    nullptr
  };
  return names;
}

inline const char *EnumNameEltwisePostOpsNestedParams(EltwisePostOpsNestedParams e) {
  if (flatbuffers::IsOutRange(e, EltwisePostOpsNestedParams_NONE, EltwisePostOpsNestedParams_EltwisePostOpPReLU)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesEltwisePostOpsNestedParams()[index];
}

enum PostOpsNestedParams {
  PostOpsNestedParams_NONE = 0,
  PostOpsNestedParams_BiasParams = 1,
  PostOpsNestedParams_ScaleParams = 2,
  PostOpsNestedParams_ScaleShiftParams = 3,
  PostOpsNestedParams_ClampParams = 4,
  PostOpsNestedParams_EluParams = 5,
  PostOpsNestedParams_PowerParams = 6,
  PostOpsNestedParams_BiasLeakyReluParams = 7,
  PostOpsNestedParams_BiasReluParams = 8,
  PostOpsNestedParams_LeakyReluParams = 9,
  PostOpsNestedParams_ReluParams = 10,
  PostOpsNestedParams_PReluParams = 11,
  PostOpsNestedParams_SigmoidParams = 12,
  PostOpsNestedParams_TanhParams = 13,
  PostOpsNestedParams_HSwishParams = 14,
  PostOpsNestedParams_SwishParams = 15,
  PostOpsNestedParams_MishParams = 16,
  PostOpsNestedParams_SoftPlusParams = 17,
  PostOpsNestedParams_FloorParams = 18,
  PostOpsNestedParams_ErfParams = 19,
  PostOpsNestedParams_RoundParams = 20,
  PostOpsNestedParams_CeilingParams = 21,
  PostOpsNestedParams_GeluParams = 22,
  PostOpsNestedParams_LogParams = 23,
  PostOpsNestedParams_ExpParams = 24,
  PostOpsNestedParams_SinhParams = 25,
  PostOpsNestedParams_CoshParams = 26,
  PostOpsNestedParams_SqrtParams = 27,
  PostOpsNestedParams_MIN = PostOpsNestedParams_NONE,
  PostOpsNestedParams_MAX = PostOpsNestedParams_SqrtParams
};

inline const PostOpsNestedParams (&EnumValuesPostOpsNestedParams())[28] {
  static const PostOpsNestedParams values[] = {
    PostOpsNestedParams_NONE,
    PostOpsNestedParams_BiasParams,
    PostOpsNestedParams_ScaleParams,
    PostOpsNestedParams_ScaleShiftParams,
    PostOpsNestedParams_ClampParams,
    PostOpsNestedParams_EluParams,
    PostOpsNestedParams_PowerParams,
    PostOpsNestedParams_BiasLeakyReluParams,
    PostOpsNestedParams_BiasReluParams,
    PostOpsNestedParams_LeakyReluParams,
    PostOpsNestedParams_ReluParams,
    PostOpsNestedParams_PReluParams,
    PostOpsNestedParams_SigmoidParams,
    PostOpsNestedParams_TanhParams,
    PostOpsNestedParams_HSwishParams,
    PostOpsNestedParams_SwishParams,
    PostOpsNestedParams_MishParams,
    PostOpsNestedParams_SoftPlusParams,
    PostOpsNestedParams_FloorParams,
    PostOpsNestedParams_ErfParams,
    PostOpsNestedParams_RoundParams,
    PostOpsNestedParams_CeilingParams,
    PostOpsNestedParams_GeluParams,
    PostOpsNestedParams_LogParams,
    PostOpsNestedParams_ExpParams,
    PostOpsNestedParams_SinhParams,
    PostOpsNestedParams_CoshParams,
    PostOpsNestedParams_SqrtParams
  };
  return values;
}

inline const char * const *EnumNamesPostOpsNestedParams() {
  static const char * const names[29] = {
    "NONE",
    "BiasParams",
    "ScaleParams",
    "ScaleShiftParams",
    "ClampParams",
    "EluParams",
    "PowerParams",
    "BiasLeakyReluParams",
    "BiasReluParams",
    "LeakyReluParams",
    "ReluParams",
    "PReluParams",
    "SigmoidParams",
    "TanhParams",
    "HSwishParams",
    "SwishParams",
    "MishParams",
    "SoftPlusParams",
    "FloorParams",
    "ErfParams",
    "RoundParams",
    "CeilingParams",
    "GeluParams",
    "LogParams",
    "ExpParams",
    "SinhParams",
    "CoshParams",
    "SqrtParams",
    nullptr
  };
  return names;
}

inline const char *EnumNamePostOpsNestedParams(PostOpsNestedParams e) {
  if (flatbuffers::IsOutRange(e, PostOpsNestedParams_NONE, PostOpsNestedParams_SqrtParams)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesPostOpsNestedParams()[index];
}

enum SoftwareLayerParams {
  SoftwareLayerParams_NONE = 0,
  SoftwareLayerParams_DummyParams = 1,
  SoftwareLayerParams_DetectionOutputParams = 2,
  SoftwareLayerParams_FlattenParams = 3,
  SoftwareLayerParams_InterpParams = 4,
  SoftwareLayerParams_NormalizeParams = 5,
  SoftwareLayerParams_PermuteParams = 6,
  SoftwareLayerParams_PriorboxParams = 7,
  SoftwareLayerParams_ProposalParams = 8,
  SoftwareLayerParams_RegionYOLOParams = 9,
  SoftwareLayerParams_ReorgYOLOParams = 10,
  SoftwareLayerParams_ReshapeParams = 11,
  SoftwareLayerParams_SoftmaxParams = 12,
  SoftwareLayerParams_CustomLayerOclParams = 13,
  SoftwareLayerParams_PassthroughParams = 14,
  SoftwareLayerParams_LayerRecordParams = 15,
  SoftwareLayerParams_ROIPoolingParams = 16,
  SoftwareLayerParams_QuantizeParams = 17,
  SoftwareLayerParams_ArgMaxParams = 18,
  SoftwareLayerParams_NormParams = 19,
  SoftwareLayerParams_EltwiseParams = 20,
  SoftwareLayerParams_ResampleParams = 21,
  SoftwareLayerParams_CorrelationParams = 22,
  SoftwareLayerParams_MVNParams = 23,
  SoftwareLayerParams_GRNParams = 24,
  SoftwareLayerParams_CTCDecoderParams = 25,
  SoftwareLayerParams_SpatialTransformParams = 26,
  SoftwareLayerParams_FakeQuantizeParams = 27,
  SoftwareLayerParams_PoolingParams = 28,
  SoftwareLayerParams_EdslParams = 29,
  SoftwareLayerParams_TileParams = 30,
  SoftwareLayerParams_PSROIPoolingParams = 31,
  SoftwareLayerParams_DeconvolutionParams = 32,
  SoftwareLayerParams_UnaryOpParams = 33,
  SoftwareLayerParams_ConvolutionParams = 34,
  SoftwareLayerParams_GatherParams = 35,
  SoftwareLayerParams_PostOpsParams = 36,
  SoftwareLayerParams_NegativeParams = 37,
  SoftwareLayerParams_ConvertParams = 38,
  SoftwareLayerParams_CustomLayerCppParams = 39,
  SoftwareLayerParams_PermuteNDParams = 40,
  SoftwareLayerParams_PadParams = 41,
  SoftwareLayerParams_InterpolateParams = 42,
  SoftwareLayerParams_CTCGreedyDecoderSeqLenParams = 43,
  SoftwareLayerParams_SpaceToDepthParams = 44,
  SoftwareLayerParams_DepthToSpaceParams = 45,
  SoftwareLayerParams_GatherElementsParams = 46,
  SoftwareLayerParams_ReversesequenceParams = 47,
  SoftwareLayerParams_LSTMCellParams = 48,
  SoftwareLayerParams_StridedSliceParams = 49,
  SoftwareLayerParams_FullyConnectedParams = 50,
  SoftwareLayerParams_SWConvolutionParams = 51,
  SoftwareLayerParams_TopKParams = 52,
  SoftwareLayerParams_ScatterElementsUpdateParams = 53,
  SoftwareLayerParams_ScatterUpdateParams = 54,
  SoftwareLayerParams_GatherNDParams = 55,
  SoftwareLayerParams_DesparsifyParams = 56,
  SoftwareLayerParams_OutShapeOfReshapeParams = 57,
  SoftwareLayerParams_UpsamplingParams = 58,
  SoftwareLayerParams_BroadcastParams = 59,
  SoftwareLayerParams_NonZeroParams = 60,
  SoftwareLayerParams_InnerLRNParams = 61,
  SoftwareLayerParams_ExpDetectionOutputParams = 62,
  SoftwareLayerParams_NMSParams = 63,
  SoftwareLayerParams_ExpGenerateProposalsParams = 64,
  SoftwareLayerParams_ExpPriorGridGeneratorParams = 65,
  SoftwareLayerParams_ExpTopKROIsParams = 66,
  SoftwareLayerParams_ROIAlignParams = 67,
  SoftwareLayerParams_ROIFeatureExtractorParams = 68,
  SoftwareLayerParams_ConcatParams = 69,
  SoftwareLayerParams_CopyParams = 70,
  SoftwareLayerParams_CropParams = 71,
  SoftwareLayerParams_SliceParams = 72,
  SoftwareLayerParams_AlignParams = 73,
  SoftwareLayerParams_ReduceParams = 74,
  SoftwareLayerParams_SCReluParams = 75,
  SoftwareLayerParams_MIN = SoftwareLayerParams_NONE,
  SoftwareLayerParams_MAX = SoftwareLayerParams_SCReluParams
};

inline const SoftwareLayerParams (&EnumValuesSoftwareLayerParams())[76] {
  static const SoftwareLayerParams values[] = {
    SoftwareLayerParams_NONE,
    SoftwareLayerParams_DummyParams,
    SoftwareLayerParams_DetectionOutputParams,
    SoftwareLayerParams_FlattenParams,
    SoftwareLayerParams_InterpParams,
    SoftwareLayerParams_NormalizeParams,
    SoftwareLayerParams_PermuteParams,
    SoftwareLayerParams_PriorboxParams,
    SoftwareLayerParams_ProposalParams,
    SoftwareLayerParams_RegionYOLOParams,
    SoftwareLayerParams_ReorgYOLOParams,
    SoftwareLayerParams_ReshapeParams,
    SoftwareLayerParams_SoftmaxParams,
    SoftwareLayerParams_CustomLayerOclParams,
    SoftwareLayerParams_PassthroughParams,
    SoftwareLayerParams_LayerRecordParams,
    SoftwareLayerParams_ROIPoolingParams,
    SoftwareLayerParams_QuantizeParams,
    SoftwareLayerParams_ArgMaxParams,
    SoftwareLayerParams_NormParams,
    SoftwareLayerParams_EltwiseParams,
    SoftwareLayerParams_ResampleParams,
    SoftwareLayerParams_CorrelationParams,
    SoftwareLayerParams_MVNParams,
    SoftwareLayerParams_GRNParams,
    SoftwareLayerParams_CTCDecoderParams,
    SoftwareLayerParams_SpatialTransformParams,
    SoftwareLayerParams_FakeQuantizeParams,
    SoftwareLayerParams_PoolingParams,
    SoftwareLayerParams_EdslParams,
    SoftwareLayerParams_TileParams,
    SoftwareLayerParams_PSROIPoolingParams,
    SoftwareLayerParams_DeconvolutionParams,
    SoftwareLayerParams_UnaryOpParams,
    SoftwareLayerParams_ConvolutionParams,
    SoftwareLayerParams_GatherParams,
    SoftwareLayerParams_PostOpsParams,
    SoftwareLayerParams_NegativeParams,
    SoftwareLayerParams_ConvertParams,
    SoftwareLayerParams_CustomLayerCppParams,
    SoftwareLayerParams_PermuteNDParams,
    SoftwareLayerParams_PadParams,
    SoftwareLayerParams_InterpolateParams,
    SoftwareLayerParams_CTCGreedyDecoderSeqLenParams,
    SoftwareLayerParams_SpaceToDepthParams,
    SoftwareLayerParams_DepthToSpaceParams,
    SoftwareLayerParams_GatherElementsParams,
    SoftwareLayerParams_ReversesequenceParams,
    SoftwareLayerParams_LSTMCellParams,
    SoftwareLayerParams_StridedSliceParams,
    SoftwareLayerParams_FullyConnectedParams,
    SoftwareLayerParams_SWConvolutionParams,
    SoftwareLayerParams_TopKParams,
    SoftwareLayerParams_ScatterElementsUpdateParams,
    SoftwareLayerParams_ScatterUpdateParams,
    SoftwareLayerParams_GatherNDParams,
    SoftwareLayerParams_DesparsifyParams,
    SoftwareLayerParams_OutShapeOfReshapeParams,
    SoftwareLayerParams_UpsamplingParams,
    SoftwareLayerParams_BroadcastParams,
    SoftwareLayerParams_NonZeroParams,
    SoftwareLayerParams_InnerLRNParams,
    SoftwareLayerParams_ExpDetectionOutputParams,
    SoftwareLayerParams_NMSParams,
    SoftwareLayerParams_ExpGenerateProposalsParams,
    SoftwareLayerParams_ExpPriorGridGeneratorParams,
    SoftwareLayerParams_ExpTopKROIsParams,
    SoftwareLayerParams_ROIAlignParams,
    SoftwareLayerParams_ROIFeatureExtractorParams,
    SoftwareLayerParams_ConcatParams,
    SoftwareLayerParams_CopyParams,
    SoftwareLayerParams_CropParams,
    SoftwareLayerParams_SliceParams,
    SoftwareLayerParams_AlignParams,
    SoftwareLayerParams_ReduceParams,
    SoftwareLayerParams_SCReluParams
  };
  return values;
}

inline const char * const *EnumNamesSoftwareLayerParams() {
  static const char * const names[77] = {
    "NONE",
    "DummyParams",
    "DetectionOutputParams",
    "FlattenParams",
    "InterpParams",
    "NormalizeParams",
    "PermuteParams",
    "PriorboxParams",
    "ProposalParams",
    "RegionYOLOParams",
    "ReorgYOLOParams",
    "ReshapeParams",
    "SoftmaxParams",
    "CustomLayerOclParams",
    "PassthroughParams",
    "LayerRecordParams",
    "ROIPoolingParams",
    "QuantizeParams",
    "ArgMaxParams",
    "NormParams",
    "EltwiseParams",
    "ResampleParams",
    "CorrelationParams",
    "MVNParams",
    "GRNParams",
    "CTCDecoderParams",
    "SpatialTransformParams",
    "FakeQuantizeParams",
    "PoolingParams",
    "EdslParams",
    "TileParams",
    "PSROIPoolingParams",
    "DeconvolutionParams",
    "UnaryOpParams",
    "ConvolutionParams",
    "GatherParams",
    "PostOpsParams",
    "NegativeParams",
    "ConvertParams",
    "CustomLayerCppParams",
    "PermuteNDParams",
    "PadParams",
    "InterpolateParams",
    "CTCGreedyDecoderSeqLenParams",
    "SpaceToDepthParams",
    "DepthToSpaceParams",
    "GatherElementsParams",
    "ReversesequenceParams",
    "LSTMCellParams",
    "StridedSliceParams",
    "FullyConnectedParams",
    "SWConvolutionParams",
    "TopKParams",
    "ScatterElementsUpdateParams",
    "ScatterUpdateParams",
    "GatherNDParams",
    "DesparsifyParams",
    "OutShapeOfReshapeParams",
    "UpsamplingParams",
    "BroadcastParams",
    "NonZeroParams",
    "InnerLRNParams",
    "ExpDetectionOutputParams",
    "NMSParams",
    "ExpGenerateProposalsParams",
    "ExpPriorGridGeneratorParams",
    "ExpTopKROIsParams",
    "ROIAlignParams",
    "ROIFeatureExtractorParams",
    "ConcatParams",
    "CopyParams",
    "CropParams",
    "SliceParams",
    "AlignParams",
    "ReduceParams",
    "SCReluParams",
    nullptr
  };
  return names;
}

inline const char *EnumNameSoftwareLayerParams(SoftwareLayerParams e) {
  if (flatbuffers::IsOutRange(e, SoftwareLayerParams_NONE, SoftwareLayerParams_SCReluParams)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesSoftwareLayerParams()[index];
}

}  // namespace MVCNN

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
//    kFC = MVCNN::SoftwareLayerParams_FullyConnectedParams,
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
//    kNMS =  MVCNN::SoftwareLayerParams_NMSParams,
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
//    kStridedSlice = MVCNN::SoftwareLayerParams_StridedSliceParams,
//    kGatherND = MVCNN::SoftwareLayerParams_GatherNDParams,
//    kSWConvolution = MVCNN::SoftwareLayerParams_SWConvolutionParams,
//    kScatterElementsUpdate = MVCNN::SoftwareLayerParams_ScatterElementsUpdateParams,
//    kScatterUpdate = MVCNN::SoftwareLayerParams_ScatterUpdateParams,
//    kOutShapeOfReshape = MVCNN::SoftwareLayerParams_OutShapeOfReshapeParams,
//    kUpsampling = MVCNN::SoftwareLayerParams_UpsamplingParams,
//    kNonZero = MVCNN::SoftwareLayerParams_NonZeroParams,
//    kBroadcast = MVCNN::SoftwareLayerParams_BroadcastParams,
//    kInnerLRN = MVCNN::SoftwareLayerParams_InnerLRNParams,
//    kExpDetectionOutput = MVCNN::SoftwareLayerParams_ExpDetectionOutputParams,
//    kExpGenerateProposals = MVCNN::SoftwareLayerParams_ExpGenerateProposalsParams,
//    kExpPriorGridGenerator = MVCNN::SoftwareLayerParams_ExpPriorGridGeneratorParams,
//    kExpTopKROIs = MVCNN::SoftwareLayerParams_ExpTopKROIsParams,
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
