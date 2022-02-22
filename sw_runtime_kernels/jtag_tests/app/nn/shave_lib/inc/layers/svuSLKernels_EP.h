/*
 * {% copyright %}
 */
#pragma once

#include <stdint.h>

// Need a double level of indirection to resolve KERNEL_SYM_PREFIX correctly
#define __KERNEL_SYM(prefix, sym) prefix##sym
#define _KERNEL_SYM(prefix, sym) __KERNEL_SYM(prefix, sym)
#define KERNEL_DATA(sym) (uint32_t) & _KERNEL_SYM(KERNEL_SYM_PREFIX, sym)
#define KERNEL_FUNC(sym) (nn::shave_lib::shaveKernelEntry) & _KERNEL_SYM(KERNEL_SYM_PREFIX, sym)

#ifdef NO_SVU_NN_CONTROLLER
PREAMBLE_FUNC(sym) & sym
#else
#define PREAMBLE_FUNC(sym) (nn::shave_lib::preamble) & _KERNEL_SYM(KERNEL_SYM_PREFIX, sym)
#endif

#define SVU_NN_KERNEL_ENTRY (void *)&_KERNEL_SYM(KERNEL_SYM_PREFIX, svuNNEntry)

#ifdef __cplusplus
    extern "C" {
#endif

extern uint8_t svuSLKernels_Base;

extern uint32_t SLK_svuNNEntry;

extern uint32_t SLK_sData;
extern uint32_t SLK_sParam;

//
// Layer kernels
//

extern uint32_t SLK_mvDeconvNxM_hwc;
extern uint32_t SLK_mvDeconvNxM_chw;
extern uint32_t SLK_mvDepthDeconv_hwc;
extern uint32_t SLK_mvDepthDeconv_chw;

extern uint32_t SLK_mvConv_ref;
extern uint32_t SLK_mvSWConvolution;

extern uint32_t SLK_mvSoftMax;
extern uint32_t SLK_fullyConnected;
extern uint32_t SLK_mvReverseSequence;
extern uint32_t SLK_nnPermute;
extern uint32_t SLK_nnQuantizer_EqualScales;
extern uint32_t SLK_nnQuantizer_PerChannelScales;
extern uint32_t SLK_nnQuantizer_EqualScales_StridedIn_CompactOut;
extern uint32_t SLK_nnResample;
extern uint32_t SLK_nnResampleAntialias;
extern uint32_t SLK_nnSpatialTransform;
extern uint32_t SLK_Passthrough;
extern uint32_t SLK_PriorboxKernel;
extern uint32_t SLK_ROIPoolingKernel;
extern uint32_t SLK_custom_ocl;
extern uint32_t SLK_custom_cpp;
extern uint32_t SLK_edsl;
extern uint32_t SLK_Dummy;
extern uint32_t SLK_proposal;
extern uint32_t SLK_regionYoloKernel_CHW;
extern uint32_t SLK_regionYoloKernel_HWC;
extern uint32_t SLK_DetectionOutput;
extern uint32_t SLK_normalize;
extern uint32_t SLK_argMaxKernel_WithNoAxis;
extern uint32_t SLK_argMaxKernel_WithAxis;
extern uint32_t SLK_argMaxKernel_WithAxis1_Top1;
extern uint32_t SLK_ReorgYoloHWC;
extern uint32_t SLK_ReorgYoloCHW;
extern uint32_t SLK_interpCHW;
extern uint32_t SLK_interpCHW_1x1;
extern uint32_t SLK_interpHWC;
extern uint32_t SLK_interpHWC_1x1;
extern uint32_t SLK_Norm_AcrossChannel;
extern uint32_t SLK_Norm_SameChannel;
extern uint32_t SLK_Eltwise;
extern uint32_t SLK_AvgPoolingKernel;
extern uint32_t SLK_CHW_mvMaxPoolMxN;
extern uint32_t SLK_HWC_mvMaxPoolMxN;
extern uint32_t SLK_correlation;
extern uint32_t SLK_CTCDecoder;
extern uint32_t SLK_GRN;
extern uint32_t SLK_MVN;
extern uint32_t SLK_fakeQuantize_CHW;
extern uint32_t SLK_fakeQuantize_HWC;
extern uint32_t SLK_nnEntry;
extern uint32_t SLK_PSROIPooling;
extern uint32_t SLK_Tile;
extern uint32_t SLK_negativeKernel;
extern uint32_t SLK_chw_postOps_3D_core;
extern uint32_t SLK_hwc_postOps_3D_core;
extern uint32_t SLK_hcw_postOps_3D_core;
extern uint32_t SLK_postOps_ND_core;
extern uint32_t SLK_mvConvert;
extern uint32_t SLK_PadKernel;
//extern uint32_t SLK_PadKernelSingle;
extern uint32_t SLK_mvGatherElements;
extern uint32_t SLK_CTCGreedyDecoderSeqLen;
extern uint32_t SLK_mvSpaceToDepth;
extern uint32_t SLK_mvDepthToSpace;
extern uint32_t SLK_LSTMCell;
extern uint32_t SLK_mvUpsamplingMEM;
extern uint32_t SLK_mvUpsamplingDMA;
extern uint32_t SLK_mvNonZeroCollect;
extern uint32_t SLK_mvInnerLRN;
extern uint32_t SLK_mvNonMaxSuppression;
extern uint32_t SLK_mvExpDetectionOutput;
extern uint32_t SLK_mvExpGenerateProposals;
extern uint32_t SLK_mvExpPriorGridGenerator;
extern uint32_t SLK_mvExpTopKROIs;

//
// Preambles
//

extern uint32_t SLK_depthDeconvCHWPreamble;
extern uint32_t SLK_depthDeconvHWCPreamble;
extern uint32_t SLK_deconvNxM_hwcPreamble;
extern uint32_t SLK_deconvNxM_chwPreamble;

extern uint32_t SLK_refConvPreamble;
extern uint32_t SLK_swConvolutionPreamble;

extern uint32_t SLK_preQuantizer;
extern uint32_t SLK_preQuantizer_EqualScales_StridedIn_CompactOut;
extern uint32_t SLK_dummyPreamble;
extern uint32_t SLK_cleanDummy;
extern uint32_t SLK_interpPreamble;
extern uint32_t SLK_fullyConnectedPreamble;
extern uint32_t SLK_preNormalize;
extern uint32_t SLK_prePriorbox;
extern uint32_t SLK_preProposal;
extern uint32_t SLK_cleanProposal;
extern uint32_t SLK_detOutPreamble;
extern uint32_t SLK_detOutCleanup;
extern uint32_t SLK_preEdsl;
extern uint32_t SLK_preResample;
extern uint32_t SLK_preSoftmax;
extern uint32_t SLK_singleShaveSoftmax;
extern uint32_t SLK_sigmoid_fp16;
extern uint32_t SLK_reorder_fp16;
extern uint32_t SLK_hswish_fp16;
extern uint32_t SLK_singleShaveMVN;
extern uint32_t SLK_elu_fp16;
extern uint32_t SLK_exp_fp16;
extern uint32_t SLK_tanh_fp16;
extern uint32_t SLK_power_fp16;
extern uint32_t SLK_preCorrelation;
extern uint32_t SLK_preROIPooling;
extern uint32_t SLK_preCTCDecoder;
extern uint32_t SLK_preReorgYolo;
extern uint32_t SLK_preRegionYolo;
extern uint32_t SLK_prePermute;
extern uint32_t SLK_prePermute1D;
extern uint32_t SLK_eltwisePreamble;
extern uint32_t SLK_cleanEltwise;
extern uint32_t SLK_argmaxPreamble;
extern uint32_t SLK_preAvgPooling;
extern uint32_t SLK_preCHW_maxPoolMxN;
extern uint32_t SLK_preHWC_maxPoolMxN;
extern uint32_t SLK_preSpatialTransform;
extern uint32_t SLK_preCustomLayerOcl;
extern uint32_t SLK_preCustomLayerCpp;
extern uint32_t SLK_execCleanupCustomLayerOcl;
extern uint32_t SLK_execCleanupCustomLayerCpp;
extern uint32_t SLK_preFakeQuantize;
extern uint32_t SLK_grnPreamble;
extern uint32_t SLK_mvnPreamble;
extern uint32_t SLK_preNorm;
extern uint32_t SLK_prePassthrough;
extern uint32_t SLK_preReShape;
extern uint32_t SLK_preGather;
extern uint32_t SLK_prePSROIPooling;
extern uint32_t SLK_preTile;
extern uint32_t SLK_negativePreamble;
extern uint32_t SLK_prePostOpsCHW;
extern uint32_t SLK_prePostOpsHWC;
extern uint32_t SLK_prePostOpsHCW;
extern uint32_t SLK_prePostOpsND;
extern uint32_t SLK_preConvert;
extern uint32_t SLK_padPreamble;
extern uint32_t SLK_padPreambleSingle;
extern uint32_t SLK_preGatherElements;
extern uint32_t SLK_interpolatePreamble;
extern uint32_t SLK_preCTCGreedyDecoderSeqLen;
extern uint32_t SLK_preSpaceToDepth;
extern uint32_t SLK_preDepthToSpace;
extern uint32_t SLK_reversesequencePreamble;
extern uint32_t SLK_preStridedSlice;
extern uint32_t SLK_LSTMCellPreamble;
extern uint32_t SLK_preGatherND;
extern uint32_t SLK_preScatterElementsUpdate;
extern uint32_t SLK_preScatterUpdate;
extern uint32_t SLK_preOutShapeOfReshape;
extern uint32_t SLK_preUpsampling;
extern uint32_t SLK_preNonZero;
extern uint32_t SLK_preBroadcast;
extern uint32_t SLK_preInnerLRN;
extern uint32_t SLK_preStaticShapeNMS;
extern uint32_t SLK_preExpDetectionOutput;
extern uint32_t SLK_preExpGenerateProposals;
extern uint32_t SLK_preExpPriorGridGenerator;
extern uint32_t SLK_preExpTopKROIs;

#ifdef __cplusplus
}
#endif
