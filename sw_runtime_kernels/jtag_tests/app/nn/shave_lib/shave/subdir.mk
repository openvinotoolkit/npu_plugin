#
# Copyright Intel Corporation.
#
# LEGAL NOTICE: Your use of this software and any required dependent software
# (the "Software Package") is subject to the terms and conditions of
# the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
# which may also include notices, disclaimers, or license terms for
# third party or open source software included in or with the Software Package,
# and your use indicates your acceptance of all such terms. Please refer
# to the "third-party-programs.txt" or other similarly-named text file
# included with the Software Package for additional details.
#

kmb_or_tbh =

ifeq ($(CONFIG_TARGET_SOC_MA2490), y)
kmb_or_tbh = y
endif
ifeq ($(CONFIG_TARGET_SOC_3100), y)
kmb_or_tbh = y
endif

ifdef kmb_or_tbh
$(eval $(call mdk-redirect-srcs,srcs-shave-y,shavelib-objs-y))
srcs-shave-y += $(wildcard kernels/*.c)
srcs-shave-y += $(wildcard kernels/*.cpp)
srcs-shave-y += $(wildcard src/*.c)
srcs-shave-y += $(wildcard src/*.cpp)

ifeq ($(CONFIG_SVU_STACK_USAGE_INSTRUMENTATION), y)
ccopt-shave-y += -mstack-usage-instrumentation
endif

include-dirs-shave-y += ../inc ../inc/layers ./inc ../../inference_runtime_common/inc

shavelib-exported-symbols-y += sData sParam sPerfCounters curPerfCounter
shavelib-preserved-symbols-y := mvSoftMax
shavelib-preserved-symbols-y += nnPermute
shavelib-preserved-symbols-y += nnQuantizer_EqualScales nnQuantizer_PerChannelScales nnQuantizer_EqualScales_StridedIn_CompactOut
shavelib-preserved-symbols-y += nnResample nnResampleAntialias
shavelib-preserved-symbols-y += Passthrough
shavelib-preserved-symbols-y += PriorboxKernel
shavelib-preserved-symbols-y += ROIPoolingKernel
shavelib-preserved-symbols-y += custom_ocl
shavelib-preserved-symbols-$(CONFIG_TARGET_SOC_MA2490) += custom_cpp
shavelib-preserved-symbols-$(CONFIG_TARGET_SOC_MA2490) += singleSoftmaxKernel
shavelib-preserved-symbols-$(CONFIG_TARGET_SOC_MA2490) += sigmoid_fp16
shavelib-preserved-symbols-$(CONFIG_TARGET_SOC_MA2490) += reorder_fp16
shavelib-preserved-symbols-$(CONFIG_TARGET_SOC_MA2490) += hswish_fp16
shavelib-preserved-symbols-y += singleShaveMVN
shavelib-preserved-symbols-$(CONFIG_TARGET_SOC_MA2490) += elu_fp16
shavelib-preserved-symbols-$(CONFIG_TARGET_SOC_MA2490) += exp_fp16
shavelib-preserved-symbols-$(CONFIG_TARGET_SOC_MA2490) += tanh_fp16
shavelib-preserved-symbols-$(CONFIG_TARGET_SOC_MA2490) += power_fp16
shavelib-preserved-symbols-$(CONFIG_TARGET_SOC_MA2490) += add_fp16
shavelib-preserved-symbols-$(CONFIG_TARGET_SOC_MA2490) += sub_fp16
shavelib-preserved-symbols-$(CONFIG_TARGET_SOC_MA2490) += min_fp16
shavelib-preserved-symbols-$(CONFIG_TARGET_SOC_MA2490) += max_fp16
shavelib-preserved-symbols-$(CONFIG_TARGET_SOC_MA2490) += mul_fp16
shavelib-preserved-symbols-$(CONFIG_TARGET_SOC_MA2490) += div_fp16
shavelib-preserved-symbols-y += edsl
shavelib-preserved-symbols-y += Dummy
shavelib-preserved-symbols-y += proposal
shavelib-preserved-symbols-y += regionYoloKernel_CHW regionYoloKernel_HWC
shavelib-preserved-symbols-y += DetectionOutput
shavelib-preserved-symbols-y += normalize
shavelib-preserved-symbols-y += argMaxKernel_WithNoAxis argMaxKernel_WithAxis argMaxKernel_WithAxis1_Top1
shavelib-preserved-symbols-y += ReorgYoloHWC ReorgYoloCHW
shavelib-preserved-symbols-y += interpCHW interpCHW_1x1 interpHWC interpHWC_1x1
shavelib-preserved-symbols-y += Norm_AcrossChannel Norm_SameChannel
shavelib-preserved-symbols-y += Eltwise
shavelib-preserved-symbols-y += AvgPoolingKernel
shavelib-preserved-symbols-y += correlation
shavelib-preserved-symbols-y += CTCDecoder
shavelib-preserved-symbols-y += GRN
shavelib-preserved-symbols-y += MVN
shavelib-preserved-symbols-y += nnSpatialTransform
shavelib-preserved-symbols-y += fakeQuantize_CHW fakeQuantize_HWC
shavelib-preserved-symbols-y += nnEntry
shavelib-preserved-symbols-y += preQuantizer preQuantizer_EqualScales_StridedIn_CompactOut dummyPreamble cleanDummy interpPreamble preNormalize prePriorbox preProposal cleanProposal detOutPreamble detOutCleanup preResample
shavelib-preserved-symbols-y += preSoftmax singleShaveSoftmax preCorrelation preROIPooling preCTCDecoder preReorgYolo preRegionYolo prePermute prePermute1D eltwisePreamble cleanEltwise argmaxPreamble preAvgPooling
shavelib-preserved-symbols-y += preCHW_maxPoolMxN preHWC_maxPoolMxN CHW_mvMaxPoolMxN HWC_mvMaxPoolMxN
shavelib-preserved-symbols-y += preSpatialTransform preCustomLayerOcl preFakeQuantize grnPreamble mvnPreamble preNorm prePassthrough preReShape execCleanupCustomLayerOcl
shavelib-preserved-symbols-$(CONFIG_TARGET_SOC_MA2490) += execCleanupCustomLayerCpp preCustomLayerCpp 
shavelib-preserved-symbols-y += PSROIPooling prePSROIPooling preEdsl
shavelib-preserved-symbols-y += Tile preTile
shavelib-preserved-symbols-y += prePostOpsCHW prePostOpsHWC prePostOpsHCW prePostOpsND
shavelib-preserved-symbols-y += chw_postOps_3D_core hwc_postOps_3D_core hcw_postOps_3D_core postOps_ND_core
shavelib-preserved-symbols-y += mvConvert preConvert
shavelib-preserved-symbols-y += PadKernel padPreamble padPreambleSingle
shavelib-preserved-symbols-y += mvGatherElements preGatherElements
shavelib-preserved-symbols-y += interpolatePreamble
shavelib-preserved-symbols-y += CTCGreedyDecoderSeqLen preCTCGreedyDecoderSeqLen
shavelib-preserved-symbols-y += preSpaceToDepth mvSpaceToDepth
shavelib-preserved-symbols-y += preDepthToSpace mvDepthToSpace
shavelib-preserved-symbols-y += mvReverseSequence reversesequencePreamble
shavelib-preserved-symbols-y += preStridedSlice
shavelib-preserved-symbols-y += LSTMCell LSTMCellPreamble
shavelib-preserved-symbols-y += preScatterElementsUpdate
shavelib-preserved-symbols-y += preScatterUpdate
shavelib-preserved-symbols-y += preOutShapeOfReshape
shavelib-preserved-symbols-y += preUpsampling mvUpsamplingMEM mvUpsamplingDMA
shavelib-preserved-symbols-y += mvNonZeroCollect preNonZero 
shavelib-preserved-symbols-y += preBroadcast
shavelib-preserved-symbols-y += preInnerLRN mvInnerLRN
shavelib-preserved-symbols-y += preStaticShapeNMS mvNonMaxSuppression
shavelib-preserved-symbols-y += preExpDetectionOutput mvExpDetectionOutput
shavelib-preserved-symbols-y += preExpGenerateProposals mvExpGenerateProposals
shavelib-preserved-symbols-y += preExpPriorGridGenerator mvExpPriorGridGenerator
shavelib-preserved-symbols-y += preExpTopKROIs mvExpTopKROIs

# deconvolution
shavelib-preserved-symbols-y += depthDeconvCHWPreamble depthDeconvHWCPreamble deconvNxM_hwcPreamble deconvNxM_chwPreamble
shavelib-preserved-symbols-y += mvDepthDeconv_chw mvDepthDeconv_hwc mvDeconvNxM_hwc mvDeconvNxM_chw

shavelib-preserved-symbols-y += mvConv_ref refConvPreamble
shavelib-preserved-symbols-y += fullyConnected fullyConnectedPreamble
shavelib-preserved-symbols-y += mvSWConvolution swConvolutionPreamble

shavelib-preserved-symbols-y += preGather
shavelib-preserved-symbols-y += preGatherND

shavelib-preserved-symbols-y += negativePreamble negativeKernel
#TODO add shave pipeprint tx queue symbol

global-symbols-y += $(foreach ep, $(shavelib-exported-symbols-y), $(SYM_PREFIX)$(ep)) svuSLKernels_Base

SYM_PREFIX=SLK_
# -include $(svuSLKernelsEP)
ccopt-lrt-y += -DKERNEL_SYM_PREFIX=$(SYM_PREFIX)

include $(MDK_ROOT_PATH)/build/buildSupport/buildShaveRisa.mk

$(eval $(call RISA_LIB_RULES,svuSLKernels,$(SYM_PREFIX),\
$(shavelib-objs-y) $(shave_OBJS),\
$(shavelib-exported-symbols-y) $(shavelib-preserved-symbols-y),svuSLKernels_Base))

endif
