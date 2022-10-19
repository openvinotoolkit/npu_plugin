//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/core/logger.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

namespace vpux {

//
// registerPipelines
//

void registerPipelines();

//
// ReferenceSWMode
//

struct ReferenceSWOptions : mlir::PassPipelineOptions<ReferenceSWOptions> {
    BoolOption enableProfiling{*this, "profiling", llvm::cl::desc("Enable profiling"), llvm::cl::init(false)};
    BoolOption enableSWProfiling{*this, "sw-profiling", llvm::cl::desc("Enable SW task profiling"),
                                 llvm::cl::init(true)};

    StrOption arch{*this, "vpu-arch", llvm::cl::desc("VPU architecture to compile for"), llvm::cl::init("VPUX30XX")};

    BoolOption enableUseUserPrecision{*this, "use-user-precision", llvm::cl::desc("Enable use-user-precision pass"),
                                      llvm::cl::init(true)};

    BoolOption enableUseUserLayout{*this, "use-user-layout", llvm::cl::desc("Enable use-user-layout pass"),
                                   llvm::cl::init(true)};

    BoolOption enableOptimizeReorders{*this, "optimize-reorders", llvm::cl::desc("Enable optimize-reorders pass"),
                                      llvm::cl::init(false)};

    bool enableCompressWeights = false;
    bool enableForceZMajorConcat = false;
    bool enableSwapTransposeWithFQ = false;
    bool enableSwapConcatWithEltwise = false;
};

void buildReferenceSWModePipeline(mlir::OpPassManager& pm, const ReferenceSWOptions& options,
                                  Logger log = Logger::global());
void buildEMUReferenceSWModePipeline(mlir::OpPassManager& pm, const ReferenceSWOptions& options,
                                     Logger log = Logger::global());

//
// ReferenceHWMode
//

struct ReferenceHWOptions : mlir::PassPipelineOptions<ReferenceHWOptions> {
    BoolOption enableProfiling{*this, "profiling", llvm::cl::desc("Enable profiling"), llvm::cl::init(false)};
    BoolOption enableDMAProfiling{*this, "dma-profiling", llvm::cl::desc("Enable DMA task profiling"),
                                  llvm::cl::init(true)};
    BoolOption enableDPUProfiling{*this, "dpu-profiling", llvm::cl::desc("Enable DPU task profiling"),
                                  llvm::cl::init(true)};
    BoolOption enableSWProfiling{*this, "sw-profiling", llvm::cl::desc("Enable SW task profiling"),
                                 llvm::cl::init(true)};

    StrOption arch{*this, "vpu-arch", llvm::cl::desc("VPU architecture to compile for"), llvm::cl::init("VPUX30XX")};

    IntOption numberOfDPUGroups{*this, "num-of-dpu-groups", llvm::cl::desc("Number of DPU groups")};

    IntOption numberOfDMAPorts{*this, "num-of-dma-ports", llvm::cl::desc("Number of DMA ports")};

    BoolOption enableUseUserPrecision{*this, "use-user-precision", llvm::cl::desc("Enable use-user-precision pass"),
                                      llvm::cl::init(true)};

    BoolOption enableConvertFCToConv{*this, "convert-fc-to-conv", llvm::cl::desc("Enable convert-fc-to-conv pass"),
                                     llvm::cl::init(true)};

    BoolOption enableConvertAvgPoolToDWConv{*this, "convert-avg-pool-to-dw-conv",
                                            llvm::cl::desc("Enable convert-avg-pool-to-dw-conv pass"),
                                            llvm::cl::init(true)};

    BoolOption enableConvertScaleShiftDW{*this, "convert-scale-shift-depthwise",
                                         llvm::cl::desc("Enable convert-scale-shift-depthwise pass"),
                                         llvm::cl::init(true)};

    BoolOption enableSplitConvWithMultipleFQ{*this, "split-conv-with-multiple-fq",
                                             llvm::cl::desc("Enable split-conv-with-multiple-fq pass"),
                                             llvm::cl::init(true)};

    BoolOption enableHandleLargeStrides{*this, "handle-large-strides",
                                        llvm::cl::desc("Enable handle-large-strides pass"), llvm::cl::init(true)};

    BoolOption enableHandleAsymmetricStrides{*this, "handle-asymmetric-strides",
                                             llvm::cl::desc("Enable handle-asymmetric-strides pass"),
                                             llvm::cl::init(true)};

    BoolOption enableLowPrecision{*this, "low-precision", llvm::cl::desc("Enable low-precision pipeline building"),
                                  llvm::cl::init(true)};

    BoolOption enableQuantDequantRemoval{*this, "quant-dequant-removal",
                                         llvm::cl::desc("Enable quantize->dequantize sequence removal"),
                                         llvm::cl::init(false)};

    BoolOption enableExpandActivationChannels{*this, "expand-activation-channels",
                                              llvm::cl::desc("Enable expand-activation-channels pass"),
                                              llvm::cl::init(true)};

    BoolOption enableUseUserLayout{*this, "use-user-layout", llvm::cl::desc("Enable use-user-layout pass"),
                                   llvm::cl::init(true)};

    BoolOption enableOptimizeReorders{*this, "optimize-reorders", llvm::cl::desc("Enable optimize-reorders pass"),
                                      llvm::cl::init(false)};

    bool enableCompressWeights = false;
    bool enableForceZMajorConcat = false;
    bool enableSwapTransposeWithFQ = false;
    BoolOption enablePropagateQuantDequant{*this, "propagate-quant-dequant",
                                           llvm::cl::desc("Enable Propagate Quantize Dequantize pass"),
                                           llvm::cl::init(true)};
    bool enableSwapConcatWithEltwise = false;
};

void buildReferenceHWModePipeline(mlir::OpPassManager& pm, const ReferenceHWOptions& options,
                                  Logger log = Logger::global());

void buildEMUReferenceHWModePipeline(mlir::OpPassManager& pm, const ReferenceHWOptions& options,
                                     Logger log = Logger::global());

//
// DefaultHWMode
//

struct DefaultHWOptions : mlir::PassPipelineOptions<DefaultHWOptions> {
    BoolOption enableProfiling{*this, "profiling", llvm::cl::desc("Enable profiling"), llvm::cl::init(false)};
    BoolOption enableDMAProfiling{*this, "dma-profiling", llvm::cl::desc("Enable DMA task profiling"),
                                  llvm::cl::init(true)};
    BoolOption enableDPUProfiling{*this, "dpu-profiling", llvm::cl::desc("Enable DPU task profiling"),
                                  llvm::cl::init(true)};
    BoolOption enableSWProfiling{*this, "sw-profiling", llvm::cl::desc("Enable SW task profiling"),
                                 llvm::cl::init(true)};

    StrOption arch{*this, "vpu-arch", llvm::cl::desc("VPU architecture to compile for"), llvm::cl::init("VPUX30XX")};

    StrOption actSparsityProfile{*this, "act-sparsity-profile", llvm::cl::desc("Activation sparsity profile"),
                                 llvm::cl::init("NONE")};

    IntOption numberOfDPUGroups{*this, "num-of-dpu-groups", llvm::cl::desc("Number of DPU groups")};

    IntOption numberOfDMAPorts{*this, "num-of-dma-ports", llvm::cl::desc("Number of DMA ports")};

    BoolOption enableUseUserPrecision{*this, "use-user-precision", llvm::cl::desc("Enable use-user-precision pass"),
                                      llvm::cl::init(true)};

    BoolOption enableConvertFCToConv{*this, "convert-fc-to-conv", llvm::cl::desc("Enable convert-fc-to-conv pass"),
                                     llvm::cl::init(true)};

    BoolOption enableConvertAvgPoolToDWConv{*this, "convert-avg-pool-to-dw-conv",
                                            llvm::cl::desc("Enable convert-avg-pool-to-dw-conv pass"),
                                            llvm::cl::init(true)};

    BoolOption enableConvertScaleShiftDW{*this, "convert-scale-shift-depthwise",
                                         llvm::cl::desc("Enable convert-scale-shift-depthwise pass"),
                                         llvm::cl::init(true)};

    BoolOption enableSplitConvWithMultipleFQ{*this, "split-conv-with-multiple-fq",
                                             llvm::cl::desc("Enable split-conv-with-multiple-fq pass"),
                                             llvm::cl::init(true)};

    BoolOption enableHandleLargeKernel{*this, "handle-large-kernel", llvm::cl::desc("Enable handle-large-kernel pass"),
                                       llvm::cl::init(true)};

    BoolOption enableHandleLargeStrides{*this, "handle-large-strides",
                                        llvm::cl::desc("Enable handle-large-strides pass"), llvm::cl::init(true)};

    BoolOption enableHandleAsymmetricStrides{*this, "handle-asymmetric-strides",
                                             llvm::cl::desc("Enable handle-asymmetric-strides pass"),
                                             llvm::cl::init(true)};

    BoolOption enableLowPrecision{*this, "low-precision", llvm::cl::desc("Enable low-precision pipeline building"),
                                  llvm::cl::init(true)};

    BoolOption enableQuantDequantRemoval{*this, "quant-dequant-removal",
                                         llvm::cl::desc("Enable quantize->dequantize sequence removal"),
                                         llvm::cl::init(false)};

    BoolOption enableUpstreamSlice{*this, "upstream-slice", llvm::cl::desc("Enable upstream-slice pipeline building"),
                                   llvm::cl::init(true)};

    BoolOption enableExpandActivationChannels{*this, "expand-activation-channels",
                                              llvm::cl::desc("Enable expand-activation-channels pass"),
                                              llvm::cl::init(true)};

    BoolOption enableUseUserLayout{*this, "use-user-layout", llvm::cl::desc("Enable use-user-layout pass"),
                                   llvm::cl::init(true)};

    BoolOption enableOptimizeReorders{*this, "optimize-reorders", llvm::cl::desc("Enable optimize-reorders pass"),
                                      llvm::cl::init(true)};

    BoolOption enableOptimizeCopies{*this, "optimize-copies", llvm::cl::desc("Enable optimize-copies pass"),
                                    llvm::cl::init(true)};

    BoolOption enableGroupAsyncExecuteOps{*this, "group-async-execute-ops",
                                          llvm::cl::desc("Enable group-async-execute-ops pass"), llvm::cl::init(true)};

    BoolOption enableCompressWeights{*this, "compress-weights", ::llvm::cl::desc("Enable compress-weights pass"),
                                     ::llvm::cl::init(false)};

    BoolOption enableForceZMajorConcat{*this, "force-z-major-concat",
                                       llvm::cl::desc("Enable transpose-reorder-concat pass"), llvm::cl::init(true)};

    BoolOption enableSwapTransposeWithFQ{*this, "swap-transpose-with-fq",
                                         ::llvm::cl::desc("Enable SwapTransposeWithFQ pass"), ::llvm::cl::init(true)};

    BoolOption enablePropagateQuantDequant{*this, "propagate-quant-dequant",
                                           llvm::cl::desc("Enable Propagate Quantize Dequantize pass"),
                                           llvm::cl::init(true)};

    BoolOption enableSwapConcatWithEltwise{*this, "swap-concat-with-eltwise",
                                           ::llvm::cl::desc("Enable SwapConcatWithEltwise pass"),
                                           ::llvm::cl::init(true)};

    BoolOption enableWeightsSwizzling{*this, "enable-weights-swizzling", ::llvm::cl::desc("Enable weights swizzling"),
                                      ::llvm::cl::init(true)};
    BoolOption enablePrefetchTiling{*this, "prefetch-tiling", llvm::cl::desc("Enable prefetch tiling pass"),
                                    llvm::cl::init(true)};

    BoolOption enableActivationSwizzling{*this, "enable-activation-swizzling",
                                         ::llvm::cl::desc("Enable activation swizzling"), ::llvm::cl::init(true)};
};

void buildDefaultHWModePipeline(mlir::OpPassManager& pm, const DefaultHWOptions& options,
                                Logger log = Logger::global());

void buildEMUDefaultHWModePipeline(mlir::OpPassManager& pm, const DefaultHWOptions& options,
                                   Logger log = Logger::global());

}  // namespace vpux
