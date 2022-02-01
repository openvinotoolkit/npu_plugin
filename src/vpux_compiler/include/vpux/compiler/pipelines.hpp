//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
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

    StrOption arch{*this, "vpu-arch", llvm::cl::desc("VPU architecture to compile for"), llvm::cl::init("KMB")};

    BoolOption enableUseUserPrecision{*this, "use-user-precision", llvm::cl::desc("Enable use-user-precision pass"),
                                      llvm::cl::init(true)};

    BoolOption enableUseUserLayout{*this, "use-user-layout", llvm::cl::desc("Enable use-user-layout pass"),
                                   llvm::cl::init(true)};

    BoolOption enableOptimizeReorders{*this, "optimize-reorders", llvm::cl::desc("Enable optimize-reorders pass"),
                                      llvm::cl::init(false)};

    bool enableCompressWeights = false;
};

void buildReferenceSWModePipeline(mlir::OpPassManager& pm, const ReferenceSWOptions& options,
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

    StrOption arch{*this, "vpu-arch", llvm::cl::desc("VPU architecture to compile for"), llvm::cl::init("KMB")};

    IntOption numberOfDPUGroups{*this, "num-of-dpu-groups", llvm::cl::desc("Number of DPU groups")};

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
};

void buildReferenceHWModePipeline(mlir::OpPassManager& pm, const ReferenceHWOptions& options,
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

    StrOption arch{*this, "vpu-arch", llvm::cl::desc("VPU architecture to compile for"), llvm::cl::init("KMB")};

    IntOption numberOfDPUGroups{*this, "num-of-dpu-groups", llvm::cl::desc("Number of DPU groups")};

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

    BoolOption enableQuantDequntRemoval{*this, "quant-dequant-removal",
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
};

void buildDefaultHWModePipeline(mlir::OpPassManager& pm, const DefaultHWOptions& options,
                                Logger log = Logger::global());

}  // namespace vpux
