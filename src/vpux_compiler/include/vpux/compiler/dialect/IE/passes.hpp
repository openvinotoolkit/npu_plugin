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

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/core/logger.hpp"

#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <memory>
#include <type_traits>

namespace vpux {
namespace IE {

//
// AdjustPrecision
//

struct AdjustPrecisionOptions : mlir::PassPipelineOptions<AdjustPrecisionOptions> {
    BoolOption enableUseUserPrecision{*this, "use-user-precision", llvm::cl::desc("Enable use-user-precision pass"),
                                      llvm::cl::init(true)};

    AdjustPrecisionOptions() = default;

    template <
            class OtherOptions,
            typename = std::enable_if_t<std::is_base_of<mlir::PassPipelineOptions<OtherOptions>, OtherOptions>::value>>
    explicit AdjustPrecisionOptions(const OtherOptions& options) {
        enableUseUserPrecision = options.enableUseUserPrecision;
    }
};

void buildAdjustPrecisionPipeline(mlir::OpPassManager& pm, const AdjustPrecisionOptions& options,
                                  Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createConvertPrecisionToFP16Pass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertPrecisionToI32Pass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUseUserPrecisionPass(Logger log = Logger::global());

//
// AdjustLayout
//

struct AdjustLayoutOptions : mlir::PassPipelineOptions<AdjustLayoutOptions> {
    BoolOption enableUseUserLayout{*this, "use-user-layout", llvm::cl::desc("Enable use-user-layout pass"),
                                   llvm::cl::init(true)};

    BoolOption enableOptimizeReorders{*this, "optimize-reorders", llvm::cl::desc("Enable optimize-reorders pass"),
                                      llvm::cl::init(true)};

    BoolOption enableForceZMajorConcat{*this, "force-z-major-concat",
                                       llvm::cl::desc("Enable transpose-reorder-concat pass"), llvm::cl::init(false)};

    AdjustLayoutOptions() = default;

    template <
            class OtherOptions,
            typename = std::enable_if_t<std::is_base_of<mlir::PassPipelineOptions<OtherOptions>, OtherOptions>::value>>
    explicit AdjustLayoutOptions(const OtherOptions& options) {
        enableUseUserLayout = options.enableUseUserLayout;
        enableOptimizeReorders = options.enableOptimizeReorders;
        enableForceZMajorConcat = options.enableForceZMajorConcat;
    }
};

void buildAdjustLayoutPipeline(mlir::OpPassManager& pm, const AdjustLayoutOptions& options,
                               Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createUseUserLayout(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustLayoutsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeReordersPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUniquifyOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertToMemPermutePass(Logger log = Logger::global());

//
// AdjustForVPU
//

void buildAdjustForVPUPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createConvertTile2PerAxisTilePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertShapeTo4DPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertConv1DToConv2DPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertPaddingsToFloorModePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertShuffleChannelsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createResolveStridedSlicePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFusePostOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFusePadOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertPadToConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapMaxPoolWithActivation(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertDeconv2DToConv2DPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertDepth2SpaceLayerPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createInsertMaxpoolToConcatLReluPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createInsertReorderBetweenTransposeAndConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapTransposeWithFQPass(Logger log = Logger::global());

//
// LowPrecision
//

struct LowPrecisionOptions : mlir::PassPipelineOptions<LowPrecisionOptions> {
    BoolOption enableQuantDequantRemoval{*this, "quant-dequant-removal",
                                         llvm::cl::desc("Enable quantize->dequantize sequence removal"),
                                         llvm::cl::init(false)};

    BoolOption enableSwapTransposeWithFQ{*this, "swap-transpose-with-fq",
                                         ::llvm::cl::desc("Enable SwapTransposeWithFQ pass"), ::llvm::cl::init(false)};

    BoolOption enablePropagateQuantDequant{*this, "propagate-quant-dequant",
                                           llvm::cl::desc("Enable Propagate Quantize Dequantize pass"),
                                           llvm::cl::init(true)};

    LowPrecisionOptions() = default;

    template <
            class OtherOptions,
            typename = std::enable_if_t<std::is_base_of<mlir::PassPipelineOptions<OtherOptions>, OtherOptions>::value>>
    explicit LowPrecisionOptions(const OtherOptions& options) {
        enableQuantDequantRemoval = options.enableQuantDequantRemoval;
        enableSwapTransposeWithFQ = options.enableSwapTransposeWithFQ;
        enablePropagateQuantDequant = options.enablePropagateQuantDequant;
    }
};

void buildLowPrecisionPipeline(mlir::OpPassManager& pm, const LowPrecisionOptions& options,
                               Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createSplitFakeQuantPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateQuantizeDequantizePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDequantizeConstPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMergeFakeQuantPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseQuantizedOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createRemoveQuantDequantSeqPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertWeightsToU8Pass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseConvertWithQuantizePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertQuantizeOpsToEltwisePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDeletePerAxisQuantizationPass(Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createResolvePWLPostOpsPass(Logger log = Logger::global());

//
// Legalization for NCE
//

std::unique_ptr<mlir::Pass> createUnrollBatchPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMatMulInputsTo2dPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertFCToConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertAvgPoolToDWConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertScaleShiftToDWPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertNearestToStridedConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitConvWithMultipleFQPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createHandleLargeStridesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createHandleAsymmetricStridesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createExpandActivationChannelsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createHandleLargeKernelsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertReduceToPoolingPass(Logger log = Logger::global());

//
// Tiling
//

std::unique_ptr<mlir::Pass> createIsolatedTilingPass(Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createPrefetchTilingPass(Logger log = Logger::global());

//
// Generic Optimizations
//

std::unique_ptr<mlir::Pass> createUpstreamSlicePass(Logger log = Logger::global());

//
// Registration
//

void registerIEPipelines();

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/dialect/IE/generated/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/dialect/IE/generated/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace IE
}  // namespace vpux
