//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/dialect.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/optional.hpp"

#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <memory>

namespace vpux {
namespace VPU {

using SparsityProfileCreateFunc = std::function<Optional<VPU::ActivationSparsityProfile>(StringRef)>;

//
// Activation sparsity options
//

struct ActivationSparsityOptions : mlir::PassPipelineOptions<ActivationSparsityOptions> {
    StrOption enableActivationSparsity{*this, "enable-activation-sparsity",
                                       llvm::cl::desc("Enable activation sparsity"), llvm::cl::init("auto")};
    StrOption actSparsityProfile{*this, "act-sparsity-profile", llvm::cl::desc("Activation sparsity profile"),
                                 llvm::cl::init("NONE")};

    ActivationSparsityOptions() = default;

    template <
            class OtherOptions,
            typename = std::enable_if_t<std::is_base_of<mlir::PassPipelineOptions<OtherOptions>, OtherOptions>::value>>
    explicit ActivationSparsityOptions(const OtherOptions& options) {
        enableActivationSparsity = options.enableActivationSparsity;
        actSparsityProfile = options.actSparsityProfile;
    }
};

//
// Weights sparsity options
//

struct WeightsSparsityOptions : mlir::PassPipelineOptions<WeightsSparsityOptions> {
    StrOption weightsSparsityHeuristic{*this, "weights-sparsity-heuristic",
                                       llvm::cl::desc("Weights sparsity heuristic (ratio or cmx)"),
                                       llvm::cl::init("ratio")};
    DoubleOption weightsSparsityThreshold{*this, "weights-sparsity-threshold",
                                          llvm::cl::desc("Weights sparsity threshold")};

    WeightsSparsityOptions() = default;

    template <
            class OtherOptions,
            typename = std::enable_if_t<std::is_base_of<mlir::PassPipelineOptions<OtherOptions>, OtherOptions>::value>>
    explicit WeightsSparsityOptions(const OtherOptions& options) {
        weightsSparsityHeuristic = options.weightsSparsityHeuristic;
        weightsSparsityThreshold = options.weightsSparsityThreshold;
    }
};

//
// Tiling options
//

struct TilingOptions : mlir::PassPipelineOptions<TilingOptions> {
    BoolOption enablePrefetchTiling{*this, "enable-prefetch", llvm::cl::desc("Enable prefetch mode"),
                                    llvm::cl::init(true)};

    BoolOption enableVerticalFusion{*this, "vertical-fusion", llvm::cl::desc("Enable vertical fusion feature"),
                                    llvm::cl::init(false)};

    TilingOptions() = default;

    template <
            class OtherOptions,
            typename = std::enable_if_t<std::is_base_of<mlir::PassPipelineOptions<OtherOptions>, OtherOptions>::value>>
    explicit TilingOptions(const OtherOptions& options) {
        enablePrefetchTiling = options.enablePrefetchTiling;
        enableVerticalFusion = options.enableVerticalFusion;
    }
};

//
// Passes
//

std::unique_ptr<mlir::Pass> createInitCompilerPass();
std::unique_ptr<mlir::Pass> createInitCompilerPass(ArchKind arch, CompilationMode compilationMode,
                                                   Optional<int> numOfDPUGroups = None,
                                                   Optional<int> numOfDMAPorts = None, Optional<int> ddrHeapSize = None,
                                                   Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createCMXConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitNCEOpsOntoWorkloadsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createCorrectNCEWorkloadsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createResolveEltwiseWithZTiledWorkloadsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createWrapVPUOpsInNCEClusterTilingPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustMemorySpacePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMultiClusterStrategyAssignmentPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createManualStrategyUtilsPass();
std::unique_ptr<mlir::Pass> createManualStrategyUtilsPass(bool writeStrategyToJSON,
                                                          StringRef writeStrategyFileLocation = "strategy_out.json",
                                                          bool readStrategyFromJSON = false,
                                                          StringRef readStrategyFileLocation = "strategy_in.json",
                                                          Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createResolvePWLPostOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDetectionOutputDecompositionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitGRUSequencePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDetectInPlaceEltwisePass(Logger log = Logger::global());

//
// Sparsity
//

std::unique_ptr<mlir::Pass> createSparsifyWeightsPass(
        VPU::WeightsSparsityHeuristic heuristic = VPU::WeightsSparsityHeuristic::RATIO,
        Optional<double> manualThreshold = None, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createRecomputeSparsityPtrsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseSparsityOpsPass(Optional<bool> fuseSparsify = None,
                                                      Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeSparsifyDesparsifyPairsPass(SparsityProfileCreateFunc sparsityProfileCreateCb,
                                                                      Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeSparsityOpsPass(SparsityProfileCreateFunc sparsityProfileCreateCb,
                                                          Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createWrapOpsInSparsifyDesparsifyPairsPass();
std::unique_ptr<mlir::Pass> createWrapOpsInSparsifyDesparsifyPairsPass(
        VPU::EnableActivationSparsityMode enableActivationSparsityMode,
        VPU::ActivationSparsityProfile actSparsityProfile, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddSparsityMapToSparseActivationsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLowerSparsityOpsPass(Optional<bool> fakeSparsify = None,
                                                       Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLowerOpsToSENCEPass(Logger log = Logger::global());

//
// Tiling
//

std::unique_ptr<mlir::Pass> createTilingStrategyAssignmentPass(bool enablePrefetchTiling = true,
                                                               Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createApplyTilingPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createWrapVerticalFusionRegionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMergeVfSubgraphsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createVfTilingPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollUnusedVerticalFusionRegionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustVFTilingStrategyPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSetupPPEPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createEnsureNCEOpsSizeRequirementsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseClampPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustTilingForPermuteQuantizePass(Logger log = Logger::global());

void buildActivationSparsityPipeline(mlir::OpPassManager& pm, const VPU::ActivationSparsityOptions& options,
                                     Logger log = Logger::global());

void buildWeightsSparsityPipeline(mlir::OpPassManager& pm, const VPU::WeightsSparsityOptions& options,
                                  Logger log = Logger::global());
void buildTilingPipeline(mlir::OpPassManager& pm, const VPU::TilingOptions& options, Logger log = Logger::global());

//
// Registration
//

void registerVPUPipelines();

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/dialect/VPU/generated/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/dialect/VPU/generated/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace VPU
}  // namespace vpux
