//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/pipelines_options.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
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

using SparsityProfileCreateFunc = std::function<std::optional<VPU::ActivationSparsityProfile>(StringRef)>;

//
// Activation sparsity options
//

struct ActivationSparsityOptions : mlir::PassPipelineOptions<ActivationSparsityOptions> {
    StrOption enableActivationSparsity{*this, "enable-activation-sparsity",
                                       llvm::cl::desc("Enable activation sparsity"), llvm::cl::init("auto")};
    StrOption actSparsityProfile{*this, "act-sparsity-profile", llvm::cl::desc("Activation sparsity profile"),
                                 llvm::cl::init("NONE")};

    ActivationSparsityOptions() = default;

    template <class OtherOptions>
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

    template <class OtherOptions>
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

    BoolOption enableVPUNNCost{*this, "vpunn-cost",
                               llvm::cl::desc("Use VPUNN cost model to get the best tiling strategy"),
                               llvm::cl::init(false)};

    BoolOption enableVerticalFusion{*this, "vertical-fusion", llvm::cl::desc("Enable vertical fusion feature"),
                                    llvm::cl::init(false)};

    BoolOption enableVerticalFusionPipelining{*this, "vertical-fusion-pipelining",
                                              llvm::cl::desc("Enable vertical fusion pipelining"),
                                              llvm::cl::init(false)};

    // Extended Tiling options - Incremental Pipeline
    BoolOption readStrategyFromJson{*this, "read-strategy-from-json",
                                    llvm::cl::desc("Read the multiclustering and tiling strategy from a JSON file"),
                                    llvm::cl::init(false)};

    BoolOption writeStrategyToJson{*this, "write-strategy-to-json",
                                   llvm::cl::desc("Write the multiclustering and tiling strategy to a JSON file"),
                                   llvm::cl::init(false)};

    TilingOptions() = default;

    template <class OtherOptions>
    explicit TilingOptions(const OtherOptions& options) {
        enablePrefetchTiling = options.enablePrefetching;
        enableVPUNNCost = options.enableVPUNNCost;
        enableVerticalFusion = options.enableVerticalFusion;
        enableVerticalFusionPipelining = options.enablePipelining;
        readStrategyFromJson = options.readStrategyFromJson;
        writeStrategyToJson = options.writeStrategyToJson;
    }
};

//
// Passes
//

std::unique_ptr<mlir::Pass> createInitCompilerPass();
std::unique_ptr<mlir::Pass> createInitCompilerPass(ArchKind arch, CompilationMode compilationMode,
                                                   std::optional<int> numOfDPUGroups = std::nullopt,
                                                   std::optional<int> numOfDMAPorts = std::nullopt,
                                                   Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createCMXConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitNCEOpsOntoWorkloadsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createResolveEltwiseWithZTiledWorkloadsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createWrapVPUOpsInNCEClusterTilingPass(bool enableExplicitDistributedTensorAttr = false,
                                                                   Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustMemorySpacePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMultiClusterStrategyAssignmentPass(bool enablePrefetchTiling = true,
                                                                     Logger log = Logger::global());
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
std::unique_ptr<mlir::Pass> createFuseNCEInterpolateConsumersPass(Logger log = Logger::global());

//
// Sparsity
//

std::unique_ptr<mlir::Pass> createSparsifyWeightsPass(
        VPU::WeightsSparsityHeuristic heuristic = VPU::WeightsSparsityHeuristic::RATIO,
        std::optional<double> manualThreshold = std::nullopt, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createRecomputeSparsityPtrsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseSparsityOpsPass(std::optional<bool> fuseSparsify = std::nullopt,
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
std::unique_ptr<mlir::Pass> createLowerSparsityOpsPass(std::optional<bool> fakeSparsify = std::nullopt,
                                                       Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitSEOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLowerOpsToSENCEPass(Logger log = Logger::global());

//
// Tiling
//

std::unique_ptr<mlir::Pass> createTilingStrategyAssignmentPass(bool enablePrefetchTiling = true, bool vpunnCost = false,
                                                               Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createApplyTilingPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createTileOverHForVFPass(bool enablePrefetchTiling = true, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createWrapVerticalFusionRegionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMoveViewOpsToVerticalFusionPass(Logger log = Logger::global());
// Tracking number [E#76838]
// Turn on the enableVerticalFusionPipelining when VF pipelining is enabled
std::unique_ptr<mlir::Pass> createMergeVfSubgraphsPass(bool enableVerticalFusionPipelining = false,
                                                       Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createVfTilingPass(bool enableVerticalFusionPipelining = false,
                                               Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollUnusedVerticalFusionRegionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createRollBackTilingStrategyPass(bool enablePrefetchTiling = true,
                                                             Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustVFTilingStrategyPass(bool enableVerticalFusionPipelining = false,
                                                             Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSetupPPEPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createEnsureNCEOpsSizeRequirementsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseClampPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustTilingForPermuteQuantizePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createStrategyManagerImplPass(bool enablePrefetchTiling = true,
                                                          Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createRemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass(Logger log = Logger::global());

void buildActivationSparsityPipeline(mlir::OpPassManager& pm, const VPU::ActivationSparsityOptions& options,
                                     Logger log = Logger::global());

void buildWeightsSparsityPipeline(mlir::OpPassManager& pm, const VPU::WeightsSparsityOptions& options,
                                  Logger log = Logger::global());
void buildTilingPipeline(mlir::OpPassManager& pm, const VPU::TilingOptions& options, Logger log = Logger::global());

//
// Strategy Pipeline
//

void buildVFPipeline(mlir::OpPassManager& pm, const VPU::TilingOptions& options, Logger log = Logger::global());
void buildSMPipeline(mlir::OpPassManager& pm, const vpux::MCAndTilingOptionsBase& options,
                     Logger log = Logger::global());

//
// DefaultHWOptions(for all devices)
//
struct DefaultHWOptionsDialectBase : public virtual vpux::DefaultHWOptionsBase {
    BoolOption enableInPlaceEltwise{*this, "enable-in-place-eltwise",
                                    llvm::cl::desc("Enable inplace eltwise op execution"), llvm::cl::init(true)};

    BoolOption enableSMPipeline{*this, "enable-SM-Pipeline", llvm::cl::desc("Enable Strategy Manager pipeline"),
                                llvm::cl::init(false)};

    // WeightsSparsityOptions
    StrOption weightsSparsityHeuristic{*this, "weights-sparsity-heuristic",
                                       llvm::cl::desc("Weights sparsity heuristic (RATIO or CMX)"),
                                       llvm::cl::init("RATIO")};

    DoubleOption weightsSparsityThreshold{*this, "weights-sparsity-threshold",
                                          llvm::cl::desc("Threshold for ratio of sparse weights values"),
                                          llvm::cl::init(-1.0)};

    // TilingOptions
    BoolOption enableVerticalFusion{*this, "vertical-fusion", llvm::cl::desc("Enable vertical fusion feature"),
                                    llvm::cl::init(true)};

    BoolOption readStrategyFromJson{*this, "read-strategy-from-json",
                                    llvm::cl::desc("Read the multiclustering and tiling strategy from a JSON file"),
                                    llvm::cl::init(false)};

    BoolOption writeStrategyToJson{*this, "write-strategy-to-json",
                                   llvm::cl::desc("Write the multiclustering and tiling strategy to a JSON file"),
                                   llvm::cl::init(false)};
};

//
// Registration
//

void registerVPUPipelines();

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/dialect/VPU/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/dialect/VPU/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace VPU
}  // namespace vpux
