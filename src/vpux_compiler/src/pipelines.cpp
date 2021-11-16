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

#include "vpux/compiler/pipelines.hpp"

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/core/passes.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/IERT/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes/arch.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/optional.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// Common utilities
//

namespace {
using IntOption = mlir::detail::PassOptions::Option<int>;
using StrOption = mlir::detail::PassOptions::Option<std::string>;
using BoolOption = mlir::detail::PassOptions::Option<bool>;

// This structure gets command line options for a pipeline.
struct MyPipelineOptions : mlir::PassPipelineOptions<MyPipelineOptions> {
    StrOption archOpt{*this, "vpu-arch", ::llvm::cl::desc("VPU architecture to compile for"), ::llvm::cl::init("KMB")};
    IntOption numberOfDPUGroupsOpt{*this, "num-of-dpu-groups", ::llvm::cl::desc("Number of DPU groups")};
    BoolOption enableLowPrecisionBuilding{
            *this, "low-precision", ::llvm::cl::desc("Enable low-precision pipeline building"), ::llvm::cl::init(true)};
    BoolOption enableConvertFCToConv{*this, "convert-fc-to-conv", ::llvm::cl::desc("Enable convert-fc-to-conv pass"),
                                     ::llvm::cl::init(true)};
    BoolOption enableConvertAvgPoolToDWConv{*this, "convert-avg-pool-to-dw-conv",
                                            ::llvm::cl::desc("Enable convert-avg-pool-to-dw-conv pass"),
                                            ::llvm::cl::init(true)};
    BoolOption enableConvertScaleShiftDW{*this, "convert-scale-shift-depthwise",
                                         ::llvm::cl::desc("Enable convert-scale-shift-depthwise pass"),
                                         ::llvm::cl::init(true)};
    BoolOption enableHandleAsymetricStrides{*this, "handle-asymmetric-strides",
                                            ::llvm::cl::desc("Enable handle-asymmetric-strides pass"),
                                            ::llvm::cl::init(true)};
    BoolOption enableExpandActivationChannels{*this, "expand-activation-channels",
                                              ::llvm::cl::desc("Enable expand-activation-channels pass"),
                                              ::llvm::cl::init(true)};
    BoolOption enableOptimizeCopies{*this, "optimize-copies", ::llvm::cl::desc("Enable optimize-copies pass"),
                                    ::llvm::cl::init(true)};
    BoolOption enableOptimizeAsyncDeps{*this, "optimize-async-deps",
                                       ::llvm::cl::desc("Enable optimize-async-deps pass"), ::llvm::cl::init(true)};
    BoolOption enableUseUserLayout{*this, "use-user-layout", ::llvm::cl::desc("Enable use-user-layout pass"),
                                   ::llvm::cl::init(true)};
    BoolOption enableOptimizeReorders{*this, "optimize-reorders", ::llvm::cl::desc("Enable optimize-reorders pass"),
                                      ::llvm::cl::init(true)};
    BoolOption enableGroupAsyncExecuteOps{*this, "group-async-execute-ops",
                                          ::llvm::cl::desc("Enable group-async-execute-ops pass"),
                                          ::llvm::cl::init(true)};
};

template <VPUIP::PhysicalMemory KIND>
mlir::Attribute getMemSpace(mlir::MLIRContext* ctx, StringRef) {
    return VPUIP::PhysicalMemoryAttr::get(ctx, KIND);
}

void buildIECommonPipeline(mlir::OpPassManager& pm, Logger log, const MyPipelineOptions& pipelineOptions = {}) {
    const auto grc = getDefaultGreedyRewriteConfig();

    if (pipelineOptions.enableUseUserLayout.getValue())
        pm.addPass(IE::createUseUserLayout(log));
    pm.addPass(IE::createAdjustLayoutsPass(log));
    if (pipelineOptions.enableOptimizeReorders.getValue())
        pm.addPass(IE::createOptimizeReordersPass(log));
    pm.addPass(IE::createConvertToMemPermutePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

void buildIEReferenceLowPrecisionPipeline(mlir::OpPassManager& pm, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(IE::createSplitFakeQuantPass(log));
    pm.addPass(IE::createDequantizeConstPass(log));
    pm.addPass(IE::createMergeFakeQuantPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

VPUIP::ArchKind getArchKind(const StrOption& archKind) {
    VPUX_THROW_UNLESS(archKind.hasValue(), "Platform architecture is not provided. Please try 'vpu-arch=KMB'");
    const std::string archKindStr = archKind.getValue();
    ::llvm::Optional<VPUIP::ArchKind> parsed = VPUIP::symbolizeArchKind(archKindStr);
    VPUX_THROW_UNLESS(parsed.hasValue(), "Unsupported platform architecture '{0}'", archKindStr);
    return parsed.getValue();
}

llvm::Optional<int> getNumOfDPUGroups(const IntOption& numOfDPUGroups) {
    if (numOfDPUGroups.hasValue())
        return numOfDPUGroups.getValue();
    return None;
}

void addConfigPass(mlir::OpPassManager& pm, const MyPipelineOptions& config, VPUIP::CompilationMode compilationMode) {
    const VPUIP::ArchKind archKind = getArchKind(config.archOpt);
    const llvm::Optional<int> numOfDPUGroups = getNumOfDPUGroups(config.numberOfDPUGroupsOpt);
    Logger log = Logger::global();
    pm.addPass(createSetCompileParamsPass(archKind, compilationMode, numOfDPUGroups, log.nest()));
}
}  // namespace

//
// ReferenceSWMode
//

void vpux::buildReferenceSWModePipeline(mlir::OpPassManager& pm, bool enableProfiling, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(mlir::createCanonicalizerPass(grc));

    // IE Dialect level
    IE::buildAdjustPrecisionPipeline(pm, log);
    IE::buildAdjustForVPUPipeline(pm, log);

    buildIEReferenceLowPrecisionPipeline(pm, log);

    pm.addPass(IE::createUpstreamSlicePass(log));

    buildIECommonPipeline(pm, log);

    // Lower IE->IERT
    buildLowerIE2IERTPipeline(pm, log);

    if (enableProfiling) {
        pm.addPass(IERT::createTimestampProfilingPass(getMemSpace<VPUIP::PhysicalMemory::DDR>, log));
    }

    // Partially lower IERT->VPUIP (Act shave tasks only)
    pm.addPass(createConvertSWLayers2VPUIPPass(log));

    // IERT Dialect level
    IERT::buildAsyncSchedulingPipeline(pm, log);
    pm.addPass(IERT::createGroupAsyncExecuteOpsPass(log));
    pm.addPass(IERT::createSetInternalMemorySpacePass(getMemSpace<VPUIP::PhysicalMemory::DDR>, log));
    pm.addPass(IERT::createStaticAllocationPass(getMemSpace<VPUIP::PhysicalMemory::DDR>, log));
    pm.addPass(IERT::createOptimizeAsyncDepsPass(log));

    // Lower IERT->VPUIP (SW mode)
    buildLowerIERT2VPUIPPipeline(pm, log);

    // VPUIP Dialect level
    pm.addPass(VPUIP::createAssignPhysicalBarriersPass(log));
    pm.addPass(VPUIP::createBarrierSimulationPass(log));
    pm.addPass(VPUIP::createDumpStatisticsOfTaskOpsPass(log));
}

//
// ReferenceHWMode
//

void vpux::buildReferenceHWModePipeline(mlir::OpPassManager& pm, bool enableProfiling, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();
    pm.addPass(mlir::createCanonicalizerPass(grc));

    // IE Dialect level
    IE::buildAdjustPrecisionPipeline(pm, log);

    pm.addPass(IE::createConvertFCToConvPass(log));
    pm.addPass(IE::createConvertAvgPoolToDWConvPass(log));
    pm.addPass(IE::createConvertScaleShiftToDWPass(log));
    // Canonicalize group convolution if necessary.
    pm.addPass(mlir::createCanonicalizerPass(grc));
    IE::buildAdjustForVPUPipeline(pm, log);
    pm.addPass(IE::createHandleLargeStridesPass(log));
    pm.addPass(IE::createHandleAsymmetricStridesPass(log));
    IE::buildLowPrecisionPipeline(pm, log);

    pm.addPass(IE::createUpstreamSlicePass(log));

    pm.addPass(IE::createExpandActivationChannelsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    buildIECommonPipeline(pm, log);

    pm.addPass(IE::createIsolatedTilingPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    // Lower IE->IERT
    buildLowerIE2IERTPipeline(pm, log);

    if (enableProfiling) {
        pm.addPass(IERT::createTimestampProfilingPass(getMemSpace<VPUIP::PhysicalMemory::CMX_NN>, log));
    }

    // Partially lower IERT->VPUIP (NCE Operations only)
    pm.addPass(createConvertToNCEOpsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    // Partially lower IERT->VPUIP (Act shave tasks only)
    pm.addPass(createConvertSWLayers2VPUIPPass(log));

    // IERT Dialect level (cont.)
    pm.addPass(IERT::createSetInternalMemorySpacePass(getMemSpace<VPUIP::PhysicalMemory::DDR>, log));
    pm.addPass(IERT::createOptimizeCopiesPass(log));
    pm.addPass(IERT::createCopyOpHoistingPass(log));
    IERT::buildAsyncSchedulingPipeline(pm, log);
    pm.addPass(IERT::createFeasibleAllocationPass(getMemSpace<VPUIP::PhysicalMemory::CMX_NN>,
                                                  getMemSpace<VPUIP::PhysicalMemory::DDR>, log));
    pm.addPass(IERT::createGroupAsyncExecuteOpsPass(log));
    pm.addPass(IERT::createStaticAllocationPass(getMemSpace<VPUIP::PhysicalMemory::DDR>, log));
    pm.addPass(IERT::createOptimizeAsyncDepsPass(log));

    // Handle WeightsTable, which requires statically allocated memory
    pm.addPass(VPUIP::createConvertWeightsTableOp2ConstPass(log));

    // Finally lower remaining IERT->VPUIP (SW mode)
    buildLowerIERT2VPUIPPipeline(pm, log);

    // VPUIP Dialect level
    pm.addPass(VPUIP::createAssignPhysicalBarriersPass(log));
    pm.addPass(VPUIP::createBarrierSimulationPass(log));
    pm.addPass(VPUIP::createDumpStatisticsOfTaskOpsPass(log));
}

//
// DefaultHWMode
//

void vpux::buildDefaultHWModePipeline(mlir::OpPassManager& pm, bool enableProfiling, Logger log,
                                      StringRef pipelineConfig) {
    auto pipelineOptions = MyPipelineOptions::createFromString(pipelineConfig);
    const auto grc = getDefaultGreedyRewriteConfig();
    pm.addPass(mlir::createCanonicalizerPass(grc));

    // IE Dialect level
    IE::buildAdjustPrecisionPipeline(pm, log);

    if (pipelineOptions->enableConvertFCToConv.getValue())
        pm.addPass(IE::createConvertFCToConvPass(log));
    if (pipelineOptions->enableConvertAvgPoolToDWConv.getValue())
        pm.addPass(IE::createConvertAvgPoolToDWConvPass(log));
    if (pipelineOptions->enableConvertScaleShiftDW.getValue())
        pm.addPass(IE::createConvertScaleShiftToDWPass(log));
    // Canonicalize group convolution if necessary.
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::buildAdjustForVPUPipeline(pm, log);
    pm.addPass(IE::createHandleLargeStridesPass(log));
    if (pipelineOptions->enableHandleAsymetricStrides.getValue())
        pm.addPass(IE::createHandleAsymmetricStridesPass(log));
    if (pipelineOptions->enableLowPrecisionBuilding.getValue())
        IE::buildLowPrecisionPipeline(pm, log);

    pm.addPass(IE::createUpstreamSlicePass(log));

    if (pipelineOptions->enableExpandActivationChannels.getValue()) {
        pm.addPass(IE::createExpandActivationChannelsPass(log));
        pm.addPass(mlir::createCanonicalizerPass(grc));
    }

    buildIECommonPipeline(pm, log, *pipelineOptions);

    pm.addPass(IE::createIsolatedTilingPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    // Lower IE->IERT
    buildLowerIE2IERTPipeline(pm, log);

    if (enableProfiling) {
        pm.addPass(IERT::createTimestampProfilingPass(getMemSpace<VPUIP::PhysicalMemory::CMX_NN>, log));
    }

    // Partially lower IERT->VPUIP (NCE Operations only)
    pm.addPass(createConvertToNCEOpsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    // Partially lower IERT->VPUIP (Act shave tasks only)
    pm.addPass(createConvertSWLayers2VPUIPPass(log));

    // IERT Dialect level (cont.)
    pm.addPass(IERT::createSetInternalMemorySpacePass(getMemSpace<VPUIP::PhysicalMemory::DDR>, log));
    if (pipelineOptions->enableOptimizeCopies.getValue()) {
        pm.addPass(IERT::createOptimizeCopiesPass(log));
        pm.addPass(IERT::createCopyOpHoistingPass(log));
    }
    IERT::buildAsyncSchedulingPipeline(pm, log);
    pm.addPass(IERT::createFeasibleAllocationPass(getMemSpace<VPUIP::PhysicalMemory::CMX_NN>,
                                                  getMemSpace<VPUIP::PhysicalMemory::DDR>, log));
    if (pipelineOptions->enableGroupAsyncExecuteOps.getValue())
        pm.addPass(IERT::createGroupAsyncExecuteOpsPass(log));
    pm.addPass(IERT::createStaticAllocationPass(getMemSpace<VPUIP::PhysicalMemory::DDR>, log));
    if (pipelineOptions->enableOptimizeAsyncDeps.getValue())
        pm.addPass(IERT::createOptimizeAsyncDepsPass(log));

    // Handle WeightsTable, which requires statically allocated memory
    pm.addPass(VPUIP::createConvertWeightsTableOp2ConstPass(log));

    // Finally lower remaining IERT->VPUIP (SW mode)
    buildLowerIERT2VPUIPPipeline(pm, log);

    // VPUIP Dialect level
    pm.addPass(VPUIP::createAssignPhysicalBarriersPass(log));
    pm.addPass(VPUIP::createBarrierSimulationPass(log));
    pm.addPass(VPUIP::createDumpStatisticsOfTaskOpsPass(log));
}

//
// registerPipelines
//

void vpux::registerPipelines() {
    mlir::PassPipelineRegistration<MyPipelineOptions>(
            "reference-sw-mode", "Compile IE Network in Reference Software mode (SW only execution)",
            [](mlir::OpPassManager& pm, const MyPipelineOptions& config) {
                addConfigPass(pm, config, VPUIP::CompilationMode::ReferenceSW);
                buildReferenceSWModePipeline(pm);
            });
    mlir::PassPipelineRegistration<MyPipelineOptions>(
            "reference-hw-mode", "Compile IE Network in Reference Hardware mode (HW and SW execution)",
            [](mlir::OpPassManager& pm, const MyPipelineOptions& config) {
                addConfigPass(pm, config, VPUIP::CompilationMode::ReferenceHW);
                buildReferenceHWModePipeline(pm);
            });
    mlir::PassPipelineRegistration<MyPipelineOptions>(
            "default-hw-mode", "Compile IE Network in Default Hardware mode (HW and SW execution)",
            [](mlir::OpPassManager& pm, const MyPipelineOptions& config) {
                addConfigPass(pm, config, VPUIP::CompilationMode::DefaultHW);
                buildDefaultHWModePipeline(pm);
            });
}
