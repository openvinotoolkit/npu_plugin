//
// Copyright 2021 Intel Corporation.
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

#include "vpux/compiler/edsl/pipeline.hpp"

#include <memory>
#include <string>
#include <unordered_map>

#include <llvm/Support/FormatVariadic.h>
#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h>
#include <mlir/Dialect/Affine/Passes.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Vector/VectorOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#ifdef ENABLE_PLAIDML
#include "pmlc/compiler/registry.h"
#include "pmlc/conversion/pxa_to_affine/passes.h"
#include "pmlc/conversion/tile_to_pxa/passes.h"
#include "pmlc/dialect/pxa/transforms/passes.h"
#include "pmlc/dialect/tile/transforms/passes.h"
#endif

#include "vpux/compiler/edsl/emit_c.hpp"
#include "vpux/compiler/edsl/passes.hpp"

using namespace mlir;  // NOLINT

namespace vpux {

namespace edsl {

#ifdef ENABLE_PLAIDML
namespace pxa = pmlc::dialect::pxa;
namespace tile = pmlc::dialect::tile;
#endif

void pipelineBuilder(OpPassManager& pm) {
#ifdef ENABLE_PLAIDML
    pm.addNestedPass<FuncOp>(tile::createExpandReshapePass());
    pm.addNestedPass<FuncOp>(tile::createComputeBoundsPass());
    //  This pass is not necessary and buggy currently. We could enable it
    //  after when it is fixed.
    //  pm.addPass(tile::createPadRangesPass());
    pm.addNestedPass<FuncOp>(tile::createPadConstraintsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    pm.addPass(pmlc::conversion::tile_to_pxa::createLowerTileToPXAPass());
    pm.addNestedPass<FuncOp>(pxa::createAffineNormalizePass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    pm.addNestedPass<FuncOp>(pxa::createFusionPass(
            /*memoryActivityThreshold=*/50 * 1024,
            /*exactlyMatch=*/true,
            /*tiledFusion=*/false,
            /*loopDepth=*/0,
            /*singleOutput=*/true));
    pm.addNestedPass<FuncOp>(pxa::createAffineNormalizePass());
    pm.addPass(createCanonicalizerPass());
    pm.addNestedPass<FuncOp>(pxa::createFusionPass(
            /*memoryActivityThreshold=*/50 * 1024,
            /*exactlyMatch=*/false,
            /*tiledFusion=*/false,
            /*loopDepth=*/0,
            /*singleOutput=*/true));
    pm.addNestedPass<FuncOp>(pxa::createAffineNormalizePass());
    pm.addPass(createCanonicalizerPass());

    pm.addNestedPass<FuncOp>(pxa::createMemRefDataFlowOptPass());
    pm.addNestedPass<FuncOp>(pxa::createAffineNormalizePass());
    pm.addPass(createCanonicalizerPass());
    pm.addNestedPass<FuncOp>(pxa::createLocalizePass());
    pm.addNestedPass<FuncOp>(pxa::createResizeTmpsPass());

    pm.addNestedPass<FuncOp>(pxa::createAffineNormalizePass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    pm.addNestedPass<FuncOp>(createSinkScalarPass());
    pm.addPass(createShavePipelinePass());
    pm.addNestedPass<FuncOp>(pxa::createAffineNormalizePass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    OpPassManager& pmModule = pm.nest<ModuleOp>();
    pmModule.addNestedPass<FuncOp>(pxa::createVectorizePass(/*strategy=*/"recursive"));
    pmModule.addNestedPass<FuncOp>(pxa::createAffineNormalizePass());
    pmModule.addPass(createCanonicalizerPass());
    pmModule.addPass(createCSEPass());
    pmModule.addPass(pmlc::conversion::pxa_to_affine::createLowerPXAToAffinePass());
    pmModule.addPass(createCanonicalizerPass());
    pmModule.addPass(createCSEPass());

    // The UnrollLoop optimizer must be run between LowerPXAToAffinePass
    // and LowerAffinePass passes.
    pmModule.addNestedPass<FuncOp>(createLoopUnrollPass(
            /*unrollFactor=*/6,
            /*unrollUpToFactor=*/true));

    pmModule.addPass(createLowerAffinePass());
    pmModule.addPass(createCanonicalizerPass());
    pmModule.addPass(createCSEPass());

    pmModule.addNestedPass<FuncOp>(createShavePatternsPass());
    pmModule.addPass(createLoopInvariantCodeMotionPass());
#else
    // To avoid the warning of unused variable
    (void)pm;
#endif
}

static constexpr const char* kTargetName = "vpux";
static constexpr const char* kPassPipelineTargetName = "lower-TILE-to-VPUIP";

void registerPipeline() {
    PassPipelineRegistration<> passPipelineReg(kPassPipelineTargetName, "VPUX eDSL SHAVE pipeline",
                                               vpux::edsl::pipelineBuilder);
}

#ifdef ENABLE_PLAIDML
class Target : public pmlc::compiler::Target {
public:
    void buildPipeline(OpPassManager& pm, StringRef) {
        vpux::edsl::pipelineBuilder(pm);
    }

    pmlc::util::BufferPtr save(pmlc::compiler::Program&, const std::unordered_map<std::string, std::string>&) {
        throw std::runtime_error(llvm::formatv("Target '{0}' does not have 'save' support.", kTargetName).str());
    }
};

void registerTarget() {
    pmlc::compiler::registerTarget(kTargetName, std::make_shared<Target>());
}
#endif

}  // namespace edsl
}  // namespace vpux
