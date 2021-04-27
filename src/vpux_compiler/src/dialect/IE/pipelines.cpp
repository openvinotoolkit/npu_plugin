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

#include "vpux/compiler/dialect/IE/passes.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// AdjustForVPU
//

void vpux::IE::buildAdjustForVPUPipeline(mlir::OpPassManager& pm, Logger log) {
    pm.addPass(IE::createConvertFCToConvPass(log));
    pm.addPass(IE::createConvertTile2PerAxisTilePass(log));
    pm.addPass(IE::createConvertPrecisionToFP16Pass(log));
    pm.addPass(IE::createConvertShapeTo4DPass(log));
    pm.addPass(IE::createConvertPaddingsToFloorModePass(log));
    pm.addPass(mlir::createCanonicalizerPass());
}

//
// LowPrecision
//

void vpux::IE::buildLowPrecisionPipeline(mlir::OpPassManager& pm, Logger log) {
    pm.addPass(IE::createSplitFakeQuantPass(log));
    pm.addPass(IE::createQuantizeConstPass(log));
    // TODO: insert advanced LPT pipeline here
    pm.addPass(IE::createDequantizeConstPass(log));
    pm.addPass(IE::createMergeFakeQuantPass(log));
}

//
// registerIEPipelines
//

void vpux::IE::registerIEPipelines() {
    mlir::PassPipelineRegistration<>("adjust-for-vpu", "Adjust IE Dialect IR for VPU target",
                                     [](mlir::OpPassManager& pm) {
                                         IE::buildAdjustForVPUPipeline(pm);
                                     });

    mlir::PassPipelineRegistration<>("low-precision", "Low precision transformations", [](mlir::OpPassManager& pm) {
        IE::buildLowPrecisionPipeline(pm);
    });
}
