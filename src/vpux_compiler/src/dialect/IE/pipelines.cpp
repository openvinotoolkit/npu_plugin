//
// Copyright Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/IE/passes.hpp"

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
// registerPipelines
//

void vpux::IE::registerPipelines() {
    mlir::PassPipelineRegistration<>("adjust-for-vpu", "Adjust IE Dialect IR for VPU target",
                                     [](mlir::OpPassManager& pm) {
                                         IE::buildAdjustForVPUPipeline(pm);
                                     });

    mlir::PassPipelineRegistration<>("low-precision", "Low precision transformations", [](mlir::OpPassManager& pm) {
        IE::buildLowPrecisionPipeline(pm);
    });
}
