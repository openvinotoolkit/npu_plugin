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

#include "vpux/compiler/pipelines.hpp"

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/IERT/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// ReferenceMode
//

void vpux::buildReferenceModePipeline(mlir::OpPassManager& pm, Logger log) {
    const auto ddrMemSpaceCb = [](mlir::MLIRContext* ctx, StringRef) -> mlir::Attribute {
        return VPUIP::PhysicalMemoryAttr::get(ctx, VPUIP::PhysicalMemory::DDR);
    };

    // IE Dialect level
    pm.addPass(mlir::createCanonicalizerPass());
    IE::buildAdjustForVPUPipeline(pm, log);
    pm.addPass(IE::createUseUserPrecisionPass(log));
    pm.addPass(mlir::createCanonicalizerPass());
    IE::buildLowPrecisionPipeline(pm, log);

    // Lower IE->IERT
    buildLowerIE2IERTPipeline(pm, log);

    // IERT Dialect level
    pm.addPass(IERT::createComposeSubViewPass(log));
    pm.addPass(mlir::createBufferDeallocationPass());
    pm.addPass(IERT::createSetInternalMemorySpacePass(ddrMemSpaceCb, log));
    pm.addPass(IERT::createStaticAllocationPass(ddrMemSpaceCb, log));

    // Lower IERT->VPUIP
    pm.addPass(createLowerIERT2VPUIPPass(log));

    // VPUIP Dialect level
    pm.addPass(VPUIP::createAddLinearSchedulingPass(log));
}

//
// registerPipelines
//

void vpux::registerPipelines() {
    mlir::PassPipelineRegistration<>("reference-mode", "Compile IE Network in Reference mode (SW only execution)",
                                     [](mlir::OpPassManager& pm) {
                                         buildReferenceModePipeline(pm);
                                     });
}
