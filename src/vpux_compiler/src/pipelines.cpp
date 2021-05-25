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
#include "vpux/compiler/core/passes.hpp"
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
    pm.addPass(createComposeSubViewPass(log));
    pm.addPass(IERT::createUseUserLayout(log));
    pm.addPass(IERT::createAdjustLayoutsPass(log));
    pm.addPass(createDeallocPlacementPass(log));
    pm.addPass(IERT::createSetInternalMemorySpacePass(ddrMemSpaceCb, log));
    pm.addPass(IERT::createStaticAllocationPass(ddrMemSpaceCb, log));

    // Lower remaining IERT->VPUIP
    buildLowerIERT2VPUIPPipeline(pm, log);

    // VPUIP Dialect level
    pm.addPass(VPUIP::createAddLinearSchedulingPass(log));
}

//
// HardwareMode
//

void vpux::buildHardwareModePipeline(mlir::OpPassManager& pm, Logger log) {
    const auto ddrMemSpaceCb = [](mlir::MLIRContext* ctx, StringRef) -> mlir::Attribute {
        return VPUIP::PhysicalMemoryAttr::get(ctx, VPUIP::PhysicalMemory::DDR);
    };

    const auto cmxMemSpaceCb = [](mlir::MLIRContext* ctx, StringRef) -> mlir::Attribute {
        return VPUIP::PhysicalMemoryAttr::get(ctx, VPUIP::PhysicalMemory::CMX_NN);
    };

    // IE Dialect level
    pm.addPass(mlir::createCanonicalizerPass());
    IE::buildAdjustForVPUPipeline(pm, log);
    pm.addPass(IE::createUseUserPrecisionPass(log));
    pm.addPass(mlir::createCanonicalizerPass());
    IE::buildLowPrecisionPipeline(pm, log);

    // Lower IE->IERT
    buildLowerIE2IERTPipeline(pm, log);

    // Partially lower IERT->VPUIP (NCE Operations only)
    pm.addPass(createConvertToNCEOpsPass());

    // IERT Dialect level
    pm.addPass(createFuseActivationsPass());
    pm.addPass(createComposeSubViewPass(log));
    pm.addPass(createDeallocPlacementPass(log));
    pm.addPass(IERT::createSetInternalMemorySpacePass(ddrMemSpaceCb, log));
    pm.addPass(IERT::createStaticAllocationPass(ddrMemSpaceCb, log));
    pm.addPass(IERT::createStaticAllocationPass(cmxMemSpaceCb, log));

    // Finally lower remaining IERT->VPUIP
    buildLowerIERT2VPUIPPipeline(pm, log);

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

    mlir::PassPipelineRegistration<>("hardware-mode", "Compile IE Network in Hardware mode (HW and SW execution)",
                                     [](mlir::OpPassManager& pm) {
                                         buildHardwareModePipeline(pm);
                                     });
}
