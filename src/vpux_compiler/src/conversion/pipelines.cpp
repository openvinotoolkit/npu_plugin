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

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/dialect/IERT/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"

#include <mlir/Dialect/StandardOps/Transforms/Passes.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// LowerIE2IERT
//

void vpux::buildLowerIE2IERTPipeline(mlir::OpPassManager& pm, Logger log) {
    pm.addPass(createConvertIE2IERTPass(log));
    pm.addPass(mlir::createFuncBufferizePass());
    pm.addPass(mlir::createBufferResultsToOutParamsPass());
    pm.addPass(mlir::createFinalizingBufferizePass());
    pm.addPass(mlir::createBufferDeallocationPass());
    pm.addPass(mlir::createCopyRemovalPass());
}

//
// LowerIERT2VPUIP
//

void vpux::buildLowerIERT2VPUIPPipeline(mlir::OpPassManager& pm, Logger log) {
    const auto memSpaceCb = [](mlir::MLIRContext* ctx, StringRef) -> mlir::Attribute {
        return VPUIP::PhysicalMemoryAttr::get(ctx, VPUIP::PhysicalMemory::DDR);
    };

    pm.addPass(IERT::createSetInternalMemorySpacePass(memSpaceCb, log));
    pm.addPass(IERT::createStaticAllocationPass(memSpaceCb, log));
    pm.addPass(createConvertIERT2VPUIPPass(log));
}

//
// registerConversionPipelines
//

void vpux::registerConversionPipelines() {
    mlir::PassPipelineRegistration<>("lower-IE-to-IERT", "Performs full lowering from the IE Dialect to IERT Dialect",
                                     [](mlir::OpPassManager& pm) {
                                         buildLowerIE2IERTPipeline(pm);
                                     });

    mlir::PassPipelineRegistration<>("lower-IERT-to-VPUIP", "Performs full lowering from IERT Dialect to VPUIP Dialect",
                                     [](mlir::OpPassManager& pm) {
                                         buildLowerIERT2VPUIPPipeline(pm);
                                     });
}
