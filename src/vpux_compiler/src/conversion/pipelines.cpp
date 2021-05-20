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

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/dialect/IERT/passes.hpp"

#include <mlir/Dialect/StandardOps/Transforms/Passes.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// LowerIE2IERT
//

void vpux::buildLowerIE2IERTPipeline(mlir::OpPassManager& pm, Logger log) {
    pm.addPass(createBufferizeIEPass(log));
    pm.addPass(mlir::createFuncBufferizePass());
    pm.addPass(mlir::createFinalizingBufferizePass());
    pm.addPass(createAddBuffersForNetResults(log));
    pm.addPass(mlir::createCanonicalizerPass());
}

//
// LowerIERT2VPUIP
//

void vpux::buildLowerIERT2VPUIPPipeline(mlir::OpPassManager& pm, Logger log) {
    pm.addPass(createConvertIERTOps2VPUIPPass(log));
}

//
// registerConversionPipelines
//

void vpux::registerConversionPipelines() {
    mlir::PassPipelineRegistration<>("lower-IE-to-IERT", "Performs full lowering from the IE Dialect to IERT Dialect",
                                     [](mlir::OpPassManager& pm) {
                                         buildLowerIE2IERTPipeline(pm);
                                     });

    mlir::PassPipelineRegistration<>("lower-IERT-to-VPUIP",
                                     "Performs full lowering from the IERT Dialect to VPUIP Dialect",
                                     [](mlir::OpPassManager& pm) {
                                         buildLowerIERT2VPUIPPipeline(pm);
                                     });
}
