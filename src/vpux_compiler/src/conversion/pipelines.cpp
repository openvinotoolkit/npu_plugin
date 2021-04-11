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

#include <mlir/Dialect/StandardOps/Transforms/Passes.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// LowerIE2IERT
//

void vpux::buildLowerIE2IERTPipeline(mlir::OpPassManager& pm, Logger log) {
    pm.addPass(createBufferizeIEPass(log));
    pm.addPass(mlir::createFuncBufferizePass());
    pm.addPass(vpux::createAddBuffersForNetResults());
    pm.addPass(mlir::createFinalizingBufferizePass());
}

//
// registerConversionPipelines
//

void vpux::registerConversionPipelines() {
    mlir::PassPipelineRegistration<>("lower-IE-to-IERT", "Performs full lowering from the IE Dialect to IERT Dialect",
                                     [](mlir::OpPassManager& pm) {
                                         buildLowerIE2IERTPipeline(pm);
                                     });
}
