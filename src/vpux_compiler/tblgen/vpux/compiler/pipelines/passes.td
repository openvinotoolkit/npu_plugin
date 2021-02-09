//
// Copyright 2020 Intel Corporation.
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

#ifndef VPUX_COMPILER_PIPELINES_PASSES
#define VPUX_COMPILER_PIPELINES_PASSES

include "mlir/Pass/PassBase.td"

//
// ReferenceMode
//

def ReferenceMode : Pass<"reference-mode", "mlir::ModuleOp"> {
    let summary = "Compiles IE Network in Reference mode (SW only execution)";

    let description = [{
        This "pipeline" pass performs full IR lowering from IE Dialect to VPUIP Dialect (using intermediate lowerings).
        It uses simple SW implementation for layer operations without any optimizations.
    }];

    let constructor = "vpux::createReferenceModePass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

#endif
