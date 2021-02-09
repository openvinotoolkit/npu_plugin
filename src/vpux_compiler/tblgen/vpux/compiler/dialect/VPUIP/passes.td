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

#ifndef VPUX_COMPILER_DIALECT_VPUIP_PASSES
#define VPUX_COMPILER_DIALECT_VPUIP_PASSES

include "mlir/Pass/PassBase.td"

//
// SetCompileParams
//

def SetCompileParams : Pass<"set-compile-params", "mlir::ModuleOp"> {
    let summary = "Set compilation parameters related to VPUIP Dialect";

    let description = [{
        This "atomic" pass attaches compilation parameters related to VPUIP Dialect
        to Module attributes and initializes IERT Dialect resources information.
    }];

    let constructor = "vpux::VPUIP::createSetCompileParamsPass(vpux::None)";

    let options = [
        Option<
            "archName", "vpu-arch",
            "std::string", [{"MA2490"}],
            "VPU architecture to compile for"
        >
    ];

    let dependentDialects = [
        "vpux::IERT::IERTDialect"
    ];
}

//
// AddLinearScheduling
//

def AddLinearScheduling : Pass<"add-linear-scheduling", "mlir::ModuleOp"> {
    let summary = "Impose simple linear execution";

    let description = [{
        The pass inserts barriers to impose simple linear execution in IR order.
    }];

    let constructor = "vpux::VPUIP::createAddLinearSchedulingPass()";
}

#endif
