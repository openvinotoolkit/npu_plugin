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

#ifndef VPUX_COMPILER_DIALECT_VPUIP_PASSES
#define VPUX_COMPILER_DIALECT_VPUIP_PASSES

include "mlir/Pass/PassBase.td"

//
// SetCompileParams
//

def SetCompileParams : PassBase<"set-compile-params", "vpux::ModulePass"> {
    let summary = "Set compilation parameters related to VPUIP Dialect";

    let description = [{
This pass attaches compilation parameters related to **VPUIP Dialect** to Module attributes and
initializes **IERT Dialect** run-time resources information.
    }];

    let constructor = "vpux::VPUIP::createSetCompileParamsPass(vpux::None)";

    let options = [
        Option<
            "archName", "vpu-arch",
            "std::string", [{"VPU3400_A0"}],
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

def AddLinearScheduling : PassBase<"add-linear-scheduling", "vpux::ModulePass"> {
    let summary = "Impose simple linear execution";

    let description = [{
This pass inserts barriers to impose simple linear execution in IR order.
    }];

    let constructor = "vpux::VPUIP::createAddLinearSchedulingPass()";
}

#endif
