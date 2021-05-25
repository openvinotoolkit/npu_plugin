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

    let constructor = "vpux::VPUIP::createSetCompileParamsPass()";

    let options = [
        Option<
            "archOpt", "vpu-arch",
            "std::string", [{"VPU3700"}],
            "VPU architecture to compile for"
        >,
        Option<
            "compilationModeOpt", "compilation-mode",
            "std::string", [{"ReferenceSW"}],
            "Set compilation mode as reference (ReferenceSW) or hardware (ReferenceHW)"
        >
    ];

    let dependentDialects = [
        "vpux::IERT::IERTDialect",
        "vpux::VPUIP::VPUIPDialect"
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
