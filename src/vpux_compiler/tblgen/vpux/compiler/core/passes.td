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

#ifndef VPUX_COMPILER_CORE_PASSES
#define VPUX_COMPILER_CORE_PASSES

include "mlir/Pass/PassBase.td"

//
// ComposeSubView
//

def ComposeSubView : PassBase<"compose-subview", "vpux::FunctionPass"> {
    let summary = "Combine composed subview ops";

    let constructor = "vpux::createComposeSubViewPass()";

    let dependentDialects = [
        "mlir::memref::MemRefDialect"
    ];
}

//
// DeallocPlacement
//

def DeallocPlacement : PassBase<"dealloc-placement", "vpux::FunctionPass"> {
    let summary = "Insert dealloc operations for dynamically allocated memory buffers";

    let description = [{
        This pass supports multi-view Operations in contrast to the MLIR standard pass.
    }];

    let constructor = "vpux::createDeallocPlacementPass()";

    let dependentDialects = [
        "mlir::memref::MemRefDialect"
    ];
}

#endif
