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
