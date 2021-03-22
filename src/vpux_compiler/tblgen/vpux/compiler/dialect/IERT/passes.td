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

#ifndef VPUX_COMPILER_DIALECT_IERT_PASSES
#define VPUX_COMPILER_DIALECT_IERT_PASSES

include "mlir/Pass/PassBase.td"

//
// AddLayoutsAndStrides
//

def AddLayoutsAndStrides : Pass<"add-layouts-and-strides", "mlir::ModuleOp"> {
    let summary = "Add layouts and strides to memref";

    let description = [{
        This pass adds default values for layouts and strides to the memref
    }];

    let constructor = "vpux::IERT::createAddLayoutsAndStridesPass()";
}

//
// SetInternalMemorySpace
//

def SetInternalMemorySpace : FunctionPass<"set-internal-memory-space"> {
    let summary = "Set specific memory space for all internal memory buffers";

    let description = [{
        This pass updates all Types for internal memory buffers and sets the specified memory space for them.
    }];

    let constructor = [{
        vpux::IERT::createSetInternalMemorySpacePass([](mlir::MLIRContext* ctx, vpux::StringRef memSpaceName) -> mlir::Attribute {
            VPUX_THROW_UNLESS(!memSpaceName.empty(), "Missing memory space option");
            return mlir::StringAttr::get(ctx, memSpaceName);
        })
    }];

    let options = [
        Option<
            "memSpaceName", "memory-space",
            "std::string", [{""}],
            "Memory space to perform allocation"
        >
    ];

    let dependentDialects = [
        "mlir::StandardOpsDialect",
        "mlir::linalg::LinalgDialect"
    ];
}

//
// StaticAllocation
//

def StaticAllocation : Pass<"static-allocation", "mlir::ModuleOp"> {
    let summary = "Replace dynamic allocations with static";

    let description = [{
        This pass replaces all dynamic `alloc`/`dealloc` Operations with `IERT.StaticAlloc`.
        It uses simple LinearScan algorithm.
    }];

    let constructor = [{
        vpux::IERT::createStaticAllocationPass([](mlir::MLIRContext* ctx, vpux::StringRef memSpaceName) -> mlir::Attribute {
            VPUX_THROW_UNLESS(!memSpaceName.empty(), "Missing memory space option");
            return mlir::StringAttr::get(ctx, memSpaceName);
        })
    }];

    let options = [
        Option<
            "memSpaceName", "memory-space",
            "std::string", [{""}],
            "Memory space to perform allocation"
        >
    ];
}

#endif
