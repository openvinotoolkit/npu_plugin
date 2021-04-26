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

#ifndef VPUX_COMPILER_DIALECT_IERT_PASSES
#define VPUX_COMPILER_DIALECT_IERT_PASSES

include "mlir/Pass/PassBase.td"

//
// AdjustLayouts
//

def AdjustLayouts : PassBase<"adjust-layouts", "vpux::FunctionPass"> {
    let summary = "Add required layouts to memref";

    let description = [{
        This pass adds the required layouts instead of the default value
        depending on the layer specification
    }];

    let constructor = "vpux::IERT::createAdjustLayoutsPass()";
}

//
// UseUserLayout
//

def UseUserLayout : PassBase<"use-user-layout", "vpux::ModulePass"> {
    let summary = "Use user layouts for entry point function prototype";

    let description = [{
        This pass updates the CNNNetwork entry point function prototype
        and use user-provided layouts for its operands and results.
        The pass inserts Reorder operations from/to topology layout.
    }];

    let constructor = "vpux::IERT::createUseUserLayout()";
}


//
// SetInternalMemorySpace
//

def SetInternalMemorySpace : PassBase<"set-internal-memory-space", "vpux::FunctionPass"> {
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
}

//
// StaticAllocation
//

def StaticAllocation : PassBase<"static-allocation", "vpux::ModulePass"> {
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

    let dependentDialects = [
        "vpux::IERT::IERTDialect"
    ];
}

//
// CMX Tiling
//

def CXMTiling : PassBase<"cmx-tiling", "vpux::FunctionPass"> {
    let summary = "Tile Operations to the condition that all their I/O fit into CMX";

    let description = [{
        This pass will replace a set of operations with the pattern OP ==> Subview->Copy->Op->Copy->SubView.
        The condition for the replacment is that for each op the SUM of their I/O tensors to fit into CMX memory
    }];

    let constructor = "vpux::IERT::createCMXTilingPass()";
}

//=================================================================================
// Asynchronous Scheduling
//=================================================================================

//
// WrapIntoAsyncRegions
//

def WrapIntoAsyncRegions : PassBase<"wrap-into-async-regions", "vpux::FunctionPass"> {
    let summary = "Wraps layer operations into asynchronous regions";

    let description = [{
        This pass wraps each IERT layer operation into async region preserving linear execution.
    }];

    let constructor = "vpux::IERT::createWrapIntoAsyncRegionsPass()";

    let dependentDialects = [
        "mlir::async::AsyncDialect"
    ];
}

#endif
