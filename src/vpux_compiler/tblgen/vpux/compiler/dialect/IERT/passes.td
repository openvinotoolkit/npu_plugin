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
// OptimizeCopies
//

def OptimizeCopies : PassBase<"optimize-copies", "vpux::FunctionPass"> {
    let summary = "Removes Copy Ops which are unnecessary";

    let description = [{
        This pass checks if Copy Op can be optimized out to reduce the amount of unnecessary DMAs and intermediate buffers.
    }];

    let constructor = "vpux::IERT::createOptimizeCopiesPass()";
}

//
// CopyOpHoisting
//

def CopyOpHoisting : PassBase<"copy-op-hoisting", "vpux::FunctionPass"> {
    let summary = "Update location of CopyOps in the block";

    let description = [{
        This pass checks all CopyOps and updates their position in the parent block to be close to source op
        that generates the copy op source buffer
    }];

    let constructor = "vpux::IERT::createCopyOpHoistingPass()";
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

def CMXTiling : PassBase<"cmx-tiling", "vpux::FunctionPass"> {
    let summary = "Tile Operations to the condition that all their I/O fit into CMX";

    let description = [{
        This pass will replace a set of operations with the pattern OP ==> Subview->Copy->Op->Copy->SubView.
        The condition for the replacment is that for each op the SUM of their I/O tensors to fit into CMX memory
    }];

    let constructor = "vpux::IERT::createCMXTilingPass()";
}

//
// TimestampProfiling
//

def TimestampProfiling : PassBase<"profiling-timestamp", "vpux::ModulePass"> {
    let summary = "DMA-Timestamp based network profiling";

    let description = [{
        This pass add dma-timestamp based network profiling.
    }];

    let constructor = [{
        vpux::IERT::createTimestampProfilingPass([](mlir::MLIRContext* ctx, vpux::StringRef memSpaceName) -> mlir::Attribute {
            VPUX_THROW_UNLESS(!memSpaceName.empty(), "Missing memory space option");
            return mlir::StringAttr::get(ctx, memSpaceName);
        })
    }];
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

//
// MoveWaitResultToAsyncBlockArgs
//

def MoveWaitResultToAsyncBlockArgs : PassBase<"move-wait-result-to-async-block-args", "vpux::FunctionPass"> {
    let summary = "Moves 'async.await' result usage from 'async.execute' body to it's operands";

    let constructor = "vpux::IERT::createMoveWaitResultToAsyncBlockArgsPass()";
}

//
// MoveViewOpsIntoAsyncRegions
//

def MoveViewOpsIntoAsyncRegions : PassBase<"move-view-ops-into-async-regions", "vpux::FunctionPass"> {
    let summary = "Moves view-like Operations inside the asynchronous regions which depends on them";

    let constructor = "vpux::IERT::createMoveViewOpsIntoAsyncRegionsPass()";
}

//
// OptimizeAsyncDeps
//

def OptimizeAsyncDeps : PassBase<"optimize-async-deps", "vpux::FunctionPass"> {
    let summary = "Optimizes dependencies between 'async.execute' operations";

    let description = [{
        The pass tries to remove extra explicit `!async.token` based dependencies,
        if they are represented implicitly (as a result of transitive dependencies).
    }];

    let constructor = "vpux::IERT::createOptimizeAsyncDepsPass()";
}

//
// GroupAsyncExecuteOps
//

def GroupAsyncExecuteOps : PassBase<"group-async-execute-ops", "vpux::FunctionPass"> {
    let summary = "Reduces number of async.execute operations";

    let description = [{
        Groups consecutive operations which utilizes the same executor and max resources into same async.execute region
    }];

    let constructor = "vpux::IERT::createGroupAsyncExecuteOpsPass()";
}

//
// FeasibleAllocation
//

def FeasibleAllocation : PassBase<"feasible-allocation", "vpux::ModulePass"> {
    let summary = "Feasible Memory Scheduling Pass";

    let description = [{
        Schedule async.execute opeations based on their dependecies and CMX memory availability
    }];

    let constructor = [{
        vpux::IERT::createFeasibleAllocationPass([](mlir::MLIRContext* ctx, vpux::StringRef memSpaceName) -> mlir::Attribute {
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

#endif
