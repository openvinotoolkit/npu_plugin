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
// OptimizeParallelCopies
//

def OptimizeParallelCopies : PassBase<"optimize-parallel-copies", "vpux::FunctionPass"> {
    let summary = "Copy the data only once for all the tiles that share the same data";
    
    let description = [{
    This pass checks all the CopyOps consumed by tiles of one tiling subgraph.
    If the CopyOps operate on the same weight or activation, merge the parallel copies into one.
    }];
    
    let constructor = "vpux::IERT::createOptimizeParallelCopiesPass()";
}

//
// CopyOpLegalization
//

def CopyOpLegalization : PassBase<"legalize-copies", "vpux::FunctionPass"> {
    let summary = "Legalizes Copy Ops which do not fit hardware capabilities";

    let description = [{
        This pass checks if Copy Op can be executed at target hardware and splits it into a few tiles if necessary.
    }];

    let constructor = "vpux::IERT::createCopyOpLegalizationPass()";
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
        vpux::IERT::createSetInternalMemorySpacePass([](mlir::MLIRContext* ctx, vpux::StringRef memSpaceName) {
            VPUX_THROW_UNLESS(!memSpaceName.empty(), "Missing memory space option");
            return vpux::IndexedSymbolAttr::get(ctx, memSpaceName);
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
        vpux::IERT::createStaticAllocationPass([](mlir::MLIRContext* ctx, vpux::StringRef memSpaceName) {
            VPUX_THROW_UNLESS(!memSpaceName.empty(), "Missing memory space option");
            return vpux::IndexedSymbolAttr::get(ctx, memSpaceName);
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
// Linearization
//

def Linearization : PassBase<"linearization", "vpux::ModulePass"> {
    let summary = "Perform linearization of the IR";

    let description = [{
        Perform linearization of the IR with fully sequential execution.
    }];

    let constructor = "vpux::IERT::createLinearizationPass()";
}

//
// BreakDataFlow
//

def BreakDataFlow : PassBase<"break-data-flow", "vpux::FunctionPass"> {
    let summary = "Breaks data flow in the graph";

    let description = [{
        This pass breaks data flow in the graph. It is required for the VPURT dialect for correct task creation
        because all VPUIP dialect tasks will be inside body of the TaskOp and it is impossible to use operation results inside another body of TaskOp.
    }];

    let constructor = "vpux::IERT::createBreakDataFlowPass()";
}

//
// DMATaskProfiling
//

def DMATaskProfiling : PassBase<"dma-task-profiling", "vpux::ModulePass"> {
    let summary = "DMA task profiling using DMA-Timestamps";

    let description = [{
        This pass add DMA task profiling.
    }];

    let constructor = [{
        vpux::IERT::createDMATaskProfilingPass([](mlir::MLIRContext* ctx, vpux::StringRef memSpaceName) {
            if (memSpaceName.empty()) {
                return vpux::IndexedSymbolAttr::get(ctx, "CMX_NN");
            }
            VPUX_THROW_UNLESS(!memSpaceName.empty(), "Missing memory space option");
            return vpux::IndexedSymbolAttr::get(ctx, memSpaceName);
        })
    }];
}

//
// DPUProfiling
//

def DPUProfiling : PassBase<"dpu-profiling", "vpux::ModulePass"> {
    let summary = "DPU task profiling";

    let description = [{
        This pass allocate required memory for DPU profiling and perform buffer spilling
    }];

    let constructor = [{
        vpux::IERT::createDPUProfilingPass([](mlir::MLIRContext* ctx, vpux::StringRef memSpaceName) {
            if (memSpaceName.empty()) {
                return vpux::IndexedSymbolAttr::get(ctx, "CMX_NN");
            }
            VPUX_THROW_UNLESS(!memSpaceName.empty(), "Missing memory space option");
            return vpux::IndexedSymbolAttr::get(ctx, memSpaceName);
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
        vpux::IERT::createFeasibleAllocationPass(
        [](mlir::MLIRContext* ctx, vpux::StringRef memSpaceName) {
            VPUX_THROW_UNLESS(!memSpaceName.empty(), "Missing memory space option");
            return vpux::IndexedSymbolAttr::get(ctx, memSpaceName);
        },
        [](mlir::MLIRContext* ctx, vpux::StringRef secondLvlMemSpaceName) {
            return vpux::IndexedSymbolAttr::get(ctx, secondLvlMemSpaceName);
        }
        )
    }];

    let options = [
        Option<
            "memSpaceName", "memory-space",
            "std::string", [{""}],
            "Memory space to perform allocation"
        >,
        Option<
            "secondLvlMemSpaceName", "second-level-memory-space",
            "std::string", [{""}],
            "Second level memory space to perform spilling"
        >
    ];

    let dependentDialects = [
        "vpux::IERT::IERTDialect"
    ];
}

//
// ConvertScalarToTensor
//

def ConvertScalarToTensor : PassBase<"convert-scalar-to-tensor", "vpux::FunctionPass"> {
    let summary = "Convert the second scalar input to tensor for Gather";

    let description = [{
        Gather doesn't support scalar inputs (input is scalar if `rank == 0`). But the second Gather input (indices) could be scalar.
        This pass convert indices to tensor with `rank == 1` and the same content.
    }];

    let constructor = "vpux::IERT::createConvertScalarToTensorPass()";
}

#endif
