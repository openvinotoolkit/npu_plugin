//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_VPUIP_PASSES
#define VPUX_COMPILER_DIALECT_VPUIP_PASSES

include "mlir/Pass/PassBase.td"

//
// OptimizeCopies
//

def OptimizeCopies : PassBase<"optimize-copies", "vpux::FunctionPass"> {
    let summary = "Removes Copy Ops which are unnecessary";

    let description = [{
        This pass checks if Copy Op can be optimized out to reduce the amount of unnecessary DMAs and intermediate buffers.
    }];

    let constructor = "vpux::VPUIP::createOptimizeCopiesPass()";
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

    let constructor = "vpux::VPUIP::createCopyOpHoistingPass()";
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

    let constructor = "vpux::VPUIP::createOptimizeParallelCopiesPass()";
}

//
// CopyOpTiling
//

def CopyOpTiling : PassBase<"tile-copies", "vpux::FunctionPass"> {
    let summary = "Legalizes Copy Ops which do not fit hardware capabilities";

    let description = [{
        This pass checks if Copy Op can be executed at target hardware and splits it into a few tiles if necessary.
        To fit hardware requirements it should copy less or equal than 16MB(2**24 bytes) and have less than 256 planes.
        The number of planes is defined by the outermost dimension in the tensor (except for N - batch).
        Depending on the order of the data in memory, there may be several options for what to count as the number of planes.
        For example, if the dimension order (from the outermost to the innermost) is NCHW, then HW (height-width) is considered a plane,
        and the number of planes equals to the value of dimension C. The number of planes for different dimension orders:
        * For NHWC - H
        * For NCHW - C
        * For NWCH - W
    }];

    let constructor = "vpux::VPUIP::createCopyOpTilingPass()";
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
        vpux::VPUIP::createSetInternalMemorySpacePass(vpux::VPU::symbolizeEnum<VPU::MemoryKind>)
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
        This pass replaces all dynamic `alloc`/`dealloc` Operations with `VPUIP.StaticAlloc`.
        It uses simple LinearScan algorithm.
    }];

    let constructor = [{
        vpux::VPUIP::createStaticAllocationPass(vpux::VPU::symbolizeEnum<VPU::MemoryKind>)
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
// Patch Weight Table
//

def PatchWeightsTable : PassBase<"patch-weight-table", "vpux::FunctionPass"> {
    let summary = "Adjusts weights and sparsity pointers after memory scheduling";

    let description = [{
        This pass adds RelocateWeightsTable transformation to weights table constants. The transformation adds weights and sparsity base pointers
        to offset that are already filled in the weights table constants.
    }];

    let constructor = "vpux::VPUIP::createPatchWeightsTablePass()";
}

// Linearization
//

def Linearization : PassBase<"linearization", "vpux::ModulePass"> {
    let summary = "Perform linearization of the IR";

    let description = [{
        Perform linearization of the IR with fully sequential execution.
    }];

    let constructor = "vpux::VPUIP::createLinearizationPass()";
}


//
// BreakDataFlow
//

def BreakDataFlow : PassBase<"break-data-flow", "vpux::FunctionPass"> {
    let summary = "Breaks the data flow in the graph";

    let description = [{
        This pass breaks the data flow in the graph. It is required for the VPURT dialect for correct task creation
        because all VPUIP dialect tasks will be inside body of the TaskOp and it is impossible to use operation results inside another body of TaskOp.
    }];

    let constructor = "vpux::VPUIP::createBreakDataFlowPass()";
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
        vpux::VPUIP::createDMATaskProfilingPass([](vpux::StringRef memSpaceName) {
            if (memSpaceName.empty()) {
                return Optional<vpux::VPU::MemoryKind>{VPU::MemoryKind::CMX_NN};
            }

            return vpux::VPU::symbolizeEnum<VPU::MemoryKind>(memSpaceName);
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
        vpux::VPUIP::createDPUProfilingPass([](vpux::StringRef memSpaceName) {
            if (memSpaceName.empty()) {
                return Optional<vpux::VPU::MemoryKind>{VPU::MemoryKind::CMX_NN};
            }

            return vpux::VPU::symbolizeEnum<VPU::MemoryKind>(memSpaceName);
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
        This pass wraps each IERT and VPUIP layer operation into async region preserving linear execution.
    }];

    let constructor = "vpux::VPUIP::createWrapIntoAsyncRegionsPass()";

    let dependentDialects = [
        "mlir::async::AsyncDialect"
    ];
}

//
// MoveWaitResultToAsyncBlockArgs
//

def MoveWaitResultToAsyncBlockArgs : PassBase<"move-wait-result-to-async-block-args", "vpux::FunctionPass"> {
    let summary = "Moves 'async.await' result usage from 'async.execute' body to it's operands";

    let constructor = "vpux::VPUIP::createMoveWaitResultToAsyncBlockArgsPass()";
}

//
// CalculateAsyncRegionCycleCost
//

def CalculateAsyncRegionCycleCost : PassBase<"calculate-async-region-cycle-cost", "vpux::FunctionPass"> {
    let summary = "Calculates cycle cost of 'async.execute'";

    let constructor = "vpux::VPUIP::createCalculateAsyncRegionCycleCostPass()";
}

//
// MoveViewOpsIntoAsyncRegions
//

def MoveViewOpsIntoAsyncRegions : PassBase<"move-view-ops-into-async-regions", "vpux::FunctionPass"> {
    let summary = "Moves view-like Operations inside the asynchronous regions which depends on them";

    let constructor = "vpux::VPUIP::createMoveViewOpsIntoAsyncRegionsPass()";
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

    let constructor = "vpux::VPUIP::createOptimizeAsyncDepsPass()";
}

//
// GroupAsyncExecuteOps
//

def GroupAsyncExecuteOps : PassBase<"group-async-execute-ops", "vpux::FunctionPass"> {
    let summary = "Reduces number of async.execute operations";

    let description = [{
        Groups consecutive operations which utilizes the same executor and max resources into same async.execute region
    }];

    let constructor = "vpux::VPUIP::createGroupAsyncExecuteOpsPass()";

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
        vpux::VPUIP::createFeasibleAllocationPass(
        [](vpux::StringRef memSpaceName) {
            VPUX_THROW_UNLESS(!memSpaceName.empty(), "Missing memory space option");
            return vpux::VPU::symbolizeEnum<VPU::MemoryKind>(memSpaceName);
        }, vpux::VPU::symbolizeEnum<VPU::MemoryKind>
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
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect"
    ];
}

//
// MaximizeUPACycles
//

def MaximizeUPACycles : PassBase<"maximize-upa-cycles", "vpux::FunctionPass"> {
    let summary = "Expand cycles for UPA ops";

    let description = [{
        For each SHAVE_UPA executor the pass recalculates cycleBegin, cycleEnd and cycleCost
        in order to expand cycle from latest producer of executor to nearest consumer. 
        It takes maximum of cycleEnd value of producers as new cycleBegin and minimum of cycleBegin
        of consumers as new cycleEnd. 
    }];

    let constructor = "vpux::VPUIP::createMaximizeUPACyclesPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];

}

//
// MoveOperationFromDDRtoCMX
//

def MoveOperationFromDDRtoCMX : PassBase<"move-operation-from-ddr-to-cmx", "vpux::FunctionPass"> {
    let summary = "Move memory of MemPermute and DepthToSpace from DDR to CMX";

    let description = [{
        This pass will move execution of some operations (e.g. DepthToSpace) from DDR to CMX
        if it is possible to achieve better performance
    }];

    let constructor = "vpux::VPUIP::createMoveOperationFromDDRtoCMXPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

//
// UPAProfiling
//

def UPAProfiling : PassBase<"upa-profiling", "vpux::ModulePass"> {
    let summary = "upa task profiling";

    let description = [{
        This pass allocate required memory in DDR space for UPA profiling and is own profiling output to the network
    }];

    let constructor = "vpux::VPUIP::createUPAProfilingPass()";

}

//
// ActShaveProfiling
//

def ActShaveProfiling : PassBase<"act-shave-profiling", "vpux::ModulePass"> {
    let summary = "ActShave task profiling";

    let description = [{
        This pass allocate required memory for ActShaveProfiling profiling and perform buffer spilling
    }];

    let constructor = [{
        vpux::VPUIP::createActShaveProfilingPass([](vpux::StringRef memSpaceName) {
            if (memSpaceName.empty()) {
                return Optional<vpux::VPU::MemoryKind>{VPU::MemoryKind::CMX_NN};
            }

            return vpux::VPU::symbolizeEnum<VPU::MemoryKind>(memSpaceName);
        })
    }];
}

//
// GroupProfilingBuffers
//

def GroupProfilingBuffers : PassBase<"group-profiling-buffers", "vpux::ModulePass"> {
    let summary = "Group profiling buffers into single profiling output buffer";

    let description = [{
        Group profiling buffers from different profiling engines into single profiling output buffer with name as
        [offset]_[profiling name]_[offset]_[profiling name] so postprocessing tool can parse it back
    }];

    let constructor = "vpux::VPUIP::createGroupProfilingBuffersPass()";
}

//
// DumpStatisticsOfTaskOps
//

def DumpStatisticsOfTaskOps : PassBase<"dump-statistics-of-task-ops", "vpux::FunctionPass"> {
    let summary = "Dump the statistics of used Task operations";

    let description = [{
        This pass dumps the statistics of used Task operations and makes a report as warning for operations not converted to DPU.
    }];

    let constructor = "vpux::VPUIP::createDumpStatisticsOfTaskOpsPass()";
}

//
// CompressWeights
//

def CompressWeights : PassBase<"compress-weights", "vpux::FunctionPass"> {
    let summary = "Compress binary data when possible";

    let description = [{
        This pass applies bitcompactor to tensor binary data. The logic is the following:
        1. Find VPUIP::NNDMAOp with Const::DeclareOp source and VPURT::DeclareBufferOp target.
        2. Check that weights size matches minimal compression size.
        3. Compress weights.
        4. Wrap compressed weights to flat tensor shapes with UInt8 data type.
        5. Replace original VPUIP::NNDMAOp with VPUIP::CompressedDMAOp
    }];

    let constructor = "vpux::VPUIP::createCompressWeightsPass()";
}

//
// UnrollClusterTiling
//

def UnrollClusterTiling : PassBase<"unroll-cluster-tiling", "vpux::FunctionPass"> {
    let summary = "Extracts child operations from ClusterTiling operation";

    let description = [{
        Legalizes chain of ClusterTiling operations according to the tile schema
    }];

    let constructor = "vpux::VPUIP::createUnrollClusterTilingPass()";

}

//
// UnrollDepthToSpaceDMA
//

def UnrollDepthToSpaceDMA : PassBase<"unroll-depth-to-space-dma", "vpux::FunctionPass"> {
    let summary = "Split DepthToSpaceDMA task with several NN DMA tasks";

    let description = [{
        This pass spilt DepthToSpaceDMA tasks with several NN DMA tasks, which are functionally equivalent.
        Each sub DepthToSpaceDMA will be converted to a NNDMA.
        1. if input/output layout is NHWC with model block_first, number of sub DepthToSpaceDMA is same as block_size.
        2. if input/output layout is NHWC with model depth_first, number of sub DepthToSpaceDMA is OH * OW / block_size.
        block_size is the size of the spatial block. It is an attribution of DepthToSpace. 
    }];

    let constructor = "vpux::VPUIP::createUnrollDepthToSpaceDMAPass()";
}

//
// ConvertTransferOpsToDMAs
//

def ConvertTransferOpsToDMAs : PassBase<"convert-transfer-ops-to-DMAs", "vpux::FunctionPass"> {
    let summary = "Convert data transfer operations to DMA";

    let constructor = "vpux::VPUIP::createConvertTransferOpsToDMAsPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect"
    ];
}

//
// ConvertAllocationsToDeclarations
//

def ConvertAllocationsToDeclarations : PassBase<"convert-allocations-to-declarations", "vpux::FunctionPass"> {
    let summary = "Convert static allocations to declarations";

    let constructor = "vpux::VPUIP::createConvertAllocationsToDeclarationsPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect"
    ];
}

//
// ConvertViewOpsToDeclarations
//

def ConvertViewOpsToDeclarations : PassBase<"convert-view-ops-to-declarations", "vpux::FunctionPass"> {
    let summary = "Convert view-like operations to declarations";

    let constructor = "vpux::VPUIP::createConvertViewOpsToDeclarationsPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect"
    ];
}

//
// ConvertAsyncOpsToTasks
//

def ConvertAsyncOpsToTasks : PassBase<"convert-async-ops-to-tasks", "vpux::FunctionPass"> {
    let summary = "Convert Async Dialect operations to tasks";

    let description = [{
        This pass inlines 'async.execute' body to parent Block and replaces '!async.token' based dependencies with
        VPUIP virtual barriers.
    }];

    let constructor = "vpux::VPUIP::createConvertAsyncOpsToTasksPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect"
    ];
}

//
// UnrollPermuteToNNDMA
//

def UnrollPermuteToNNDMA : PassBase<"unroll-permute-to-nndma", "vpux::FunctionPass"> {
    let summary = "Transform PermuteDMA task with one or several PermuteDMA tasks";

    let description = [{
        This pass unrolls PermuteDMA task to one or several PermuteDMA tasks.
        The number of PermuteDMA depend on the number of planes (num_planes <= 256).
        1. NCHW -> NHWC: The number of planes is C.
        2. NHWC -> NCHW: The number of planes is H * W, and W must <= 256.
    }];

    let constructor = "vpux::VPUIP::createUnrollPermuteToNNDMAPass()";
}

//
// AlignmentForSwizzling
//

def AlignmentForSwizzling : PassBase<"alignment-for-swizzling", "vpux::FunctionPass"> {
    let summary = "Configure alignment for buffers which can have swizzling feature enabled";

    let description = [{
        On HW with swizzling support (VPUX37XX) enable activation swizzling for DPU to DPU
        buffers. This includes setting specific swizzling key and alignment as part of
        allocation operation.
        Swizzling requirement:
        - buffer needs to be properly aligned
        - swizzled buffers must be given in CMX space with size of multiple of 512
        - activation buffer must be one produced and consumed by DPU type task
        - buffer for weights can be swizzled and needs to have swizzling transformation performed on the content
        Device supported swizzling key
        - 0: 16 bytes alignment
        - 1: 1024 bytes alignment
        - 2: 2048 bytes alignment
        - 3: 4096 bytes alignment
        - 4: 8192 bytes alignment
        - 5: 16384 bytes alignment
    }];

    let constructor = "vpux::VPUIP::createAlignmentForSwizzling()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect"
    ];
}

//
// Swizzle Constant
//

def SwizzleConstant : PassBase<"swizzle-constant", "vpux::FunctionPass"> {
    let summary = "Swizzle weights and weight table for given swizzle key";

    let description = [{
        Swizzles weight table and weights with the swizzle key set in AlignmentForSwizzling Pass
    }];

    let constructor = "vpux::VPUIP::createSwizzleConstantPass()";
}

//
// DMA Barrier Optimization
//

def DMABarrierOptimization : PassBase<"dma-barrier-optimization", "vpux::FunctionPass"> {
    let summary = "Optimize DMA related barriers after dma port has been assigned for VPUX37XX";
    let constructor = "vpux::VPUIP::createDMABarrierOptimizationPass()";
}


#endif
