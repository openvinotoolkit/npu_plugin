//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_VPUIP_PASSES
#define VPUX_COMPILER_DIALECT_VPUIP_PASSES

include "mlir/Pass/PassBase.td"

//
// MovePureViewOpBeforeCopy
//

def MovePureViewOpBeforeCopy : PassBase<"move-pure-view-op-before-copy", "vpux::FunctionPass"> {
    let summary = "Move pure view-like operations before copy";

    let description = [{
        By moving pure view-like ops, this pass creates copy operation chains, that can be fused:
        Before: CopyOp -> PermuteCast -> GenericReshape -> CopyOp
        After: PermuteCast -> GenericReshape -> CopyOp -> CopyOp
    }];

    let constructor = "vpux::VPUIP::createMovePureViewOpBeforeCopyPass()";
}

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
// OptimizeSpillingCopies
//

def OptimizeSpillingCopies : PassBase<"optimize-spilling-copies", "vpux::FunctionPass"> {
    let summary = "Optimize pattern with CMX->DDR->CMX spilling";

    let description = [{
        Optimize following pattern

        NCE Task 16 channels output -> Copy 16 channels -> Concat -> Subview -> 3 channels

        to

        NCE Task 16 channels output -> Copy 3 channels -> Concat -> Subview -> 3 channels

    }];

    let constructor = "vpux::VPUIP::createOptimizeSpillingCopiesPass()";
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
// ConvertEltwiseToInPlace
//

def ConvertEltwiseToInPlace : PassBase<"convert-eltwise-to-in-place", "vpux::FunctionPass"> {
    let summary = "Convert Eltwise operation to read and write to the same buffer in memory";

    let description = [{
        This pass will check if Eltwise operation was selected for inplace execution
        and convert the Eltwise to write the result into one of the inputs in memory.
        A view operation is added to support different quantization parameters for input/output.
        Can be extended to support different input/output memory requirements with a subview.
    }];

    let constructor = "vpux::VPUIP::createConvertEltwiseToInPlacePass()";
}

//
// SetMemorySpace
//

def SetMemorySpace : PassBase<"set-memory-space", "vpux::FunctionPass"> {
    let summary = "Set specific memory space for all memory buffers";

    let description = [{
        This pass updates all Types for internal memory buffers and function arguments and sets the specified memory space for them.
        Also updates the operand types for grouping operations, to cover scenarios where some operands are buffers and some are constants.
    }];

    let constructor = [{
        vpux::VPUIP::createSetMemorySpacePass(vpux::VPU::symbolizeEnum<VPU::MemoryKind>)
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
// DMATaskProfilingReserveMem
//

def DMATaskProfilingReserveMem : PassBase<"dma-task-profiling-reserve-mem", "vpux::ModulePass"> {
    let summary = "DMA task profiling memory reserving";

    let description = [{
        This pass adds in ModuleOp information about reserved memory for DMA profiling.
    }];

    let constructor = "vpux::VPUIP::createDMATaskProfilingReserveMemPass()";
}

//
// DMATaskProfilingAfterBarrierSched
//

def DMATaskProfilingAfterBarrierSched : PassBase<"dma-task-profiling-after-barrier", "vpux::ModulePass"> {
    let summary = "DMA task profiling handling after barrier scheduled";

    let description = [{
        This pass adds DMA profiling tasks after barrier scheduler.
    }];

    let constructor = "vpux::VPUIP::createDMATaskProfilingAfterBarrierSchedPass()";
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
// WrapWithPermuteAsNNDMA
//

def WrapWithPermuteAsNNDMA: PassBase<"wrap-with-permute-as-nndma", "vpux::FunctionPass"> {
    let summary = "Wrap op and permute operation as dma and fuse unnecessary copy ops";

    let description = [{
        This pass will replace specific op and permute as one DMA and fuse the unnecessary following copy ops
    }];

    let constructor = "vpux::VPUIP::createWrapWithPermuteAsNNDMAPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

//
// ConvertExpand
//

def ConvertExpand : PassBase<"convert-expand", "vpux::FunctionPass"> {
    let summary = "Convert Expand that cannot fuse with permute to copy and concat subgraph";

    let description = [{
        This pass will convert the Expand to copy and concat subgraph.
    }];

    let constructor = "vpux::VPUIP::createConvertExpandPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

//
// ConvertToDMA
//

def ConvertToDMA: PassBase<"convert-to-dma", "vpux::FunctionPass"> {
    let summary = "Convert Permute and DepthToSpace from SW/UPA ops to DMA ops";

    let description = [{
        This pass will convert some SW/UPA operations (e.g. DepthToSpace, Permute) to DMA ops
        if it is possible to achieve better performance
    }];

    let constructor = "vpux::VPUIP::createConvertToDMAPass()";

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

    let dependentDialects = [
        "vpux::VPURT::VPURTDialect"
    ];
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
// DumpStatisticsOfTaskOpsPass
//

def DumpStatisticsOfTaskOpsPass : PassBase<"dump-statistics-of-task-ops", "vpux::FunctionPass"> {
    let summary = "Dump the statistics of operations (used Task operations and weights compression)";

    let description = [{
        This pass dumps the statistics for of used operations (e.g. tasks, weights compression) and makes a report as warning for operations not converted to DPU.
    }];

    let constructor = "vpux::VPUIP::createDumpStatisticsOfTaskOpsPass()";
}

//
// CompressWeightsBTC
//

def CompressWeightsBTC : PassBase<"compress-weights-btc", "vpux::FunctionPass"> {
    let summary = "Compress binary data when possible using BitCompactor";

    let description = [{
        This pass applies bitcompactor to tensor binary data. The logic is the following:
        1. Find VPUIP::NNDMAOp with Const::DeclareOp source and VPURT::DeclareBufferOp target.
        2. Check that weights size matches minimal compression size.
        3. Compress weights.
        4. Wrap compressed weights to flat tensor shapes with UInt8 data type.
        5. Replace original VPUIP::NNDMAOp with VPUIP::CompressedDMAOp

        This pass also handles multicluster cases, where NNDMAOp is wrapped in NCEClusterTiling with
        DistributedBuffer output type (with DUPLICATED). SOK case is supported as well since this pass
        is set to be executed after unroll-cluster-tiling, which splits SEGMENTED buffers into per-cluster
        chunks.

        This pass behaves differently for pre-VPU37XX and post-VPU37XX platforms. For former the compression
        is done using huffman encoding and applied only to quantized data types, for the latter the
        compression is done using bit-compactor library.
    }];

    let constructor = "vpux::VPUIP::createCompressWeightsBTCPass()";
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
// UnrollSpaceToDepthDMA
//

def UnrollSpaceToDepthDMA : PassBase<"unroll-space-to-depth-dma", "vpux::FunctionPass"> {
    let summary = "Split SpaceToDepthDMA task with several NN DMA tasks";

    let description = [{
        This pass splits SpaceToDepthDMA tasks into several NN DMA tasks, which are functionally equivalent.
        Each sub SpaceToDepthDMA will be converted to a NNDMA.
    }];

    let constructor = "vpux::VPUIP::createUnrollSpaceToDepthDMAPass()";
}

//
// UnrollExpandDMA
//

def UnrollExpandDMA : PassBase<"unroll-expand-dma", "vpux::FunctionPass"> {
    let summary = "Unroll expand task with several NN DMA tasks";

    let description = [{
        This pass unroll ExpandDMA tasks with several NN DMA tasks, which are functionally equivalent.
        Each sub ExpandDMA will be converted to a NNDMA.
    }];

    let constructor = "vpux::VPUIP::createUnrollExpandDMAPass()";
}

//
// UnrollPerAxisTileDMA
//

def UnrollPerAxisTileDMA : PassBase<"unroll-per-axis-tile-dma", "vpux::FunctionPass"> {
    let summary = "Split PerAxisTileDMA task with several NN DMA tasks";

    let description = [{
        This pass splits PerAxisTileDMA tasks into several NN DMA tasks, which are functionally equivalent.
        Each sub PerAxisTileDMA will be converted to a NNDMA.
    }];

    let constructor = "vpux::VPUIP::createUnrollPerAxisTileDMAPass()";
}

//
// UnrollUpsamplingDMA
//

def UnrollUpsamplingDMA : PassBase<"unroll-upsampling-dma", "vpux::FunctionPass"> {
    let summary = "Unroll upsampling task with several NN DMA tasks";

    let description = [{
        This pass unroll UpsamplingDMA tasks with several NN DMA tasks, which are functionally equivalent.
        Each sub UpsamplingDMA will be converted to a NNDMA.
    }];

    let constructor = "vpux::VPUIP::createUnrollUpsamplingDMAPass()";
}

//
// UnrollSwKernel
//

def UnrollSwKernel : PassBase<"unroll-sw-kernel", "vpux::FunctionPass"> {
    let summary = "Unroll SwKernel task with several SwKernel.Run";

    let description = [{
        This pass unroll SwKernel task with several SwKernel.Run, which are functionally equivalent.
        Each SwkernelRun will be wrapped into a SwKernel task.
    }];

    let constructor = "vpux::VPUIP::createUnrollSwKernelPass()";
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
// ConvertFuncArgsToDeclarations
//

def ConvertFuncArgsToDeclarations : PassBase<"convert-func-args-to-declarations", "vpux::FunctionPass"> {
    let summary = "Replace use of function arguments with result of DeclareBuffer";

    let description = [{
        Operands that are network arguments are replaced by the result of DeclareBuffer operation
        with the corresponding buffer section (NetworkInput/NetworkOutput)
    }];

    let constructor = "vpux::VPUIP::createConvertFuncArgsToDeclarationsPass()";

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
// Swizzling
//

def Swizzling : PassBase<"swizzling", "vpux::FunctionPass"> {
    let summary = "Configure swizzling for eligible buffers";

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

    let constructor = "vpux::VPUIP::createSwizzlingPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect"
    ];
}

//
// DMA Barrier Optimization
//

def DMABarrierOptimization : PassBase<"dma-barrier-optimization", "vpux::FunctionPass"> {
    let summary = "Optimize DMA related barriers after dma port has been assigned for VPUX37XX";
    let constructor = "vpux::VPUIP::createDMABarrierOptimizationPass()";
}

//
// OperationStubbing
//

def OperationStubbing : PassBase<"operation-stubbing", "vpux::FunctionPass"> {
    let summary = "Stub operations with StubOp";

    let constructor = "vpux::VPUIP::createOperationStubbingPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect",
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect"
    ];
}

//
// AdjustCompressConvInputs
//

def AdjustCompressConvInputs : PassBase<"adjust-compress-conv-inputs", "vpux::FunctionPass"> {
    let summary = "Modify compress conv inputs";

    let description = [{
        This pass checks if weights from a Convolution op were previously padded with zero, remove that pad and insert a ShapeCast op.
        Also add a ShapeCast Op to activations if channels there are less than 16 channels
    }];

    let constructor = "vpux::VPUIP::createAdjustCompressConvInputsPass()";
}

//
// FuseConstants
//

def FuseConstants : PassBase<"fuse-constants", "vpux::FunctionPass"> {
    let summary = "Fuse constant inputs of NCEClusterOp";

    let description = [{
        Concatenates input constants into one in the following order:
            weight_table -> weights -> weights_sparsity_map -> activation_window
        For any NCEClusterTaskOp if the number of constants to fuse is 1 such layers are skipped
        Special Case with DWCONV, if the constants to fuse are == 2 such DWCONV are skipped as weights are not constants
        Special Case for Compressed Conv layer, if the weights are ShapeCast these such layers are skipped
    }];

    let constructor = "vpux::VPUIP::createFuseConstantsPass()";
}

//
// Patch Fused Constants
//

def PatchFusedConstants : PassBase<"patch-fused-constants", "vpux::FunctionPass"> {
    let summary = "Patch the weight table fused in fused constant";

    let description = [{
        This pass converts the U8 weight table values to I32. Updates the address of each weight set
        and activation window present in the fused constant for each output channel of the layer
    }];

    let constructor = "vpux::VPUIP::createPatchFusedConstantsPass()";
}

//
// PropagateCompressionScheme
//

def PropagateCompressionScheme : PassBase<"propagate-compression-scheme", "vpux::FunctionPass"> {
    let summary = "Compresses the type of the sparse weights";

    let description = [{
        Propagates the compression scheme attribute from the sparse buffer type to the individual types
        that are grouped into a sparse buffer. Starting from the sparse weights constant, all types up to
        the consumer NCE operation will have the compression scheme present.
    }];

    let constructor = "vpux::VPUIP::createPropagateCompressionSchemePass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

//
// UngroupSparseBuffers
//

def UngroupSparseBuffers : PassBase<"ungroup-sparse-buffers", "vpux::FunctionPass"> {
    let summary = "Ungroups sparse buffers into individual buffers";

    let description = [{
        Splits operations that work with sparse buffers into multiple operations,
        each working with an individual buffer.

        These separate operations are then surrounded by UngroupSparseBuffer
        and / or GroupSparseBuffer operations, which can be optimized-out by
        the canonicalizer.
    }];

    let constructor = "vpux::VPUIP::createUngroupSparseBuffersPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

//
// FlattenSparseWeightsTypes
//

def FlattenSparseWeightsTypes : PassBase<"flatten-sparse-weights-types", "vpux::FunctionPass"> {
    let summary = "Flattens types that have a compression scheme";

    let description = [{
        Flattens the type of the sparse weights into a binary buffer of values.
        The weights operand of a NCE operation maintains the original type.
    }];

    let constructor = "vpux::VPUIP::createFlattenSparseWeightsTypesPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

//
// ComputeSESizes
//

def ComputeSESizes : PassBase<"compute-se-sizes", "vpux::FunctionPass"> {
    let summary = "Computes the storage element sizes for sparse NCEClusterTask activations";

    let constructor = "vpux::VPUIP::createComputeSESizesPass()";

    let description = [{
        Computes the storage element sizes for the sparse activations of NCEClusterTasks.
        The pass should be called twice:
        - once with the `onlyInputsConcatOverC` set to true before use-def chains are lost,
          in order to populate the correct information for consumers of buffers that have been
          concatenated over channels; these can be explicit concat operations, broadcasted
          activations or produced by multiple variants;
        - once after cluster tiling operations are unrolled, to populate the sizes for the
          rest of the sparse activations based on the number of channels present after unrolling.

        Eltwise inputs must have both inputs generated with the same storage element size value.
    }];

    let options = [
        Option<
            "onlyInputsConcatOverC", "only-inputs-concat-over-c",
            "bool", "false",
            "Flag to choose whether to handle only inputs concatenated over channels on this pass call"
        >
    ];

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

//
// ResolveDMAWithSwizzling
//

def ResolveDMAWithSwizzling : PassBase<"resolve-dma-with-swizzling", "vpux::FunctionPass"> {
    let summary = "Transform DMAs that operate on swizzled buffers";

    let description = [{
        This pass will transform DMAs that operate on swizzled buffers so that they
        copy buffers of size aligned to 512 to satisfy swizzling HW restrictions.
        Transformation will result in a flat buffer copy of required size
    }];

    let constructor = "vpux::VPUIP::createResolveDMAWithSwizzlingPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect"
    ];
}

//
// TileActShaveKernelTask
//

def TileActShaveKernelTask : PassBase<"tile-act-shave-kernel-task", "vpux::FunctionPass"> {
    let summary = "Tile act shave kernel task on multi shaves";

    let description = [{
        This pass will tile act kernel run task on multi shaves.
    }];

    let constructor = "vpux::VPUIP::createTileActShaveKernelTaskPass()";
}

#endif
