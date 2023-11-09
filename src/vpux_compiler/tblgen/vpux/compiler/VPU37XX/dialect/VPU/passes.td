//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_ARCH_37XX_PASSES
#define VPUX_COMPILER_DIALECT_VPU_ARCH_37XX_PASSES

include "mlir/Pass/PassBase.td"

//
// AdjustForOptimizedSwKernel
//

def AdjustForOptimizedSwKernel : PassBase<"adjust-for-optimized-sw-kernel", "vpux::FunctionPass"> {
    let summary = "Adjust shape or layout of sw ops to leverage optimized kernel implementation";

    let description = [{
        The pass adjusts the shape or layout of software ops to better leverage optimized shave
        implementation
        Supported Optimizations:
        - Softmax:
          * Axis0: Adjust shape to leverage the optimized kernel for softmax with axis=0
          * MultiShave: Adjust shape to gather dimensions in the tile dimension to utilize more shave engines
    }];

    let constructor = "vpux::VPU::arch37xx::createAdjustForOptimizedSwKernelPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// SplitRealDFTOpsPass
//

def SplitRealDFTOps : PassBase<"split-real-dft-ops", "vpux::FunctionPass"> {
    let summary = "Replace RDFT and IRDFT operations with a subgraph of smaller operations";

    let description = [{
        Replace RDFT and IRDFT operations with a subgraph of smaller operations.
        VPU.RDFT = {VPU.RDFTUncutOp->VPU.SliceOp}
        VPU.IRDFT = {VPU.IDFTOp->VPU.IRDFTLastAxisOp}
    }];

    let constructor = "vpux::VPU::arch37xx::createSplitRealDFTOpsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// DecomposeGather
//

def DecomposeGather : PassBase<"decompose-gather", "vpux::FunctionPass"> {
    let summary = "Decompose gather into gatherSlice and ExtractValue";

    let description = [{
        The pass can Decompose gather into gatherSlice and ExtractValue when can't get feasible tiling strategy.
    }];

    let constructor = "vpux::VPU::arch37xx::createDecomposeGatherPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

#endif
