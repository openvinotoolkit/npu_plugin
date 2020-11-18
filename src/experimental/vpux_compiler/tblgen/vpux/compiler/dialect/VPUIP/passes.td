//
// Copyright 2020 Intel Corporation.
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

#ifndef VPUX_COMPILER_DIALECT_VPUIP_PASSES
#define VPUX_COMPILER_DIALECT_VPUIP_PASSES

include "mlir/Pass/PassBase.td"

//
// AssignTensorOffsetsDDR
//

def AssignTensorOffsetsDDR : Pass<"assign-tensor-offsets-ddr", "mlir::ModuleOp"> {
    let summary = "Assign offset attribute to DeclareTensor Operations in DDR memory";

    let description = [{
        The pass uses simple LinearScan algorithm.
    }];

    let constructor = "vpux::VPUIP::createAssignTensorOffsetsDDRPass()";
}

//
// AddLinearScheduling
//

def AddLinearScheduling : Pass<"add-linear-scheduling", "mlir::ModuleOp"> {
    let summary = "Impose simple linear execution";

    let description = [{
        The pass inserts barriers to impose simple linear execution in IR order.
    }];

    let constructor = "vpux::VPUIP::createAddLinearSchedulingPass()";
}

//
// RemoveExtraDMA
//

def RemoveExtraDMA : FunctionPass<"remove-extra-dma"> {
    let summary = "Removes extra DMA tasks";

    let description = [{
        The pass removes DMA tasks, which copies between same physical memory and with the same strides.
    }];

    let constructor = "vpux::VPUIP::createRemoveExtraDMAPass()";
}

#endif
