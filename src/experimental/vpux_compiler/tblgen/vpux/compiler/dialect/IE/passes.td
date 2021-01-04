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

#ifndef VPUX_COMPILER_DIALECT_IE_PASSES
#define VPUX_COMPILER_DIALECT_IE_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Dialect/Linalg/IR/LinalgOps.td"

//
// ConvertPrecisionToFP16
//

def ConvertPrecisionToFP16 : Pass<"convert-precision-to-fp16", "mlir::ModuleOp"> {
    let summary = "Convert tensors precision from FP32 to FP16";

    let description = [{
        This "atomic" pass replaces all FP32 tensors with FP16.
        It updates both function bodies as well as Function signatures.
    }];

    let constructor = "vpux::IE::createConvertPrecisionToFP16Pass()";
}

//
// ConvertShapeTo4D
//

def ConvertShapeTo4D : Pass<"convert-shape-to-4d", "mlir::ModuleOp"> {
    let summary = "Convert tensors shapes to 4D";

    let description = [{
        This "atomic" pass replaces all ND tensor with 4D analogues.
        Tensor with rank greater than 4 will be treated as an error.
    }];

    let constructor = "vpux::IE::createConvertShapeTo4DPass()";

    let dependentDialects = [
        "mlir::linalg::LinalgDialect"
    ];
}

//
// ConvertPaddingsToFloorMode
//

def ConvertPaddingsToFloorMode : FunctionPass<"convert-paddings-to-floor-mode"> {
    let summary = "Convert Convolution and Pooling layers paddings to FLOOR rouding mode";

    let description = [{
        This "atomic" pass updates padding attributes for Convolution and Pooling layers.
        It switches layer rounding mode to FLOOR and updates paddings to satisfy output shape.
    }];

    let constructor = "vpux::IE::createConvertPaddingsToFloorModePass()";
}

//
// AdjustForVPU
//

def AdjustForVPU : Pass<"adjust-for-vpu", "mlir::ModuleOp"> {
    let summary = "Adjust IE Dialect IR for VPU target";

    let description = [{
        This "pipeline" pass includes various adaptation passes to adjust the IR for VPU target.
    }];

    let constructor = "vpux::IE::createAdjustForVPUPass()";

    let dependentDialects = [
        "mlir::linalg::LinalgDialect"
    ];
}

#endif
