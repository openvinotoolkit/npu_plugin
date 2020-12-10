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
// AdjustPrecisionForVPU
//

def AdjustPrecisionForVPU : Pass<"adjust-precision-for-vpu", "mlir::ModuleOp"> {
    let summary = "Adjusts the IE CNN Network layers inputs/outputs for VPU";

    let description = [{
        This pass replaces all FP32 with FP16 and replaces all Integer types with I32.
    }];

    let constructor = "vpux::IE::createAdjustPrecisionForVPUPass()";
}

//
// ConvertNDOpsTo4D
//

def ConvertNDOpsTo4D : Pass<"convert-Nd-ops-to-4d", "mlir::ModuleOp"> {
    let summary = "Convert input/output types to 4D tensors";

    let description = [{
        This pass replaces all ND operations to 4d tensor. Tensor > 4D will be treated as an error.
    }];

    let constructor = "vpux::IE::createConvertNDOpsTo4DPass()";

    let dependentDialects = [
        "mlir::linalg::LinalgDialect"
    ];


}

#endif
