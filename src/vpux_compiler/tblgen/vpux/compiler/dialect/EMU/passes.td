//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_EMU_PASSES
#define VPUX_COMPILER_DIALECT_EMU_PASSES

include "mlir/Pass/PassBase.td"

//
// SqueezeBiasShape
//

def SqueezeBiasShape : PassBase<"squeeze-bias-shape", "vpux::FunctionPass"> {
    let summary = "Squeeze shape of the bias tensor.";

    let description = [{
        Reshape the bias tensor into a 1D tensor by only keeping the channel dimension.
        Needed due to limitations on firmware side.
    }];

    let constructor = "vpux::EMU::createSqueezeBiasShapePass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// AdjustFQPrecision
//

def AdjustFQPrecision : PassBase<"adjust-fq-precision", "vpux::FunctionPass"> {
    let summary = "Convert the precision of the FQ constants.";

    let description = [{
        Ensure the FQ constants for in/out low/high are of FP16 dtype.
        This is needed becaus of emulator restriction of scheduling SW with
        different datatypes per input tensors.
    }];

    let constructor = "vpux::EMU::createAdjustFQPrecisionPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

#endif
