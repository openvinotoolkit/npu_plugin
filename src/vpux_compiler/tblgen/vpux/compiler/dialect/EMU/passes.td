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

#ifndef VPUX_COMPILER_DIALECT_EMU_PASSES
#define VPUX_COMPILER_DIALECT_EMU_PASSES

include "mlir/Pass/PassBase.td"

//
// SqueezeBiasShape
//

def SqueezeBiasShape : PassBase<"squeeze-bias-shape", "vpux::FunctionPass"> {
    let summary = "Squeeze shape of the bias tensor.";

    let description = [{
        Reshape the bias tensor into a 1D tensor by taking the 2nd dim.
        Needed due to limitations on firmware side.
    }];

    let constructor = "vpux::EMU::createSqueezeBiasShapePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// AdjustFQPrecision
//

def AdjustFQPrecision : PassBase<"adjust-fq-precision", "vpux::FunctionPass"> {
    let summary = "Convert the precision of the FQ constants.";

    let description = [{
        Ensure the FQ constants for in/out low/high are of FP16 dtype.
        This is needed becaus of emulator restriction of schedulign SW with
        different datatypes per input tensors.
    }];

    let constructor = "vpux::EMU::createAdjustFQPrecisionPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

#endif
