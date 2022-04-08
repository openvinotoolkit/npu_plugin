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

//
// AddWeightsTableToEltwiseOps
//

def AddWeightsTableToEltwiseOps : PassBase<"add-weights-table-to-eltwise", "vpux::FunctionPass"> {
    let summary = "Add weights table to EMU.NCEClusterTask op with task type ELTWISE or AVEPOOL.";

    let description = [{
        Eltwise & AVGPool NCEClusterTask ops can parse their mult and shift from weights table in emulator,
        unlike in board case. This way, there's no need to update quant params during serialization for their output.
    }];

    let constructor = "vpux::EMU::createAddWeightsTableToEmuPass()";
}

//
// RemoveWeightsAlignment
//

def RemoveWeightsAlignment : PassBase<"remove-weights-alignment", "vpux::FunctionPass"> {
    let summary = "Adjust weights shape for EMU.NCEClusterTask ops with task type CMCONV and DWCONV.";

    let description = [{
        VPU NCEDepthConvolutionOp and Channel Major NCEConvolutionOp have their weights aligned during
        IE to VPU lowering. Emulator cannot compute the ops with aligned weights or infer
        the unaligned shape. Therefore, EMU.NCEClusterTask with CMCONV or DWCONV task types will have the weights
        alignment removed.
    }];

    let constructor = "vpux::EMU::createRemoveWeightsAlignmentPass()";
}

#endif
