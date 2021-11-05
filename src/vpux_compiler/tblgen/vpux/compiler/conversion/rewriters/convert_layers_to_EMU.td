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

#ifndef VPUX_COMPILER_CONVERSION_REWRITERS_CONVERT_LAYERS_TO_EMU
#define VPUX_COMPILER_CONVERSION_REWRITERS_CONVERT_LAYERS_TO_EMU

include "vpux/compiler/dialect/IE/ops.td"
include "vpux/compiler/dialect/EMU/ops.td"

include "mlir/IR/OpBase.td"

//
// Constraints
//

def HasNoValue : Constraint<CPred<"$0 == nullptr">>;

//
// IE.SoftMax -> EMU.SoftMaxUPA
//

def createSoftMaxUPAOp :
        NativeCodeCall<[{
            $_builder.create<vpux::EMU::SoftMaxUPAOp>($_loc, $0.getType(), $1, $2)
        }]>;

def RewriteSoftMax :
        Pat<
            (IE_SoftMaxOp:$result $input, $axisInd),
            (createSoftMaxUPAOp $result, $input, $axisInd)
        >;

#endif
