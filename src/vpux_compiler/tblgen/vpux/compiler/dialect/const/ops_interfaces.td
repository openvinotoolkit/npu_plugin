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

#ifndef VPUX_COMPILER_DIALECT_CONST_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_CONST_OPS_INTERFACES

include "mlir/IR/OpBase.td"

/*
def Const_OpInterface : OpInterface<"ConstOpInterface"> {
    let description = "Interface for const Const_Op";

    let cppNamespace = "vpux::Const";

    let methods = [
        InterfaceMethod<
            "Serialize the Task to BLOB format",
            "void",
            "serialize", (ins "std::vector<uint8_t>&":$buffer)
        >,
    ];
}
*/

#endif
