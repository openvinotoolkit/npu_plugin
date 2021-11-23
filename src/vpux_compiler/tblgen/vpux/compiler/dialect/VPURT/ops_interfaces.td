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

#ifndef VPUX_COMPILER_DIALECT_VPURT_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_VPURT_OPS_INTERFACES

include "mlir/IR/OpBase.td"

//
// SerializeInterface
//

def VPURT_SerializeInterface : OpInterface<"SerializeInterface"> {
    let description = "Interface for blob serialization";

    let cppNamespace = "vpux::VPURT";

    let methods = [
        InterfaceMethod<
            "Serialize the Task to BLOB format",
            "vpux::VPUIP::BlobWriter::SpecificTask",
            "serialize", (ins "vpux::VPUIP::BlobWriter&":$writer)
        >,
    ];
}

#endif
