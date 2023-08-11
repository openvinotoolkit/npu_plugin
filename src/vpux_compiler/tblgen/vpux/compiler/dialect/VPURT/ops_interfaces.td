//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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
