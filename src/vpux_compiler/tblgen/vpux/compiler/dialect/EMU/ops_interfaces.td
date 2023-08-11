//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_EMU_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_EMU_OPS_INTERFACES

include "mlir/IR/OpBase.td"

//
// SerializeInterface
//

def EMU_SerializeInterface : OpInterface<"SerializeInterface"> {
    let description = "Interface to serialize tasks for EMU backend";

    let cppNamespace = "vpux::EMU";

    let methods = [

        InterfaceMethod<
            "Serialize the Task to BLOB format",
            "vpux::EMU::BlobWriter::SpecificTask",
            "serialize", (ins "vpux::EMU::BlobWriter&":$writer)
        >,

        StaticInterfaceMethod<
            "Get the executor for the Operation",
            "vpux::VPU::ExecutorKind", "getExecutorKind", (ins),
            [{
                return ConcreteOp::getExecutorKind();
            }]
        >,
    ];

}

#endif
