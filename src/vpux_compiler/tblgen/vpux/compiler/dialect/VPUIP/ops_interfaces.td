//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_VPUIP_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_VPUIP_OPS_INTERFACES

include "mlir/IR/OpBase.td"

//
// TaskOpInterface
//

def VPUIP_TaskOpInterface : OpInterface<"TaskOpInterface"> {
    let description = "Interface for VPUIP Task";

    let cppNamespace = "vpux::VPUIP";

    let methods = [
        InterfaceMethod<
            "Serialize the Task to BLOB format",
            "vpux::VPUIP::BlobWriter::SpecificTask",
            "serialize", (ins "vpux::VPUIP::BlobWriter&":$writer)
        >,

        StaticInterfaceMethod<
            "Get the executor for the Operation",
            "vpux::VPU::ExecutorKind", "getExecutorKind", (ins),
            [{
                return ConcreteOp::getExecutorKind();
            }]
        >,
    ];

    let extraTraitClassDeclaration = [{
        vpux::IndexedSymbolAttr getExecutor() {
            return vpux::VPUIP::getTaskOpExecutor($_op);
        }
    }];
}

//
// UPATask
//

def VPUIP_UPATask : NativeOpTrait<"UPATask"> {
    let cppNamespace = "vpux::VPUIP";
}

//
// Legacy4D
//

def VPUIP_Legacy4D : NativeOpTrait<"Legacy4D"> {
    let cppNamespace = "vpux::VPUIP";
}

#endif
