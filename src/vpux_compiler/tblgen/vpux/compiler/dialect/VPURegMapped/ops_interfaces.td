//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUREGMAPPED_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_VPUREGMAPPED_OPS_INTERFACES

include "mlir/IR/OpBase.td"

def VPURegMapped_TaskOpInterface : OpInterface<"TaskOpInterface"> {
    let description = "Task Op interface";
    let cppNamespace = "vpux::VPURegMapped";
    let methods = [
        StaticInterfaceMethod<
            "Get task type",
            "vpux::VPURegMapped::TaskType", "getTaskType"
        >,
        InterfaceMethod<
            "Get index",
            "vpux::VPURegMapped::IndexType", "getIndexType", (ins), [{}], [{
                return $_op.getType();
            }]
        >,
        InterfaceMethod<
            "Set task location",
            "void", "setTaskLocation", (ins "mlir::Value":$location), [{}], [{
                $_op.getTaskLocationMutable().assign(location);
            }]
        >
    ];
}

#endif
