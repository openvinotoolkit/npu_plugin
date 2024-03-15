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
            "Get the single result of the op",
            "mlir::Value", "getResult", (ins), [{
                return $_op.getResult();
            }]
        >,
        InterfaceMethod<
            "Set task location",
            "void", "setTaskLocation", (ins "mlir::Value":$location), [{}], [{
                $_op.getTaskLocationMutable().assign(location);
            }]
        >,

        InterfaceMethod<
            "Get the location projection of this task",
            "mlir::Value", "getTaskLocation", (ins), [{}], [{
                return $_op.taskLocation();
            }]
        >,
        InterfaceMethod<
            "Get the previous task inside list",
            "vpux::VPURegMapped::TaskOpInterface", "getPreviousTask", (ins),
            [{
                auto prev = $_op.getPreviousTask();
                if(prev)
                    return mlir::cast<VPURegMapped::TaskOpInterface>(prev.getDefiningOp());
                else
                    return nullptr;
            }]
        >,
        InterfaceMethod<
            "Set the previous task of this Op",
            "void", "setPreviousTask", (ins "mlir::Value":$previousTask), [{}], [{
                $_op.getPreviousTaskMutable().assign(previousTask);
            }]
        >

    ];
}

#endif
