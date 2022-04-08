//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_VPUIPREGMAPPED_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_VPUIPREGMAPPED_OPS_INTERFACES

include "mlir/IR/OpBase.td"

//
// ExecutableTaskOpInterface
//

def VPUIPRegMapped_ExecutableTaskOpInterface : OpInterface<"ExecutableTaskOpInterface"> {
    let description = "Interface for VPUIPRegMapped Executable Task";

    let cppNamespace = "vpux::VPUIPRegMapped";

    let methods = [
        InterfaceMethod<
            "Barriers that will free this task to run",
            "mlir::ValueRange", "waitBarriers", (ins),
            [{
                return $_op.waitBarriers();
            }]
        >,
        InterfaceMethod<
            "Barriers that will free this task to run",
            "mlir::MutableOperandRange", "waitBarriersMutable", (ins),
            [{
                return $_op.waitBarriersMutable();
            }]
        >,

        InterfaceMethod<
            "Barriers that will be at least partially unlocked when this task is complete",
            "mlir::ValueRange", "updateBarriers", (ins),
            [{
                return $_op.updateBarriers();
            }]
        >,
        InterfaceMethod<
            "Barriers that will be at least partially unlocked when this task is complete",
            "mlir::MutableOperandRange", "updateBarriersMutable", (ins),
            [{
                return $_op.updateBarriersMutable();
            }]
        >,
        InterfaceMethod<
            "Specifies the value by which barrier counter needs to be incremented taking into consideration the task."
            " All tasks returns 1 except DPUInvariant which hits the barrier as many times as amount of associated DPUVariants",
            "std::size_t", "getBarrierHitsCount", (ins), [{}],
            [{
                return 1;
            }]
        >,
    ];
}

//
// SingleOutputAsIndexOp
//

def VPUIPRegMapped_SingleOutputAsIndexOp : NativeOpTrait<"SingleOutputAsIndexOp"> {
    let cppNamespace = "vpux::VPUIPRegMapped";
}

#endif
