//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUMI37XX_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_VPUMI37XX_OPS_INTERFACES

include "mlir/IR/OpBase.td"

//
// ExecutableTaskOpInterface
//

def VPUMI37XX_ExecutableTaskOpInterface : OpInterface<"ExecutableTaskOpInterface"> {
    let description = "Interface for VPUMI37XX Executable Task";

    let cppNamespace = "vpux::VPUMI37XX";

    let methods = [
        InterfaceMethod<
            "Barriers that will free this task to run",
            "mlir::ValueRange", "waitBarriers", (ins),
            [{
                return $_op.getWaitBarriers();
            }]
        >,
        InterfaceMethod<
            "Barriers that will free this task to run",
            "mlir::MutableOperandRange", "waitBarriersMutable", (ins),
            [{
                return $_op.getWaitBarriersMutable();
            }]
        >,

        InterfaceMethod<
            "Barriers that will be at least partially unlocked when this task is complete",
            "mlir::ValueRange", "updateBarriers", (ins),
            [{
                return $_op.getUpdateBarriers();
            }]
        >,
        InterfaceMethod<
            "Barriers that will be at least partially unlocked when this task is complete",
            "mlir::MutableOperandRange", "updateBarriersMutable", (ins),
            [{
                return $_op.getUpdateBarriersMutable();
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

def VPUMI37XX_SingleOutputAsIndexOp : NativeOpTrait<"SingleOutputAsIndexOp"> {
    let cppNamespace = "vpux::VPUMI37XX";
}

#endif
