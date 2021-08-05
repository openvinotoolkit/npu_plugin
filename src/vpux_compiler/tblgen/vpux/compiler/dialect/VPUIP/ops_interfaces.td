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
            "Serialize the Task to BLOB format",
            "vpux::VPUIP::BlobWriter::SpecificTask",
            "serialize", (ins "vpux::VPUIP::BlobWriter&":$writer)
        >,

        StaticInterfaceMethod<
            "Get the VPUIP TaskType for the Operation",
            "vpux::VPUIP::TaskType", "getTaskType", (ins),
            [{
                return ConcreteOp::getTaskType();
            }]
        >,
    ];

    let extraTraitClassDeclaration = [{
        void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects) {
            vpux::VPUIP::getTaskEffects($_op, effects);
        }
    }];
}

//
// UPATaskOpInterface
//

def VPUIP_UPATaskOpInterface : OpInterface<"UPATaskOpInterface"> {
    let description = "Interface for VPUIP UPA Task";

    let cppNamespace = "vpux::VPUIP";

    let methods = [
        InterfaceMethod<
            "Get maximal number of UPA SHAVEs to use",
            "vpux::Optional<int64_t>", "maxShaves", (ins),
            [{
                return $_op.maxShaves();
            }]
        >,
        InterfaceMethod<
            "Update maximal number of UPA SHAVEs to use",
            "void", "setMaxShaves", (ins "int64_t":$maxShaves),
            [{}],
            [{
                $_op.maxShavesAttr(vpux::getIntAttr($_op.getContext(), maxShaves));
            }]
        >,

        InterfaceMethod<
            "Is current task the trailing SW layer",
            "bool", "isTrailingSWLayer", (ins),
            [{
                return $_op.isTrailingSWLayer();
            }]
        >,
        InterfaceMethod<
            "Mark current task as trailing SW layer",
            "void", "markAsTrailingSWLayer", (ins),
            [{}],
            [{
                $_op.isTrailingSWLayerAttr(mlir::UnitAttr::get($_op.getContext()));
            }]
        >
    ];

    let verify = [{
        return vpux::VPUIP::verifyUPATask($_op);
    }];

    let extraTraitClassDeclaration = [{
        static vpux::VPUIP::TaskType getTaskType() {
            return vpux::VPUIP::TaskType::UPA;
        }
    }];
}

//
// Legacy4D
//

def VPUIP_Legacy4D : NativeOpTrait<"Legacy4D"> {
    let cppNamespace = "vpux::VPUIP";
}

#endif
