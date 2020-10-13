//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
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
            "All inputs for this task",
            "mlir::ValueRange", "inputTensors", (ins)
        >,
        InterfaceMethod<
            "All outputs for this task",
            "mlir::ValueRange", "outputTensors", (ins)
        >,

        InterfaceMethod<
            "Barriers that will free this task to run",
            "mlir::ValueRange", "waitBarriers", (ins)
        >,
        InterfaceMethod<
            "Barriers that will free this task to run",
            "mlir::MutableOperandRange", "waitBarriersMutable", (ins)
        >,

        InterfaceMethod<
            "Barriers that will be at least partially unlocked when this task is complete",
            "mlir::ValueRange", "updateBarriers", (ins)
        >,
        InterfaceMethod<
            "Barriers that will be at least partially unlocked when this task is complete",
            "mlir::MutableOperandRange", "updateBarriersMutable", (ins)
        >,

        InterfaceMethod<
            "Serialize the Task to BLOB format",
            "vpux::VPUIP::BlobWriter::SpecificTask",
            "serialize", (ins "vpux::VPUIP::BlobWriter&":$writer)
        >,

        StaticInterfaceMethod<
            "Get the VPUIP TaskType for the Operation",
            "vpux::VPUIP::TaskType", "getTaskType", (ins)
        >
    ];

    let extraTraitClassDeclaration = [{
        void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects) {
            vpux::VPUIP::getTaskEffects(this->getOperation(), effects);
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
            "uint32_t", "maxShaves", (ins)
        >,
        InterfaceMethod<
            "Update maximal number of UPA SHAVEs to use",
            "void", "setMaxShaves", (ins "uint32_t":$maxShaves),
            [{
                $_op.maxShavesAttr(vpux::getInt32Attr($_op.getContext(), maxShaves));
            }]
        >,

        InterfaceMethod<
            "Is current task the trailing SW layer",
            "bool", "isTrailingSWLayer", (ins)
        >,
        InterfaceMethod<
            "Mark current task as trailing SW layer",
            "void", "markAsTrailingSWLayer", (ins),
            [{
                $_op.isTrailingSWLayerAttr(mlir::UnitAttr::get($_op.getContext()));
            }]
        >
    ];

    let verify = [{
        return vpux::VPUIP::verifyUPATask($_op);
    }];
}

#endif
