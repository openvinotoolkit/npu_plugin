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
            "vpux::Optional<uint32_t>", "maxShaves", (ins),
            [{
                return $_op.maxShaves();
            }]
        >,
        InterfaceMethod<
            "Update maximal number of UPA SHAVEs to use",
            "void", "setMaxShaves", (ins "uint32_t":$maxShaves),
            [{}],
            [{
                $_op.maxShavesAttr(vpux::getInt32Attr($_op.getContext(), maxShaves));
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
// NCETaskOpInterface
//

def VPUIP_NCETaskOpInterface : OpInterface<"NCETaskOpInterface"> {
    let description = "Interface for VPUIP NCE Task";

    let cppNamespace = "vpux::VPUIP";

    let verify = [{
        return vpux::VPUIP::verifyNCETask($_op);
    }];

    let extraTraitClassDeclaration = [{
        static vpux::VPUIP::TaskType getTaskType() {
            return vpux::VPUIP::TaskType::NCE2;
        }
    }];
}

//
// SameShape
//

def VPUIP_SameShape : NativeOpTrait<"SameShape"> {
    let cppNamespace = "vpux::VPUIP";
}

//
// SameElementType
//

def VPUIP_SameElementType : NativeOpTrait<"SameElementType"> {
    let cppNamespace = "vpux::VPUIP";
}

//
// SameDimsOrder
//

def VPUIP_SameDimsOrder : NativeOpTrait<"SameDimsOrder"> {
    let cppNamespace = "vpux::VPUIP";
}

//
// SameInOutDimsOrder
//

def VPUIP_SameInOutDimsOrder : NativeOpTrait<"SameInOutDimsOrder"> {
    let cppNamespace = "vpux::VPUIP";
}

//
// SameInOutDimsOrder_NCHW_NHWC
//

def VPUIP_SameInOutDimsOrder_NCHW_NHWC : NativeOpTrait<"SameInOutDimsOrder_NCHW_NHWC"> {
    let cppNamespace = "vpux::VPUIP";
}

//
// SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
//

def VPUIP_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC : NativeOpTrait<"SameInOutDimsOrder_CHW_HWC_NCHW_NHWC"> {
    let cppNamespace = "vpux::VPUIP";
}

//
// AnyDimsOrder
//

def VPUIP_AnyDimsOrder : NativeOpTrait<"AnyDimsOrder"> {
    let cppNamespace = "vpux::VPUIP";
}

//
// Legacy4D
//

def VPUIP_Legacy4D : NativeOpTrait<"Legacy4D"> {
    let cppNamespace = "vpux::VPUIP";
}

#endif
