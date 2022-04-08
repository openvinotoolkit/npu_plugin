//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_IERT_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_IERT_OPS_INTERFACES

include "mlir/IR/OpBase.td"

//
// LayerOpInterface
//

def IERT_LayerOpInterface : OpInterface<"LayerOpInterface"> {
    let description = "Base interface for IERT Layer Operation";

    let cppNamespace = "vpux::IERT";

    let methods = [
        InterfaceMethod<
            "Get all layer input memory buffers",
            "mlir::OperandRange", "getInputs", (ins),
            [{}],
            [{
                return vpux::IERT::getLayerInputs($_op);
            }]
        >,

        InterfaceMethod<
            "Get all layer output memory buffers",
            "mlir::OperandRange", "getOutputs", (ins),
            [{}],
            [{
                return vpux::IERT::getLayerOutputs($_op);
            }]
        >,

        InterfaceMethod<
            "Get all layer memory buffers",
            "llvm::detail::concat_range<mlir::OpOperand, llvm::MutableArrayRef<mlir::OpOperand>, llvm::MutableArrayRef<mlir::OpOperand>>",
            "getOpOperands", (ins),
            [{}],
            [{
                return llvm::concat<mlir::OpOperand>($_op.getInOpOperands(), $_op.getOutOpOperands());
            }]
        >,

        InterfaceMethod<
            "Get all layer input memory buffers",
            "llvm::MutableArrayRef<mlir::OpOperand>", "getInOpOperands", (ins),
            [{}],
            [{
                return vpux::IERT::getLayerInOpOperands($_op);
            }]
        >,

        InterfaceMethod<
            "Get all layer output memory buffers",
            "llvm::MutableArrayRef<mlir::OpOperand>", "getOutOpOperands", (ins),
            [{}],
            [{
                return vpux::IERT::getLayerOutOpOperands($_op);
            }]
        >
    ];

    let extraTraitClassDeclaration = [{
        mlir::Value getViewSource(ptrdiff_t resultInd = 0) {
            return vpux::IERT::getLayerViewSource($_op, resultInd);
        }

        void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects) {
            vpux::IERT::getLayerEffects($_op, effects);
        }
    }];

    let verify = [{
        return vpux::IERT::verifyLayer($_op);
    }];
}

//
// InferLayerReturnTypes
//

class IERT_InferLayerReturnTypes<int numResult> {
    code impl = [{
        static mlir::LogicalResult inferReturnTypes(
                mlir::MLIRContext*,
                llvm::Optional<mlir::Location>,
                mlir::ValueRange operands,
                mlir::DictionaryAttr,
                mlir::RegionRange,
                llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
            return vpux::IERT::inferLayerReturnTypes(operands, }] #numResult# [{, inferredReturnTypes);
        }
    }];
}

//
// AsyncLayerOpInterface
//

def IERT_AsyncLayerOpInterface : OpInterface<"AsyncLayerOpInterface"> {
    let description = "Interface for layers that will be executed asynchronously in separate Executor";

    let cppNamespace = "vpux::IERT";

    let methods = [
        InterfaceMethod<
            "Get Executor for the asynchronous launch",
            "vpux::IndexedSymbolAttr", "getExecutor", (ins)
        >
    ];
}

//
// SoftwareLayerOpInterface
//

def IERT_SoftwareLayerOpInterface : OpInterface<"SoftwareLayerOpInterface"> {
    let description = "Interface for layers that will be executed on SHAVEs. (For VPUX37XX only)";

    let cppNamespace = "vpux::IERT";

    let methods = [
        InterfaceMethod<
            "Get information about kernel's arguments, entry point and etc.",
            "vpux::IERT::KernelInfo", "getKernelInfo", (ins)
        >
    ];
}

//
// SameShape
//

def IERT_SameShape : NativeOpTrait<"SameShape"> {
    let cppNamespace = "vpux::IERT";
}

//
// SameElementType
//

def IERT_SameElementType : NativeOpTrait<"SameElementType"> {
    let cppNamespace = "vpux::IERT";
}

//
// SameDimsOrder
//

def IERT_SameDimsOrder : NativeOpTrait<"SameDimsOrder"> {
    let cppNamespace = "vpux::IERT";
}

//
// SameInOutDimsOrder
//

def IERT_SameInOutDimsOrder : NativeOpTrait<"SameInOutDimsOrder"> {
    let cppNamespace = "vpux::IERT";
}

//
// SameInOutDimsOrder_NCHW_NHWC
//

def IERT_SameInOutDimsOrder_NCHW_NHWC : NativeOpTrait<"SameInOutDimsOrder_NCHW_NHWC"> {
    let cppNamespace = "vpux::IERT";
}

//
// SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
//

def IERT_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC : NativeOpTrait<"SameInOutDimsOrder_CHW_HWC_NCHW_NHWC"> {
    let cppNamespace = "vpux::IERT";
}

//
// AnyDimsOrder
//

def IERT_AnyDimsOrder : NativeOpTrait<"AnyDimsOrder"> {
    let cppNamespace = "vpux::IERT";
}

#endif
