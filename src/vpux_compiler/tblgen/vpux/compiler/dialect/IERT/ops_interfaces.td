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
        >,

        InterfaceMethod<
            "Get information about current layout for Layer inputs and outputs",
            "vpux::IE::DataOrderInfo", "getDataOrderInfo", (ins),
            [{}],
            [{
                return vpux::IERT::getLayerDataOrderInfo($_op);
            }]
        >,
    ];

    let extraTraitClassDeclaration = [{
        mlir::Value getViewSource(ptrdiff_t resultInd = 0) {
            return vpux::IERT::getLayerViewSource($_op, resultInd);
        }
    }];

    let verify = [{
        return vpux::IERT::verifyLayer($_op);
    }];
}

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
