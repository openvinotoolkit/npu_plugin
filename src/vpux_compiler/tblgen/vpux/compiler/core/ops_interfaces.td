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

#ifndef VPUX_COMPILER_CORE_OPS_INTERFACES
#define VPUX_COMPILER_CORE_OPS_INTERFACES

include "mlir/IR/OpBase.td"

//
// MultiViewOpInterface
//

def MultiViewOpInterface : OpInterface<"MultiViewOpInterface"> {
    let description = [{
        An extended version of view-like operation.
        It allows to define several resulting views in single operation.
    }];

    let cppNamespace = "vpux";

    let methods = [
        InterfaceMethod<
            "The source buffer from which the corresponding view is created, NULL if the result is not a view",
            "mlir::Value", "getViewSource", (ins "ptrdiff_t":$resultInd)
        >
    ];
}

//===========================================================================================
// Layer interfaces and traits
//===========================================================================================

//
// LayerInterface
//

def LayerInterface : OpInterface<"LayerInterface"> {
    let description = "Base interface for Layer Operation";

    let cppNamespace = "vpux";

    let methods = [
        InterfaceMethod<
            "Get all layer input tensors",
            "mlir::OperandRange", "getInputs", (ins),
            [{}],
            [{
                return $_op->getOperands();
            }]
        >,

        InterfaceMethod<
            "Get all layer output tensors",
            "mlir::ResultRange", "getOutputs", (ins),
            [{}],
            [{
                return $_op->getResults();
            }]
        >,

        InterfaceMethod<
            "Get information about current layout for Layer inputs and outputs",
            "vpux::DataOrderInfo", "getDataOrderInfo", (ins),
            [{}],
            [{
                return vpux::getLayerDataOrderInfo($_op);
            }]
        >,
    ];

    let extraTraitClassDeclaration = [{
        static mlir::LogicalResult inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> loc,
                                                    mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                    mlir::RegionRange regions, mlir::SmallVectorImpl<mlir::Type>& inferredTypes) {
            return vpux::inferTensorTypes(ConcreteOp::inferReturnTypeComponents, ctx, loc, operands, attrs, regions, inferredTypes);
        }
    }];

    let verify = [{
        return vpux::verifyLayer($_op);
    }];
}

//===========================================================================================
// Run-time layer interfaces and traits
//===========================================================================================

//
// RTLayerInterface
//

def RTLayerInterface : OpInterface<"RTLayerInterface"> {
    let description = "Base interface for RT Layer Operation";

    let cppNamespace = "vpux";

    let methods = [
        InterfaceMethod<
            "Get all layer input memory buffers",
            "mlir::OperandRange", "getInputs", (ins),
            [{}],
            [{
                return vpux::getRTLayerInputs($_op);
            }]
        >,

        InterfaceMethod<
            "Get all layer output memory buffers",
            "mlir::OperandRange", "getOutputs", (ins),
            [{}],
            [{
                return vpux::getRTLayerOutputs($_op);
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
                return vpux::getRTLayerInOpOperands($_op);
            }]
        >,

        InterfaceMethod<
            "Get all layer output memory buffers",
            "llvm::MutableArrayRef<mlir::OpOperand>", "getOutOpOperands", (ins),
            [{}],
            [{
                return vpux::getRTLayerOutOpOperands($_op);
            }]
        >,

        InterfaceMethod<
            "Get information about current layout for Layer inputs and outputs",
            "vpux::DataOrderInfo", "getDataOrderInfo", (ins),
            [{}],
            [{
                return vpux::getRTLayerDataOrderInfo($_op);
            }]
        >,
    ];

    let extraTraitClassDeclaration = [{
        mlir::Value getViewSource(ptrdiff_t resultInd = 0) {
            return vpux::getRTLayerViewSource($_op, resultInd);
        }
    }];

    let verify = [{
        return vpux::verifyRTLayer($_op);
    }];
}

//
// InferReturnTypesForRTLayer
//

class InferReturnTypesForRTLayer<int numResult> {
    code impl = [{
        static mlir::LogicalResult inferReturnTypes(
                mlir::MLIRContext*,
                llvm::Optional<mlir::Location>,
                mlir::ValueRange operands,
                mlir::DictionaryAttr,
                mlir::RegionRange,
                llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
            return vpux::inferRTLayerReturnTypes(operands, }] #numResult# [{, inferredReturnTypes);
        }
    }];
}
//
// SameShape
//

def SameShape : NativeOpTrait<"SameShape"> {
    let cppNamespace = "vpux";
}

//
// SameElementType
//

def SameElementType : NativeOpTrait<"SameElementType"> {
    let cppNamespace = "vpux";
}

//
// SameDimsOrder
//

def SameDimsOrder : NativeOpTrait<"SameDimsOrder"> {
    let cppNamespace = "vpux";
}

//
// SameInOutDimsOrder
//

def SameInOutDimsOrder : NativeOpTrait<"SameInOutDimsOrder"> {
    let cppNamespace = "vpux";
}

//
// SameInOutDimsOrder_NCHW_NHWC
//

def SameInOutDimsOrder_NCHW_NHWC : NativeOpTrait<"SameInOutDimsOrder_NCHW_NHWC"> {
    let cppNamespace = "vpux";
}

//
// SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
//

def SameInOutDimsOrder_CHW_HWC_NCHW_NHWC : NativeOpTrait<"SameInOutDimsOrder_CHW_HWC_NCHW_NHWC"> {
    let cppNamespace = "vpux";
}

//
// AnyDimsOrder
//

def AnyDimsOrder : NativeOpTrait<"AnyDimsOrder"> {
    let cppNamespace = "vpux";
}

//===========================================================================================
// Specific per-layer interfaces
//===========================================================================================

//
// ConvertLayerInterface
//

def ConvertLayerInterface : OpInterface<"ConvertLayerInterface"> {
    let description = "Interface for Convert Layer";

    let cppNamespace = "vpux";

    let methods = [
        InterfaceMethod<
            "Get the input Type",
            "mlir::ShapedType", "inputType", (ins),
            [{}],
            [{
                return $_op.input().getType().template cast<mlir::ShapedType>();
            }]
        >,
        InterfaceMethod<
            "Get the output Type",
            "mlir::ShapedType", "outputType", (ins),
            [{}],
            [{
                return $_op.output().getType().template cast<mlir::ShapedType>();
            }]
        >
    ];

    let verify = [{
        return vpux::verifyConvertLayer($_op);
    }];
}

//
// SoftMaxLayerInterface
//

def SoftMaxLayerInterface : OpInterface<"SoftMaxLayerInterface"> {
    let description = "Interface for SoftMax Layer";

    let cppNamespace = "vpux";

    let methods = [
        InterfaceMethod<
            "Get the input Type",
            "mlir::ShapedType", "inputType", (ins),
            [{}],
            [{
                return $_op.input().getType().template cast<mlir::ShapedType>();
            }]
        >,
        InterfaceMethod<
            "Get the output Type",
            "mlir::ShapedType", "outputType", (ins),
            [{}],
            [{
                return $_op.output().getType().template cast<mlir::ShapedType>();
            }]
        >,
        InterfaceMethod<
            "Get the axis index in logical dimensions",
            "int64_t", "axisInd", (ins),
            [{
                return $_op.axisInd();
            }]
        >,
        InterfaceMethod<
            "Get the axis parameter in logical dimensions",
            "vpux::Dim", "getAxisDim", (ins),
            [{}],
            [{
                return vpux::Dim($_op.axisInd());
            }]
        >,
    ];

    let verify = [{
        return vpux::verifySoftMaxLayer($_op);
    }];
}

//
// ConvolutionLayerInterface
//

def ConvolutionLayerInterface : OpInterface<"ConvolutionLayerInterface"> {
    let description = "Interface for Convolution Layer";

    let cppNamespace = "vpux";

    let methods = [
        InterfaceMethod<
            "Get the source Value for the Operation",
            "mlir::Value", "input", (ins)
        >,
        InterfaceMethod<
            "Get the source Value for the Operation",
            "mlir::Value", "filter", (ins)
        >,
        InterfaceMethod<
            "Get the source Value for the Operation",
            "mlir::Value", "bias", (ins)
        >,

        InterfaceMethod<
            "Get destination Value for the Operation",
            "mlir::Value", "output", (ins)
        >,
    ];
}

#endif
