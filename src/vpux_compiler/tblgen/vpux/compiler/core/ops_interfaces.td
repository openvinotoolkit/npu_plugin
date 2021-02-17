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

#ifndef VPUX_COMPILER_CORE_OPS_INTERFACES
#define VPUX_COMPILER_CORE_OPS_INTERFACES

include "mlir/IR/OpBase.td"

//
// ConstantInterface
//

def ConstantInterface : OpInterface<"ConstantInterface"> {
    let description = "Interface for Constant Tensor/Buffer Declaration";

    let cppNamespace = "vpux";

    let methods = [
        InterfaceMethod<
            "Get the constant content from the Operation",
            "vpux::ConstContentAttr", "getContent", (ins),
            [{}],
            [{
                return $_op.value().template cast<vpux::ConstContentAttr>();
            }]
        >,
        InterfaceMethod<
            "Get the content Type",
            "mlir::ShapedType", "getContentType", (ins),
            [{}],
            [{
                return $_op.getContent().getType();
            }]
        >,

        InterfaceMethod<
            "Get resulting Value for the Operation",
            "mlir::Value", "output", (ins),
            [{
                return $_op.output();
            }]
        >,
        InterfaceMethod<
            "Get the output Value Type",
            "mlir::ShapedType", "getActualType", (ins),
            [{}],
            [{
                return $_op.getType();
            }]
        >,

        InterfaceMethod<
            "Convert Content to actual output Type",
            "void", "convertTo", (ins "MutableArrayRef<char>":$buf),
            [{}],
            [{
                $_op.getContent().convertTo($_op.getActualType(), buf);
            }]
        >,
    ];

    let verify = [{
        return vpux::verifyConstant($_op);
    }];

    let extraTraitClassDeclaration = [{
        static mlir::OpFoldResult foldTrait(mlir::Operation* op, mlir::ArrayRef<mlir::Attribute>) {
            return mlir::cast<vpux::ConstantInterface>(op).getContent();
        }
    }];
}

//
// LayerInterface
//

def LayerInterface : OpInterface<"LayerInterface"> {
    let description = "Base interface for Layer Operation";

    let cppNamespace = "vpux";

    let methods = [
        InterfaceMethod<
            "Get all inputs (tensors or memory buffers) for the Operation",
            "vpux::SmallVector<mlir::Value>", "getInputs", (ins),
            [{
                return $_op.getInputs();
            }]
        >,

        InterfaceMethod<
            "Get the outputs (tensors or memory buffers) for the Operation",
            "vpux::SmallVector<mlir::Value>", "getOutputs", (ins),
            [{
                return $_op.getOutputs();
            }]
        >,
    ];

    let verify = [{
        return vpux::verifyLayer($_op);
    }];
}

//
// SingleInputAndResultLayer
//

def SingleInputAndResultLayer : NativeOpTrait<"SingleInputAndResultLayer"> {
    let cppNamespace = "vpux";
}

//
// ConvertLayerInterface
//

def ConvertLayerInterface : OpInterface<"ConvertLayerInterface"> {
    let description = "Interface for Convert Layer";

    let cppNamespace = "vpux";

    let methods = [
        InterfaceMethod<
            "Get the source Value for the Operation",
            "mlir::Value", "input", (ins),
            [{
                return $_op.input();
            }]
        >,
        InterfaceMethod<
            "Get the input Type",
            "mlir::ShapedType", "inputType", (ins),
            [{}],
            [{
                return $_op.input().getType().template cast<mlir::ShapedType>();
            }]
        >,

        InterfaceMethod<
            "Get destination Value for the Operation",
            "mlir::Value", "output", (ins),
            [{
                return $_op.output();
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
            "Get the source Value for the Operation",
            "mlir::Value", "input", (ins),
            [{
                return $_op.input();
            }]
        >,
        InterfaceMethod<
            "Get the input Type",
            "mlir::ShapedType", "inputType", (ins),
            [{}],
            [{
                return $_op.input().getType().template cast<mlir::ShapedType>();
            }]
        >,

        InterfaceMethod<
            "Get destination Value for the Operation",
            "mlir::Value", "output", (ins),
            [{
                return $_op.output();
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
            "uint32_t", "axisInd", (ins),
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
            "mlir::Value", "input", (ins),
            [{
                return $_op.input();
            }]
        >,
        InterfaceMethod<
            "Get the source Value for the Operation",
            "mlir::Value", "filter", (ins),
            [{
                return $_op.filter();
            }]
        >,
        InterfaceMethod<
            "Get the source Value for the Operation",
            "mlir::Value", "bias", (ins),
            [{
                return $_op.bias();
            }]
        >,

        InterfaceMethod<
            "Get destination Value for the Operation",
            "mlir::Value", "output", (ins),
            [{
                return $_op.output();
            }]
        >,
    ];

    let extraTraitClassDeclaration = [{
        vpux::SmallVector<mlir::Value> getInputs() {
            if ($_op.bias() != nullptr) {
                return {$_op.input(), $_op.filter(), $_op.bias()};
            } else {
                return {$_op.input(), $_op.filter()};
            }
        }

        vpux::SmallVector<mlir::Value> getOutputs() {
            return {$_op.output()};
        }
    }];
}

//
// ClampLayerInterface
//

def ClampLayerInterface : OpInterface<"ClampLayerInterface"> {
    let description = "Interface for Clamp Layer";

    let cppNamespace = "vpux";

    let methods = [
        InterfaceMethod<
            "Get the source Value for the Operation",
            "mlir::Value", "input", (ins),
            [{
                return $_op.input();
            }]
        >,
        InterfaceMethod<
            "Get destination Value for the Operation",
            "mlir::Value", "output", (ins),
            [{
                return $_op.output();
            }]
        >,

        InterfaceMethod<
            "Get the lower bound of the <min;max> range",
            "float", "getMin", (ins),
            [{}],
            [{
                return $_op.min().convertToFloat();
            }]
        >,
        InterfaceMethod<
            "Get the upper bound of the <min;max> range",
            "float", "getMax", (ins),
            [{}],
            [{
                return $_op.max().convertToFloat();
            }]
        >,
    ];
}

//
// EluLayerInterface
//

def EluLayerInterface : OpInterface<"EluLayerInterface"> {
    let description = "Interface for Elu Layer";

    let cppNamespace = "vpux";

    let methods = [
        InterfaceMethod<
            "Get the source Value for the Operation",
            "mlir::Value", "input", (ins),
            [{
                return $_op.input();
            }]
        >,
        InterfaceMethod<
            "Get destination Value for the Operation",
            "mlir::Value", "output", (ins),
            [{
                return $_op.output();
            }]
        >,

        InterfaceMethod<
            "Get the scale for the negative factor",
            "float", "getX", (ins),
            [{}],
            [{
                return $_op.x().convertToFloat();
            }]
        >,
    ];
}

//
// PReluLayerInterface
//

def PReluLayerInterface : OpInterface<"PReluLayerInterface"> {
    let description = "Interface for PRelu Layer";

    let cppNamespace = "vpux";

    let methods = [
        InterfaceMethod<
            "Get the source Value for the Operation",
            "mlir::Value", "input", (ins),
            [{
                return $_op.input();
            }]
        >,
        InterfaceMethod<
            "Get the source Value for the Operation",
            "mlir::Value", "negative_slope", (ins),
            [{
                return $_op.negative_slope();
            }]
        >,

        InterfaceMethod<
            "Get destination Value for the Operation",
            "mlir::Value", "output", (ins),
            [{
                return $_op.output();
            }]
        >,
    ];

    let extraTraitClassDeclaration = [{
        vpux::SmallVector<mlir::Value> getInputs() {
            return {$_op.input(), $_op.negative_slope()};
        }

        vpux::SmallVector<mlir::Value> getOutputs() {
            return {$_op.output()};
        }
    }];

}

//
// LeakyReluLayerInterface
//

def LeakyReluLayerInterface : OpInterface<"LeakyReluLayerInterface"> {
    let description = "Interface for LeakyRelu Layer";

    let cppNamespace = "vpux";

    let methods = [
        InterfaceMethod<
            "Get the source Value for the Operation",
            "mlir::Value", "input", (ins),
            [{
                return $_op.input();
            }]
        >,
        InterfaceMethod<
            "Get destination Value for the Operation",
            "mlir::Value", "output", (ins),
            [{
                return $_op.output();
            }]
        >,

        InterfaceMethod<
            "Get the source Value for the Operation",
            "float", "getNegativeSlope", (ins),
            [{}],
            [{
                return $_op.negative_slope().convertToFloat();
            }]
        >,
    ];
}

//
// EltwiseLayerInterface
//

def EltwiseLayerInterface : OpInterface<"EltwiseLayerInterface"> {
    let description = "Interface for ltwise Layer";

    let cppNamespace = "vpux";

    let extraTraitClassDeclaration = [{
        vpux::SmallVector<mlir::Value> getInputs() {
            return {$_op.input1(), $_op.input2()};
        }

        vpux::SmallVector<mlir::Value> getOutputs() {
            return {$_op.output()};
        }
    }];
}

//
// SwishLayerInterface
//

def SwishLayerInterface : OpInterface<"SwishLayerInterface"> {
    let description = "Interface for Swish Layer";

    let cppNamespace = "vpux";

    let extraTraitClassDeclaration = [{
        vpux::SmallVector<mlir::Value> getInputs() {
            if ($_op.beta() != nullptr) {
                return {$_op.input(), $_op.beta()};
            }
            else {
                return {$_op.input()};
            }
        }

        vpux::SmallVector<mlir::Value> getOutputs() {
            return {$_op.output()};
        }
    }];
}

//
// ScaleShiftLayerInterface
//

def ScaleShiftLayerInterface : OpInterface<"ScaleShiftLayerInterface"> {
    let description = "Interface for ScaleShift Layer";

    let cppNamespace = "vpux";

    let extraTraitClassDeclaration = [{
        vpux::SmallVector<mlir::Value> getInputs() {
            return {$_op.input(), $_op.weights(), $_op.biases()};
        }

        vpux::SmallVector<mlir::Value> getOutputs() {
            return {$_op.output()};
        }
    }];
}

#endif
