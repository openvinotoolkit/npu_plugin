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
// InferReturnTypesForLayer
//

class InferReturnTypesForLayer<int numResult> {
    code impl = [{
        static ::mlir::LogicalResult inferReturnTypes(
                ::mlir::MLIRContext *context,
                ::llvm::Optional<::mlir::Location> location,
                ::mlir::ValueRange operands,
                ::mlir::DictionaryAttr attributes,
                ::mlir::RegionRange regions,
                ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {
            inferRTLayerReturnTypes(operands, }] #numResult# [{, inferredReturnTypes);
            return ::mlir::success();
        }
    }];
}

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

//
// RTLayer
//

def RTLayer : NativeOpTrait<"RTLayer"> {
    let cppNamespace = "vpux";
}

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
            "vpux::DataOrderInfo", "getDataOrderInfo", (ins)>,

        InterfaceMethod<
            "Get all inputs (tensors or memory buffers) for the Operation",
            "mlir::OperandRange", "getInputs", (ins)>,

        InterfaceMethod<
            "Get the outputs (tensors or memory buffers) for the Operation",
            "mlir::OperandRange", "getOutputs", (ins)>,

        InterfaceMethod<
            "Get all inputs (tensors or memory buffers) for the Operation",
            "llvm::detail::concat_range<mlir::OpOperand, MutableArrayRef<mlir::OpOperand>, MutableArrayRef<mlir::OpOperand>>", "getOpOperands", (ins)>,

        InterfaceMethod<
            "Get all inputs (tensors or memory buffers) for the Operation",
            "MutableArrayRef<mlir::OpOperand>", "getInOpOperands", (ins)>,

        InterfaceMethod<
            "Get the outputs (tensors or memory buffers) for the Operation",
            "MutableArrayRef<mlir::OpOperand>", "getOutOpOperands", (ins)>,
    ];

    let verify = [{
        return vpux::verifyLayer($_op);
    }];
}

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

        StaticInterfaceMethod<
            "Get the filter outChannel Dim",
            "vpux::Dim", "filter_out_channel_dim", (ins),
            [{}],
            [{
                return vpux::Dim(0);
            }]
        >,

        StaticInterfaceMethod<
            "Get the filter inChannel Dim",
            "vpux::Dim", "filter_in_channel_dim", (ins),
            [{}],
            [{
                return vpux::Dim(1);
            }]
        >,

        StaticInterfaceMethod<
            "Get the filter spatial Y Dim",
            "vpux::Dim", "filter_spatial_height_dim", (ins),
            [{}],
            [{
                return vpux::Dim(2);
            }]
        >,

        StaticInterfaceMethod<
            "Get the filter spatial X Dim",
            "vpux::Dim", "filter_spatial_width_dim", (ins),
            [{}],
            [{
                return vpux::Dim(3);
            }]
        >,

        StaticInterfaceMethod<
            "Get the nr of kernel dims available",
            "size_t", "filter_spatial_dims", (ins),
            [{}],
            [{
                return 2;
            }]
        >,

        StaticInterfaceMethod<
            "Get the filter dim index of a spatial dim",
            "vpux::Dim", "filter_spatial_dim", (ins "int32_t":$index),
            [{}],
            [{
                VPUX_THROW_UNLESS(index >= 0 && index < 2, "Wrong spatial dimension index '{0}'", index);
                return vpux::Dim(index+2);
            }]
        >,
    ];
}


def TensorInterface_4D : OpInterface<"TensorInterface_4D"> {
    let description = "Interface for getting tensor DIM infromation";

    let cppNamespace = "vpux";

    let methods = [
        StaticInterfaceMethod<
            "Get the activation batch Dim",
            "vpux::Dim", "act_batch_dim", (ins),
            [{}],
            [{
                return vpux::Dim(0);
            }]
        >,
        StaticInterfaceMethod<
            "Get the activation channel Dim",
            "vpux::Dim", "act_channel_dim", (ins),
            [{}],
            [{
                return vpux::Dim(1);
            }]
        >,
        StaticInterfaceMethod<
            "Get the activation height Dim",
            "vpux::Dim", "act_height_dim", (ins),
            [{}],
            [{
                return vpux::Dim(2);
            }]
        >,
        StaticInterfaceMethod<
            "Get the activation widht Dim",
            "vpux::Dim", "act_width_dim", (ins),
            [{}],
            [{
                return vpux::Dim(3);
            }]
        >,

        StaticInterfaceMethod<
            "Get the tensor Dim index of a spatial dim index",
            "vpux::Dim", "act_spatial_dim", (ins "int32_t":$index),
            [{}],
            [{
                VPUX_THROW_UNLESS(index >= 0 && index < 2, "Wrong spatial dimension index '{0}'", index);
                return vpux::Dim(index+2);
            }]
        >
    ];
}

#endif
