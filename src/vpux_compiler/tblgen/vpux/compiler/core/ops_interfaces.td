//
// Copyright 2020 Intel Corporation.
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
            "Get the nr of kernel spatial dims available",
            "uint32_t", "filter_spatial_dims", (ins),
            [{}],
            [{
                return 2;
            }]
        >,

        StaticInterfaceMethod<
            "Get the filter dim index of a spatial dim",
            "vpux::Dim", "filter_spatial_dim", (ins "size_t":$index),
            [{}],
            [{
                VPUX_THROW_UNLESS(index < 2, "Wrong spatial dimension index '{0}'", index);
                return vpux::Dim(index + 2);
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
            "Get the activation width Dim",
            "vpux::Dim", "act_width_dim", (ins),
            [{}],
            [{
                return vpux::Dim(3);
            }]
        >,

        StaticInterfaceMethod<
            "Get the nr of activation spatial dims available",
            "uint32_t", "act_spatial_dims", (ins),
            [{}],
            [{
                return 2;
            }]
        >,

        StaticInterfaceMethod<
            "Get the tensor Dim index of a spatial dim index",
            "vpux::Dim", "act_spatial_dim", (ins "size_t":$index),
            [{}],
            [{
                VPUX_THROW_UNLESS(index < 2, "Wrong spatial dimension index '{0}'", index);
                return vpux::Dim(index + 2);
            }]
        >
    ];
}

#endif
