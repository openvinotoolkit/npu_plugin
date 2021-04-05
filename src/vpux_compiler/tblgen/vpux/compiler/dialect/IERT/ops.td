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

#ifndef VPUX_COMPILER_DIALECT_IERT_OPS
#define VPUX_COMPILER_DIALECT_IERT_OPS

include "vpux/compiler/core/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/IERT/dialect.td"
include "vpux/compiler/dialect/IERT/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"

include "mlir/Dialect/Quant/QuantOpsBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/RegionKindInterface.td"

//
// Base classes
//

class IERT_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<
            IERT_Dialect,
            mnemonic,
            traits
        >;

class IERT_LayerOp<string mnemonic, list<OpTrait> traits = []> :
        IERT_Op<
            mnemonic,
            !listconcat(
                traits,
                [
                    IERT_RunTimeLayer,
                    MemoryEffectsOpInterface,
                    DeclareOpInterfaceMethods<LayerInterface>
                ]
            )
        > {
    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands)
    }];
}

//
// RunTimeResourcesOp
//

def IERT_RunTimeResourcesOp :
        IERT_Op<
            "RunTimeResources",
            [
                IsolatedFromAbove,
                HasParent<"mlir::ModuleOp">,
                NoRegionArguments
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Definition of run-time resources";

    let description = [{
        This operation defines various resources consumed at run-time:

          * Available memory spaces for interal buffers.
          * Used memory spaces for interal buffers.
          * Executors for asynchronous calls.
    }];

    let regions = (region
        SizedRegion<1>:$availableMemory,
        SizedRegion<1>:$usedMemory,
        SizedRegion<1>:$executors
    );

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins)>
    ];

    let verifier = [{
        return vpux::IERT::verifyOp(*this);
    }];

    let extraClassDeclaration = [{
        static vpux::IERT::RunTimeResourcesOp getFromModule(mlir::ModuleOp module);

        vpux::IERT::MemoryResourceOp addAvailableMemory(mlir::Attribute kind, vpux::Byte size);
        vpux::IERT::MemoryResourceOp getAvailableMemory(mlir::Attribute kind);
        auto getAvailableMemory() { return availableMemory().getOps<vpux::IERT::MemoryResourceOp>(); }

        vpux::IERT::MemoryResourceOp setUsedMemory(mlir::Attribute kind, vpux::Byte size);
        vpux::IERT::MemoryResourceOp getUsedMemory(mlir::Attribute kind);
        auto getUsedMemory() { return usedMemory().getOps<vpux::IERT::MemoryResourceOp>(); }

        vpux::IERT::ExecutorResourceOp addExecutor(mlir::Attribute kind, uint32_t count, bool withSubRegion = false);
        vpux::IERT::ExecutorResourceOp getExecutor(mlir::Attribute kind);
        auto getExecutors() { return executors().getOps<vpux::IERT::ExecutorResourceOp>(); }
    }];

    let assemblyFormat = [{
        attr-dict
        `availableMemory` `:` $availableMemory
        `usedMemory` `:` $usedMemory
        `executors` `:` $executors
    }];
}

//
// MemoryResourceOp
//

def IERT_MemoryResourceOp :
        IERT_Op<
            "MemoryResource",
            [
                HasParent<"vpux::IERT::RunTimeResourcesOp">
            ]
        > {
    let summary = "Information about memory resource";

    let description = [{
        The memory resource is defined by the following attributes:

          * Kind - optional kind of memory space.
          * Size - size in bytes of memory space.
    }];

    let arguments = (ins
        OptionalAttr<AnyAttr>:$kind,
        I64Attr:$byteSize
    );

    let extraClassDeclaration = [{
        vpux::Byte size() { return vpux::Byte(byteSize()); }
    }];

    let assemblyFormat = [{
        $byteSize `bytes` (`of` $kind^)?
        attr-dict
    }];
}

//
// ExecutorResourceOp
//

def IERT_ExecutorResourceOp :
        IERT_Op<
            "ExecutorResource",
            [
                ParentOneOf<["vpux::IERT::ExecutorResourceOp","vpux::IERT::RunTimeResourcesOp"]>,
                NoRegionArguments
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Information about executor resource";

    let description = [{
        The executor resource is defined by the following attributes:

          * Kind - optional kind of the executor.
          * Count - number of executor units.
    }];

    let arguments = (ins
        AnyAttr:$kind,
        I32Attr:$count
    );

    let regions = (region
        VariadicRegion<SizedRegion<1>>:$subExecutors
    );

    let verifier = [{
        return vpux::IERT::verifyOp(*this);
    }];

    let extraClassDeclaration = [{
        vpux::IERT::ExecutorResourceOp addSubExecutor(mlir::Attribute kind, uint32_t count, bool withSubRegion = false);
        vpux::IERT::ExecutorResourceOp getSubExecutor(mlir::Attribute kind);
    }];

    let assemblyFormat = [{
        attr-dict
        $count `of` $kind
        $subExecutors
    }];
}

//
// StaticAllocOp
//

def IERT_StaticAllocOp :
        IERT_Op<
            "StaticAlloc",
            [
                NoSideEffect
            ]
        > {
    let summary = "InferenceEngine run-time static buffer allocation";

    let arguments = (ins
        I64Attr:$offset
    );

    let results = (outs
        AnyMemRef:$memory
    );

    let assemblyFormat = [{
        `<` $offset `>` attr-dict `->` type(results)
    }];
}

//
// ConstantOp
//

def IERT_ConstantOp :
        IERT_Op<
            "Constant",
            [
                ConstantLike,
                NoSideEffect,
                DeclareOpInterfaceMethods<ConstantInterface>
            ]
        > {
    let summary = "Constant memory buffer declaration";

    let description = [{
        This operation can perform extra transformations for constant content:

        * Reshape
        * Relayout
        * Precision conversion
    }];

    let arguments = (ins
        ElementsAttr:$value
    );

    let results = (outs
        AnyStaticShapeMemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        type($output) `=` $value
    }];
}

//
// ConvertOp
//

def IERT_ConvertOp :
        IERT_LayerOp<
            "Convert",
            [
                SameOperandsShape,
                SingleInputAndResultLayer,
                DeclareOpInterfaceMethods<ConvertLayerInterface>
            ]
        > {
    let summary = "InferenceEngine run-time Convert layer";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output
    );
}

//
// CopyOp
//

def IERT_CopyOp :
        IERT_LayerOp<
            "Copy",
            [
                SameOperandsShape,
                SameOperandsElementType,
                SingleInputAndResultLayer
            ]
        > {
    let summary = "InferenceEngine run-time Copy layer";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output
    );
}

//
// SoftMaxOp
//

def IERT_SoftMaxOp :
        IERT_LayerOp<
            "SoftMax",
            [
                SameOperandsShape,
                SameOperandsElementType,
                SingleInputAndResultLayer,
                DeclareOpInterfaceMethods<SoftMaxLayerInterface>
            ]
        > {
    let summary = "InferenceEngine run-time SoftMax layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output,

        I32Attr:$axisInd
    );
}

//
// AvgPoolOp
//

def IERT_AvgPoolOp :
        IERT_LayerOp<
            "AvgPool",
            [
                SameOperandsElementType,
                SingleInputAndResultLayer
            ]
        > {
    let summary = "InferenceEngine run-time AvgPool layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output,

        I32ArrayAttr:$kernel_size,
        I32ArrayAttr:$strides,
        I32ArrayAttr:$pads_begin,
        I32ArrayAttr:$pads_end
    );
}

//
// MaxPoolOp
//

def IERT_MaxPoolOp :
        IERT_LayerOp<
            "MaxPool",
            [
                SameOperandsElementType,
                SingleInputAndResultLayer
            ]
        > {
    let summary = "InferenceEngine run-time MaxPool layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output,

        I32ArrayAttr:$kernel_size,
        I32ArrayAttr:$strides,
        I32ArrayAttr:$pads_begin,
        I32ArrayAttr:$pads_end
    );
}

//
// ConvolutionOp
//

def IERT_ConvolutionOp :
        IERT_LayerOp<
            "Convolution",
            [
                SameOperandsElementType,
                DeclareOpInterfaceMethods<ConvolutionLayerInterface>,
                DeclareOpInterfaceMethods<LayerInterface>
            ]
        > {
    let summary = "InferenceEngine run-time Convolution layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$filter,
        Optional<MemRefOf<[F16, F32]>>:$bias,
        MemRefOf<[F16, F32]>:$output,

        I32ArrayAttr:$strides,
        I32ArrayAttr:$pads_begin,
        I32ArrayAttr:$pads_end,
        I32ArrayAttr:$dilations
    );
}

//
// GroupConvolutionOp
//

def IERT_GroupConvolutionOp :
        IERT_LayerOp<
            "GroupConvolution",
            [
                SameOperandsElementType,
                DeclareOpInterfaceMethods<ConvolutionLayerInterface>,
                DeclareOpInterfaceMethods<LayerInterface>
            ]
        > {
    let summary = "InferenceEngine run-time GroupConvolution layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$filter,
        Optional<MemRefOf<[F16, F32]>>:$bias,
        MemRefOf<[F16, F32]>:$output,

        I32ArrayAttr:$strides,
        I32ArrayAttr:$pads_begin,
        I32ArrayAttr:$pads_end,
        I32ArrayAttr:$dilations,
        OptionalAttr<I32Attr>:$groups
    );
}

//
// ReLUOp
//

def IERT_ReLUOp :
        IERT_LayerOp<
            "ReLU",
            [
                SingleInputAndResultLayer
            ]
        > {
    let summary = "InferenceEngine run-time ReLU layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output
    );
}

//
// SigmoidOp
//

def IERT_SigmoidOp :
        IERT_LayerOp<
            "Sigmoid",
            [
                SingleInputAndResultLayer
            ]
        > {
    let summary = "InferenceEngine run-time Sigmoid layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output
    );
}

//
// ClampOp
//

def IERT_ClampOp :
        IERT_LayerOp<
            "Clamp",
            [
                SameOperandsShape,
                SameOperandsElementType,
                SingleInputAndResultLayer,
                DeclareOpInterfaceMethods<ClampLayerInterface>
            ]
        > {
    let summary = "InferenceEngine run-time Clamp layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output,

        F32Attr:$min,
        F32Attr:$max
    );
}

//
// EluOp
//

def IERT_EluOp :
        IERT_LayerOp<
            "Elu",
            [
                SameOperandsShape,
                SameOperandsElementType,
                SingleInputAndResultLayer,
                DeclareOpInterfaceMethods<EluLayerInterface>
            ]
        > {
    let summary = "InferenceEngine run-time Elu layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output,

        F32Attr:$x
    );
}

//
// HSwishOp
//

def IERT_HSwishOp :
        IERT_LayerOp<
            "HSwish",
            [
                SameOperandsShape,
                SameOperandsElementType,
                SingleInputAndResultLayer
            ]
        > {
    let summary = "InferenceEngine run-time HSwish layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output
    );
}

//
// TanhOp
//

def IERT_TanhOp :
        IERT_LayerOp<
            "Tanh",
            [
                SameOperandsShape,
                SameOperandsElementType,
                SingleInputAndResultLayer
            ]
        > {
    let summary = "InferenceEngine run-time Tanh layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output
    );
}

//
// FakeQuantizeOp
//

def IERT_FakeQuantizeOp :
        IERT_LayerOp<
            "FakeQuantize"
        > {
    let summary = "InferenceEngine FakeQuantize layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$input_low,
        MemRefOf<[F16, F32]>:$input_high,
        MemRefOf<[F16, F32]>:$output_low,
        MemRefOf<[F16, F32]>:$output_high,
        MemRefOf<[F16, F32]>:$output,

        I32Attr:$levels
    );

    let extraClassDeclaration = [{
        vpux::SmallVector<mlir::Value> getInputs() {
            return {input(), input_low(), input_high(), output_low(), output_high()};
        }

        vpux::SmallVector<mlir::Value> getOutputs() {
            return {output()};
        }
    }];
}

//
// QuantizeOp
//

def IERT_QuantizeOp :
        IERT_LayerOp<
            "Quantize",
            [
                SingleInputAndResultLayer
            ]
        > {
    let summary = "InferenceEngine run-time Quantize layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[quant_QuantizedType]>:$output
    );
}

//
// DequantizeOp
//

def IERT_DequantizeOp :
        IERT_LayerOp<
            "Dequantize",
            [
                SingleInputAndResultLayer
            ]
        > {
    let summary = "InferenceEngine run-time Dequantize layer";

    let arguments = (ins
        MemRefOf<[quant_QuantizedType]>:$input,
        MemRefOf<[F16, F32]>:$output
    );
}

//
// PReluOp
//

def IERT_PReluOp :
        IERT_LayerOp<
            "PRelu",
            [
                SameOperandsElementType,
                DeclareOpInterfaceMethods<PReluLayerInterface>,
                DeclareOpInterfaceMethods<LayerInterface>
            ]
        > {
    let summary = "InferenceEngine run-time PRelu layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$negative_slope,
        MemRefOf<[F16, F32]>:$output
    );
}

//
// LeakyReluOp
//

def IERT_LeakyReluOp :
        IERT_LayerOp<
            "LeakyRelu",
            [
                SingleInputAndResultLayer,
                SameOperandsElementType,
                SameOperandsShape,
                DeclareOpInterfaceMethods<LeakyReluLayerInterface>,
                DeclareOpInterfaceMethods<LayerInterface>
            ]
        > {
    let summary = "InferenceEngine run-time LeakyRelu layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output,

        F32Attr:$negative_slope
    );
}

//
// AddOp
//

def IERT_AddOp :
        IERT_LayerOp<
            "Add",
            [
                SameOperandsElementType,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<EltwiseLayerInterface>
            ]
        > {
    let summary = "InferenceEngine run-time Add layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input1,
        MemRefOf<[F16, F32]>:$input2,
        MemRefOf<[F16, F32]>:$output
    );
}

//
// MultiplyOp
//

def IERT_MultiplyOp :
        IERT_LayerOp<
            "Multiply",
            [
                SameOperandsElementType,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<EltwiseLayerInterface>
            ]
        > {
    let summary = "InferenceEngine run-time Multiply layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input1,
        MemRefOf<[F16, F32]>:$input2,
        MemRefOf<[F16, F32]>:$output
    );
}

//
// DivideOp
//

def IERT_DivideOp :
        IERT_LayerOp<
            "Divide",
            [
                SameOperandsElementType,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<EltwiseLayerInterface>
            ]
        > {
    let summary = "InferenceEngine run-time Divide layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input1,
        MemRefOf<[F16, F32]>:$input2,
        MemRefOf<[F16, F32]>:$output
    );
}

//
// SquaredDifferenceOp
//

def IERT_SquaredDifferenceOp :
        IERT_LayerOp<
            "SquaredDifference",
            [
                SameOperandsElementType,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<EltwiseLayerInterface>
            ]
        > {
    let summary = "InferenceEngine run-time SquaredDifference layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input1,
        MemRefOf<[F16, F32]>:$input2,
        MemRefOf<[F16, F32]>:$output
    );
}

//
// PowerOp
//

def IERT_PowerOp :
        IERT_LayerOp<
            "Power",
            [
                SameOperandsElementType,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<EltwiseLayerInterface>
            ]
        > {
    let summary = "InferenceEngine run-time Power layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input1,
        MemRefOf<[F16, F32]>:$input2,
        MemRefOf<[F16, F32]>:$output
    );
}

//
// FloorModOp
//

def IERT_FloorModOp :
        IERT_LayerOp<
            "FloorMod",
            [
                SameOperandsElementType,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<EltwiseLayerInterface>
            ]
        > {
    let summary = "InferenceEngine run-time FloorMod layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input1,
        MemRefOf<[F16, F32]>:$input2,
        MemRefOf<[F16, F32]>:$output
    );
}

//
// MinimumOp
//

def IERT_MinimumOp :
        IERT_LayerOp<
            "Minimum",
            [
                SameOperandsElementType,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<EltwiseLayerInterface>
            ]
        > {
    let summary = "InferenceEngine run-time Minimum layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input1,
        MemRefOf<[F16, F32]>:$input2,
        MemRefOf<[F16, F32]>:$output
    );
}

//
// MaximumOp
//

def IERT_MaximumOp :
        IERT_LayerOp<
            "Maximum",
            [
                SameOperandsElementType,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<EltwiseLayerInterface>
            ]
        > {
    let summary = "InferenceEngine run-time Maximum layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input1,
        MemRefOf<[F16, F32]>:$input2,
        MemRefOf<[F16, F32]>:$output
    );
}

//
// GenericReshape
//

def IERT_GenericReshapeOp :
        IERT_Op<
            "GenericReshape",
            [
                NoSideEffect,
                SameOperandsAndResultElementType,
                DeclareOpInterfaceMethods<ViewLikeOpInterface>
            ]
        > {
    let summary = "InferenceEngine run-time generic Reshape layer";

    let arguments = (ins
        AnyStridedMemRef:$input
    );

    let results = (outs
        AnyStridedMemRef:$output
    );

    let verifier = [{
        return vpux::IERT::verifyOp(*this);
    }];

    let hasFolder = 1;
}

//
// DetectionOutputOp
//

def IERT_DetectionOutputOp :
        IERT_LayerOp<
            "DetectionOutput",
            [
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<DetectionOutputLayerInterface>
            ]
        > {
    let summary = "InferenceEngine run-time DetectionOutput layer";

    let arguments = (ins
        MemRefOf<[AnyFloat]>:$in_box_logits,
        MemRefOf<[AnyFloat]>:$in_class_preds,
        MemRefOf<[AnyFloat]>:$in_proposals,
        Optional<MemRefOf<[AnyFloat]>>:$in_additional_preds,
        Optional<MemRefOf<[AnyFloat]>>:$in_additional_proposals,

        MemRefOf<[AnyFloat]>:$output,

        IE_DetectionOutputAttrs:$attr
    );
}

//
// SwishOp
//

def IERT_SwishOp :
        IERT_LayerOp<
            "Swish",
            [
                SameOperandsElementType,
                DeclareOpInterfaceMethods<SwishLayerInterface>,
                DeclareOpInterfaceMethods<LayerInterface>
            ]
        > {
    let summary = "InferenceEngine run-time Swish layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        Optional<MemRefOf<[F16, F32]>>:$beta,
        MemRefOf<[F16, F32]>:$output,

        OptionalAttr<F32Attr>:$beta_value
    );
}

//
// ScaleShiftOp
//

def IERT_ScaleShiftOp :
        IERT_LayerOp<
            "ScaleShift",
            [
                AttrSizedOperandSegments,
                SameOperandsElementType,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<ScaleShiftLayerInterface>
            ]
        > {
    let summary = "InferenceEngine run-time ScaleShift layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        Optional<MemRefOf<[F16, F32]>>:$weights,
        Optional<MemRefOf<[F16, F32]>>:$biases,
        MemRefOf<[F16, F32]>:$output
    );
}

//
// GRNOp
//

def IERT_GRNOp :
        IERT_LayerOp<
            "GRN",
            [
                SingleInputAndResultLayer,
                SameOperandsElementType,
                DeclareOpInterfaceMethods<GRNLayerInterface>,
                DeclareOpInterfaceMethods<LayerInterface>
            ]
        > {
    let summary = "InferenceEngine run-time GRN layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output,

        F32Attr:$bias
    );
}

//
// TileOp
//

def IERT_TileOp :
        IERT_LayerOp<
            "Tile"
        > {
    let summary = "InferenceEngine run-time Tile layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[SI64]>:$repeats,

        MemRefOf<[F16, F32]>:$output
    );

    let extraClassDeclaration = [{
        vpux::SmallVector<mlir::Value> getInputs() {
            return {input(), repeats()};
        }

        vpux::SmallVector<mlir::Value> getOutputs() {
            return {output()};
        }
    }];
}

//
// PerAxisTileOp
//

def IERT_PerAxisTileOp :
        IERT_LayerOp<
            "PerAxisTile",
            [
                SingleInputAndResultLayer
            ]
        > {
    let summary = "InferenceEngine run-time per axis Tile layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output,

        I32Attr:$axis,
        I32Attr:$tiles
    );
}

//
// NegativeOp
//

def IERT_NegativeOp :
        IERT_LayerOp<
            "Negative",
            [
                SameOperandsElementType,
                SingleInputAndResultLayer
            ]
        > {
    let summary = "InferenceEngine run-time Negative layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output
    );
}
//
// ROIPoolingOp
//

def IERT_ROIPoolingOp :
        IERT_LayerOp<
            "ROIPooling",
            [
                DeclareOpInterfaceMethods<ROIPoolingLayerInterface>
            ]
        > {
    let summary = "InferenceEngine run-time ROIPooling layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$coords,
        MemRefOf<[F16, F32]>:$output,

        I32ArrayAttr:$output_size,
        F32Attr:$spatial_scale,
        IE_ROIPoolingMethod:$method
    );
}

//
// FullyConnectedOp
//

def IERT_FullyConnectedOp:
        IERT_LayerOp<
            "FullyConnected",
            [
                DeclareOpInterfaceMethods<FullyConnectedLayerInterface>,
                DeclareOpInterfaceMethods<LayerInterface>
            ]
        > {
    let summary = "InferenceEngine run-time FullyConnected layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$weights,
        Optional<MemRefOf<[F16, F32]>>:$bias,
        MemRefOf<[F16, F32]>:$output
    );
}

//
// TransposeOp
//

def IERT_TransposeOp :
        IERT_LayerOp<
            "Transpose",
            [
                SingleInputAndResultLayer
            ]
        > {
    let summary = "InferenceEngine run-time Transpose layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        Optional<MemRefOf<[SI64]>>:$order,
        MemRefOf<[F16, F32]>:$output,

        AffineMapAttr:$order_value
    );
}


//
// CTCGreedyDecoderOp
//

def IERT_CTCGreedyDecoderOp :
        IERT_LayerOp<
            "CTCGreedyDecoder",
            [
                DeclareOpInterfaceMethods<CTCGreedyDecoderLayerInterface>,
                DeclareOpInterfaceMethods<LayerInterface>
            ]
        > {
    let summary = "InferenceEngine run-time CTCGreedyDecoder layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$sequenceLengths,
        MemRefOf<[F16, F32]>:$output,

        UnitAttr:$mergeRepeated
    );
}

//
// CTCGreedyDecoderSeqLenOp
//

def IERT_CTCGreedyDecoderSeqLenOp :
        IERT_LayerOp<
            "CTCGreedyDecoderSeqLen",
            [
                DeclareOpInterfaceMethods<CTCGreedyDecoderSeqLenLayerInterface>,
                DeclareOpInterfaceMethods<LayerInterface>
            ]
        > {
    let summary = "InferenceEngine run-time CTCGreedyDecoderSeqLen layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[SI32]>:$sequenceLength,
        Optional<MemRefOf<[SI32]>>:$blankIndex,
        MemRefOf<[SI32]>:$output,
        MemRefOf<[SI32]>:$outputLength,

        UnitAttr:$mergeRepeated
    );
}

#endif
