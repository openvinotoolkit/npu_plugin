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

#ifndef VPUX_COMPILER_DIALECT_IE_OPS
#define VPUX_COMPILER_DIALECT_IE_OPS

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/core/ops_interfaces.td"
include "vpux/compiler/dialect/IE/dialect.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Dialect/Quant/QuantOpsBase.td"

//
// Base classes
//

class IE_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<
            IE_Dialect,
            mnemonic,
            traits
        >;

class IE_LayerOp<string mnemonic, list<OpTrait> traits = []> :
        IE_Op<
            mnemonic,
            [
                NoSideEffect,
                InferTypeOpInterface,
                DeclareOpInterfaceMethods<InferShapedTypeOpInterface, ["inferReturnTypeComponents"]>,
                DeclareOpInterfaceMethods<LayerInterface>
            ] # traits
        > {
    let extraClassDeclaration = [{
        static bool isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
            return vpux::isCompatibleShapeAndElemType(lhs, rhs);
        }
    }];

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
}

//
// CNNNetworkOp
//

def IE_CNNNetworkOp :
        IE_Op<
            "CNNNetwork",
            [
                IsolatedFromAbove,
                HasParent<"mlir::ModuleOp">,
                NoRegionArguments,
                DeclareOpInterfaceMethods<SymbolUserOpInterface>
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "InferenceEngine CNN Network description";

    let description = [{
        This operation is bound to MLIR Module and holds extra information about InferenceEngine CNN Network:

          * Precision and layout for user-provided inputs.
          * Precision and layout for user-provided outputs.
          * Entry point (Function name) for the network inference.
    }];

    let arguments = (ins
        FlatSymbolRefAttr:$entryPoint
    );

    let regions = (region
        SizedRegion<1>:$inputsInfo,
        SizedRegion<1>:$outputsInfo
    );

    let extraClassDeclaration = [{
        size_t getNetInputsCount();
        vpux::SmallVector<vpux::IE::DataInfoOp, 1> getInputsInfo();

        size_t getNetOutputsCount();
        vpux::SmallVector<vpux::IE::DataInfoOp, 1> getOutputsInfo();

        static void getFromModule(
                mlir::ModuleOp module,
                vpux::IE::CNNNetworkOp& netInfo,
                mlir::FuncOp& netFunc);
    }];

    let assemblyFormat = [{
        attr-dict
        `entryPoint` `:` $entryPoint
        `inputsInfo` `:` $inputsInfo
        `outputsInfo` `:` $outputsInfo
    }];

    let verifier = [{
        return vpux::IE::verifyOp(*this);
    }];
}

//
// DataInfoOp
//

def IE_DataInfoOp :
        IE_Op<
            "DataInfo",
            [
                IsolatedFromAbove,
                HasParent<"vpux::IE::CNNNetworkOp">
            ]
        > {
    let summary = "Information about InferenceEngine CNN Network input/output Data object";

    let description = [{
        This operation is bound to `IE.CNNNetwork` Operation and holds information about Data object:

          * Name
          * Original shape
          * User-defined precision (element type)
          * User-defined layout
    }];

    let arguments = (ins
        StrAttr:$name,
        TypeAttr:$userType
    );

    let extraClassDeclaration = [{
        vpux::DimsOrder getDimsOrder();
    }];

    let assemblyFormat = [{
        $name `:` $userType
        attr-dict
    }];

    let verifier = [{
        return vpux::IE::verifyOp(*this);
    }];
}

//
// ConvertOp
//

def IE_ConvertOp :
        IE_LayerOp<
            "Convert",
            [
                DeclareOpInterfaceMethods<CastOpInterface>,
                DeclareOpInterfaceMethods<ConvertLayerInterface>
            ]
        > {
    let summary = "InferenceEngine Convert layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        TypeAttr:$dstType
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

//
// ReorderOp
//

def IE_ReorderOp :
        IE_LayerOp<
            "Reorder",
            [
                SameOperandsShape
            ]
        > {
    let summary = "InferenceEngine Reorder layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        AffineMapAttr:$dstOrder
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;

    let extraClassDeclaration = [{
        static bool isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
            return lhs == rhs;
        }
    }];
}

//
// SoftMaxOp
//

def IE_SoftMaxOp :
        IE_LayerOp<
            "SoftMax",
            [
                ResultsAreFloatLike,
                SameOperandsAndResultType,
                DeclareOpInterfaceMethods<SoftMaxLayerInterface>
            ]
        > {
    let summary = "InferenceEngine SoftMax layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        IntAttr:$axisInd
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let hasFolder = 1;
}

//
// TileOp
//

def IE_TileOp :
        IE_LayerOp<
            "Tile"
        > {
    let summary = "InferenceEngine Tile layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[SI64]>:$repeats
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

//
// PerAxisTileOp
//

def IE_PerAxisTileOp :
        IE_LayerOp<
            "PerAxisTile"
        > {
    let summary = "InferenceEngine per axis Tile layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        IntAttr:$axis,
        IntAttr:$tiles
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}
//
// ReLUOp
//

def IE_ReLUOp :
        IE_LayerOp<
            "ReLU",
            [
                ResultsAreFloatLike,
                SameOperandsAndResultType
            ]
        > {
    let summary = "InferenceEngine ReLU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// SplitOp
//

def IE_SplitOp :
        IE_LayerOp<
            "Split"
        > {
    let summary = "InferenceEngine Split layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<AnyRankedTensor>:$axis,

        IntAttr:$num_splits,
        OptionalAttr<IntAttr>:$axis_value
    );

    let results = (outs
        Variadic<AnyRankedTensor>:$outputs
    );

    let hasCanonicalizer = 1;
}

//
// PowerOp
//

def IE_PowerOp :
        IE_LayerOp<
            "Power",
            [
                ResultsAreFloatLike
            ]
        > {
    let summary = "InferenceEngine Power layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input1,
        RankedTensorOf<[F16, F32]>:$input2,

        IE_AutoBroadcastType:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// AddOp
//

def IE_AddOp :
        IE_LayerOp<
            "Add",
            [
                Commutative,
                ResultsAreFloatLike
            ]
        > {
    let summary = "InferenceEngine Add layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input1,
        RankedTensorOf<[F16, F32]>:$input2,

        IE_AutoBroadcastType:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

//
// DivideOp
//

def IE_DivideOp :
        IE_LayerOp<
            "Divide",
            [
                ResultsAreFloatLike
            ]
        > {
    let summary = "InferenceEngine Divide layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input1,
        RankedTensorOf<[F16, F32]>:$input2,

        IE_AutoBroadcastType:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// SquaredDiffOp
//

def IE_SquaredDifferenceOp :
        IE_LayerOp<
            "SquaredDiff",
            [
                ResultsAreFloatLike
            ]
        > {
    let summary = "InferenceEngine SquaredDiff layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input1,
        RankedTensorOf<[F16, F32]>:$input2,

        IE_AutoBroadcastType:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// FloorModOp
//

def IE_FloorModOp :
        IE_LayerOp<
            "FloorMod",
            [
                ResultsAreFloatLike
            ]
        > {
    let summary = "InferenceEngine FloorMod layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input1,
        RankedTensorOf<[F16, F32]>:$input2,

        IE_AutoBroadcastType:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// MultiplyOp
//

def IE_MultiplyOp :
        IE_LayerOp<
            "Multiply",
            [
                Commutative,
                ResultsAreFloatLike
            ]
        > {
    let summary = "InferenceEngine Multiply layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input1,
        RankedTensorOf<[F16, F32]>:$input2,

        IE_AutoBroadcastType:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

//
// ConvolutionOp
//

def IE_ConvolutionOp :
        IE_LayerOp<
            "Convolution",
            [
                DeclareOpInterfaceMethods<TensorInterface_4D>,
                DeclareOpInterfaceMethods<MultiLayerInterface>,
                DeclareOpInterfaceMethods<ConvolutionLayerInterface>
            ]
        > {
    let summary = "InferenceEngine Convolution layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$input,
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$filter,
        Optional<RankedTensorOf<[F16, F32]>>:$bias,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        I64ArrayAttr:$dilations,

        OptionalAttr<IE_PostOp>:$post_op
    );

    let results = (outs
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$output
    );

    let hasCanonicalizer = 1;
}

//
// GroupConvolutionOp
//

def IE_GroupConvolutionOp :
        IE_LayerOp<
            "GroupConvolution",
            [
                ResultsAreFloatLike,
                DeclareOpInterfaceMethods<TensorInterface_4D>,
                DeclareOpInterfaceMethods<MultiLayerInterface>,
                DeclareOpInterfaceMethods<ConvolutionLayerInterface>
            ]
        > {
    let summary = "InferenceEngine GroupConvolution layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$filter,
        Optional<RankedTensorOf<[F16, F32]>>:$bias,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        I64ArrayAttr:$dilations,
        OptionalAttr<IntAttr>:$groups,

        OptionalAttr<IE_PostOp>:$post_op
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let hasCanonicalizer = 1;
}

//
// AvgPoolOp
//

def IE_AvgPoolOp :
        IE_LayerOp<
            "AvgPool",
            [
                ResultsAreFloatLike,
                DeclareOpInterfaceMethods<TensorInterface_4D>
            ]
        > {
    let summary = "InferenceEngine AvgPool layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        I64ArrayAttr:$kernel_size,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        IE_RoundingType:$rounding_type,
        UnitAttr:$exclude_pads
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// MaxPoolOp
//

def IE_MaxPoolOp :
        IE_LayerOp<
            "MaxPool",
            [
                DeclareOpInterfaceMethods<TensorInterface_4D>,
                DeclareOpInterfaceMethods<MultiLayerInterface>
            ]
        > {
    let summary = "InferenceEngine MaxPool layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$input,

        I64ArrayAttr:$kernel_size,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        IE_RoundingType:$rounding_type,

        OptionalAttr<IE_PostOp>:$post_op
    );

    let results = (outs
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$output
    );
}

//
// GatherOp
//

def IE_GatherOp :
        IE_LayerOp<
            "Gather"
        > {
    let summary = "InferenceEngine Gather layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        AnyRankedTensor:$indices,
        AnyRankedTensor:$axis
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// ClampOp
//

def IE_ClampOp :
        IE_LayerOp<
            "Clamp",
            [
                ResultsAreFloatLike,
                SameOperandsAndResultType
            ]
        > {
    let summary = "InferenceEngine Clamp layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        F64Attr:$min,
        F64Attr:$max
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// EluOp
//

def IE_EluOp :
        IE_LayerOp<
            "Elu",
            [
                ResultsAreFloatLike,
                SameOperandsAndResultType
            ]
        > {
    let summary = "InferenceEngine Elu layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        F64Attr:$x
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// ReshapeOp
//

def IE_ReshapeOp :
        IE_LayerOp<
            "Reshape",
            [
                DeclareOpInterfaceMethods<ViewLikeOpInterface>
            ]
        > {
    let summary = "InferenceEngine Reshape layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$shape,

        UnitAttr:$special_zero,
        OptionalAttr<I64ArrayAttr>:$shape_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

//
// SqueezeOp
//

def IE_SqueezeOp :
        IE_LayerOp<
            "Squeeze",
            [
                DeclareOpInterfaceMethods<ViewLikeOpInterface>
            ]
        > {
    let summary = "InferenceEngine Squeeze layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$axes_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

//
// UnsqueezeOp
//

def IE_UnsqueezeOp :
        IE_LayerOp<
            "Unsqueeze",
            [
                DeclareOpInterfaceMethods<ViewLikeOpInterface>
            ]
        > {

    let summary = "InferenceEngine Unsqueeze layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$axes_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

//
// SigmoidOp
//

def IE_SigmoidOp :
        IE_LayerOp<
            "Sigmoid",
            [
                ResultsAreFloatLike,
                SameOperandsAndResultType
            ]
        > {
    let summary = "InferenceEngine Sigmoid layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// LRNOp
//

def IE_LRNOp :
        IE_LayerOp<
            "LRN"
        > {
    let summary = "InferenceEngine LRN layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        AnyRankedTensor:$axis,

        F64Attr:$alpha,
        F64Attr:$beta,
        F64Attr:$bias,
        IntAttr:$size
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// MinimumOp
//

def IE_MinimumOp :
        IE_LayerOp<"Minimum"> {
    let summary = "InferenceEngine Minimum layer";

    let arguments = (ins
        AnyRankedTensor:$input1,
        AnyRankedTensor:$input2,

        IE_AutoBroadcastType:$auto_broadcast
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// MaximumOp
//

def IE_MaximumOp :
        IE_LayerOp<"Maximum"> {
    let summary = "InferenceEngine Maximum layer";

    let arguments = (ins
        AnyRankedTensor:$input1,
        AnyRankedTensor:$input2,

        IE_AutoBroadcastType:$auto_broadcast
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// FakeQuantizeOp
//

def IE_FakeQuantizeOp :
        IE_LayerOp<
            "FakeQuantize",
            [
                ResultsAreFloatLike
            ]
        > {
    let summary = "InferenceEngine FakeQuantize layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$input_low,
        RankedTensorOf<[F16, F32]>:$input_high,
        RankedTensorOf<[F16, F32]>:$output_low,
        RankedTensorOf<[F16, F32]>:$output_high,

        IntAttr:$levels,
        IE_AutoBroadcastType:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// MatMul
//

def IE_MatMulOp:
        IE_LayerOp<
            "MatMul",
            [
                ResultsAreFloatLike
            ]
        > {
    let summary = "InferenceEngine MatMul layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input1,
        RankedTensorOf<[F16, F32]>:$input2,

        UnitAttr:$transpose_a,
        UnitAttr:$transpose_b
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let hasCanonicalizer = 1;
}

//
// TanhOp
//

def IE_TanhOp :
        IE_LayerOp<
            "Tanh",
            [
                ResultsAreFloatLike,
                SameOperandsAndResultType
            ]
        > {
    let summary = "InferenceEngine Tanh layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// ExpOp
//

def IE_ExpOp :
        IE_LayerOp<
            "Exp",
            [
                ResultsAreFloatLike,
                SameOperandsAndResultType
            ]
        > {
    let summary = "InferenceEngine Exp layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// HSwishOp
//

def IE_HSwishOp :
        IE_LayerOp<
            "HSwish",
            [
                ResultsAreFloatLike,
                SameOperandsAndResultType
            ]
        > {
    let summary = "InferenceEngine HSwish layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// MishOp
//

def IE_MishOp :
        IE_LayerOp<
            "Mish",
            [
                ResultsAreFloatLike,
                SameOperandsAndResultType
            ]
        > {
    let summary = "InferenceEngine Mish layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// TransposeOp
//

def IE_TransposeOp :
        IE_LayerOp<
            "Transpose"
        > {
    let summary = "InferenceEngine Transpose layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<RankedTensorOf<[SI64]>>:$order,

        OptionalAttr<AffineMapAttr>:$order_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

//
// ProposalOp
//

def IE_ProposalOp :
        IE_LayerOp<
            "Proposal"
        > {
    let summary = "InferenceEngine Proposal layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$class_probs,
        RankedTensorOf<[F16, F32]>:$bbox_deltas,
        RankedTensorOf<[F16, F32]>:$image_shape,

        IE_ProposalAttrs:$proposal_attrs
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// InterpolateOp
//

def IE_InterpolateOp :
        IE_LayerOp<
            "Interpolate",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine Interpolate layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$sizes,
        Optional<RankedTensorOf<[F16, F32]>>:$scales,
        Optional<RankedTensorOf<[AnyInteger]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$sizes_attr,
        OptionalAttr<F64ArrayAttr>:$scales_attr,
        OptionalAttr<I64ArrayAttr>:$axes_attr,

        IE_InterpolateAttr:$attr
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let hasCanonicalizer = 1;
}

//
// TopKOp
//

def IE_TopKOp :
        IE_LayerOp<
            "TopK"
        > {
    let summary = "InferenceEngine TopK layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        0DTensorOf<[AnyInteger]>:$k,

        IntAttr:$axis,
        IE_TopKMode:$mode,
        IE_TopKSortType:$sort,
        TypeAttr:$element_type
    );

    let results = (outs
        AnyRankedTensor:$output_values,
        AnyRankedTensor:$target_shape
    );
}

//
// RegionYoloOp
//

def IE_RegionYoloOp :
        IE_LayerOp<
            "RegionYolo"
        > {
    let summary = "InferenceEngine RegionYolo layer";

    let arguments = (ins
        4DTensorOf<[AnyFloat]>:$input,

        IntAttr:$coords,
        IntAttr:$classes,
        IntAttr:$regions,
        BoolAttr:$do_softmax,
        I64ArrayAttr:$mask,
        IntAttr:$axis,
        IntAttr:$end_axis,
        F64ArrayAttr:$anchors
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// ReorgYoloOp
//

def IE_ReorgYoloOp :
        IE_LayerOp<
            "ReorgYolo"
        > {
    let summary = "InferenceEngine ReorgYolo layer";

    let arguments = (ins
        4DTensorOf<[AnyInteger, AnyFloat]>:$input,

        IntAttr:$stride
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// DetectionOutputOp
//

def IE_DetectionOutputOp :
        IE_LayerOp<
            "DetectionOutput",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine DetectionOutput layer";

    let arguments = (ins
        2DTensorOf<[AnyFloat]>:$in_box_logits,
        2DTensorOf<[AnyFloat]>:$in_class_preds,
        3DTensorOf<[AnyFloat]>:$in_proposals,
        Optional<2DTensorOf<[AnyFloat]>>:$in_additional_preds,
        Optional<2DTensorOf<[AnyFloat]>>:$in_additional_proposals,

        IE_DetectionOutputAttrs:$attr
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// NormalizeL2Op
//

def IE_NormalizeL2Op :
        IE_LayerOp<
            "NormalizeL2"
        > {
    let summary = "InferenceEngine NormalizeL2 layer";

    let arguments = (ins
        AnyRankedTensor:$data,
        RankedTensorOf<[AnyInteger]>:$axes,

        F64Attr:$eps,
        IE_EpsMode:$eps_mod
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// ConcatOp
//

def IE_ConcatOp :
        IE_LayerOp<
            "Concat"
        > {
    let summary = "InferenceEngine Concat layer";

    let arguments = (ins
        Variadic<AnyRankedTensor>:$inputs,

        IntAttr:$axis
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// ROIPoolingOp
//

def IE_ROIPoolingOp :
        IE_LayerOp<
            "ROIPooling",
            [
                ResultsAreFloatLike
            ]
        > {
    let summary = "InferenceEngine ROIPooling layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$coords,

        I64ArrayAttr:$output_size,
        F64Attr:$spatial_scale,
        IE_ROIPoolingMethod:$method
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// StridedSliceOp
//

def IE_StridedSliceOp :
        IE_LayerOp<
            "StridedSlice",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine StridedSlice layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<1DTensorOf<[AnyInteger]>>:$begins,
        Optional<1DTensorOf<[AnyInteger]>>:$ends,
        Optional<1DTensorOf<[AnyInteger]>>:$strides,

        OptionalAttr<I64ArrayAttr>:$begins_attr,
        OptionalAttr<I64ArrayAttr>:$ends_attr,
        OptionalAttr<I64ArrayAttr>:$strides_attr,

        I64ArrayAttr:$begin_mask,
        I64ArrayAttr:$end_mask,
        I64ArrayAttr:$new_axis_mask,
        I64ArrayAttr:$shrink_axis_mask,
        I64ArrayAttr:$ellipsis_mask
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
}

//
// DeconvolutionOp (ConvolutionBackprop in ngraph)
//

def IE_DeconvolutionOp:
        IE_LayerOp<
            "Deconvolution"
        > {
    let summary = "InferenceEngine Deconvolution layer";

    let arguments = (ins
        AnyRankedTensor:$feature,
        AnyRankedTensor:$filter,
        Optional<1DTensorOf<[AnyInteger]>>:$output_shape,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        I64ArrayAttr:$dilations,
        I64ArrayAttr:$output_padding
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// PReluOp
//

def IE_PReluOp :
        IE_LayerOp<
            "PRelu"
        > {
    let summary = "InferenceEngine PRelu layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$negative_slope
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let hasCanonicalizer = 1;
}

//
// LeakyReluOp
//

def IE_LeakyReluOp :
        IE_LayerOp<
            "LeakyRelu",
            [
                SameOperandsAndResultType
            ]
        > {
    let summary = "InferenceEngine LeakyRelu layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        F64Attr:$negative_slope
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// SwishOp
//

def IE_SwishOp :
        IE_LayerOp<
            "Swish"
        > {
    let summary = "InferenceEngine Swish layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        Optional<RankedTensorOf<[F16, F32]>>:$beta,

        OptionalAttr<F64Attr>:$beta_value
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let hasCanonicalizer = 1;
}

//
// ScaleShiftOp
//

def IE_ScaleShiftOp :
        IE_LayerOp<
            "ScaleShift",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine ScaleShift layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        Optional<RankedTensorOf<[F16, F32]>>:$weights,
        Optional<RankedTensorOf<[F16, F32]>>:$biases
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let hasCanonicalizer = 1;
}

//
// GRNOp
//

def IE_GRNOp :
        IE_LayerOp<
            "GRN",
            [
                SameOperandsAndResultType
            ]
        > {
    let summary = "InferenceEngine GRN layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        F64Attr:$bias
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// NegativeOp
//

def IE_NegativeOp :
        IE_LayerOp<
            "Negative",
            [
                SameOperandsAndResultType
            ]
        > {
    let summary = "InferenceEngine Negative layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// FullyConnected
//

def IE_FullyConnectedOp:
        IE_LayerOp<
            "FullyConnected",
            [
                ResultsAreFloatLike
            ]
        > {
    let summary = "InferenceEngine FullyConnected layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$weights,
        Optional<RankedTensorOf<[F16, F32]>>:$bias
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let hasCanonicalizer = 1;
}

//
// CTCGreedyDecoderOp
//

def IE_CTCGreedyDecoderOp :
        IE_LayerOp<
            "CTCGreedyDecoder",
            [
                ResultsAreFloatLike
            ]
        > {
    let summary = "InferenceEngine CTCGreedyDecoder layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$sequenceLengths,

        UnitAttr:$mergeRepeated
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// CTCGreedyDecoderSeqLenOp
//

def IE_CTCGreedyDecoderSeqLenOp :
        IE_LayerOp<
            "CTCGreedyDecoderSeqLen"
        > {
    let summary = "InferenceEngine CTCGreedyDecoderSeqLen layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[SI32]>:$sequenceLength,
        Optional<RankedTensorOf<[SI32]>>:$blankIndex,

        UnitAttr:$mergeRepeated
    );

    let results = (outs
        RankedTensorOf<[SI32]>:$output,
        RankedTensorOf<[SI32]>:$outputLength
    );
}

//
// PadOp
//

def IE_PadOp :
        IE_LayerOp<
            "Pad",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine Pad layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$pads_begin,
        Optional<RankedTensorOf<[AnyInteger]>>:$pads_end,
        Optional<RankedTensorOf<[AnyInteger, AnyFloat]>>:$pad_value,

        OptionalAttr<I64ArrayAttr>:$pads_begin_attr,
        OptionalAttr<I64ArrayAttr>:$pads_end_attr,
        OptionalAttr<F64Attr>:$pad_value_attr,

        IE_PadMode:$mode
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;

    let assemblyFormat = [{
        `(` $input `)` (`[` $pads_begin^ `,` $pads_end (`,` $pad_value^)? `]`)? attr-dict `:` type(operands) `->` type(results)
    }];
}

//
// ExpandOp
//

def IE_ExpandOp :
        IE_LayerOp<
            "Expand"
        > {
    let summary = "Expand tensor with uninitialized values";

    let arguments = (ins
        AnyRankedTensor:$input,

        I64ArrayAttr:$pads_begin_attr,
        I64ArrayAttr:$pads_end_attr
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// LSTMCellOp
//

def IE_LSTMCellOp :
        IE_LayerOp<
            "LSTMCell",
            [
                ResultsAreFloatLike
            ]
        > {
    let summary = "InferenceEngine LSTMCell layer";

    let arguments = (ins
        2DTensorOf<[F16, F32]>:$inputData,
        2DTensorOf<[F16, F32]>:$initialHiddenState,
        2DTensorOf<[F16, F32]>:$initialCellState,
        2DTensorOf<[F16, F32]>:$weights,
        2DTensorOf<[F16, F32]>:$reccurenceWeights,
        1DTensorOf<[F16, F32]>:$biases,

        IntAttr:$hiddenSize
    );

    let results = (outs
        2DTensorOf<[F16, F32]>:$outputHiddenState,
        2DTensorOf<[F16, F32]>:$outputCellState
    );
}
#endif
