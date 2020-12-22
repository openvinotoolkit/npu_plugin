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

#ifndef VPUX_COMPILER_DIALECT_IE_OPS
#define VPUX_COMPILER_DIALECT_IE_OPS

include "vpux/compiler/core/ops_interfaces.td"
include "vpux/compiler/dialect/IE/dialect.td"
include "vpux/compiler/dialect/IE/attributes.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

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
            !listconcat(
                traits,
                [
                    NoSideEffect,
                    NativeOpTrait<"InferTensorType">,
                    InferTypeOpInterface,
                    DeclareOpInterfaceMethods<InferShapedTypeOpInterface>
                ]
            )
        > {
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
                SingleBlockImplicitTerminator<"vpux::IE::EndOp">,
                NoRegionArguments,
                DeclareOpInterfaceMethods<SymbolUserOpInterface>
            ]
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
        vpux::SmallVector<vpux::IE::DataInfoOp, 1> getInputsInfo();
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
// EndOp
//

def IE_EndOp :
        IE_Op<
            "End",
            [
                NoSideEffect,
                Terminator,
                NativeOpTrait<"ReturnLike">,
                HasParent<"vpux::IE::CNNNetworkOp">
            ]
        > {
    let summary = "End indicator for IE CNNNetwork inputs/outputs info lists";

    let assemblyFormat = "attr-dict";
}

//
// ConvertOp
//

def IE_ConvertOp :
        IE_LayerOp<
            "Convert",
            [
                SingleInputAndResultLayer,
                DeclareOpInterfaceMethods<LayerInterface>,
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
// SoftMaxOp
//

def IE_SoftMaxOp :
        IE_LayerOp<
            "SoftMax",
            [
                ResultsAreFloatLike,
                SameOperandsAndResultType,
                SingleInputAndResultLayer,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<SoftMaxLayerInterface>
            ]
        > {
    let summary = "InferenceEngine SoftMax layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        I32Attr:$axisInd
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
        AnyRankedTensor:$repeats
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
                SameOperandsAndResultType,
                SingleInputAndResultLayer,
                DeclareOpInterfaceMethods<LayerInterface>
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
        AnyRankedTensor:$axis,

        I32Attr:$num_splits
    );

    let results = (outs
        Variadic<AnyRankedTensor>:$outputs
    );
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
}

//
// MultiplyOp
//

def IE_MultiplyOp :
        IE_LayerOp<
            "Multiply",
            [
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
}

//
// ConvolutionOp
//

def IE_ConvolutionOp :
        IE_LayerOp<
            "Convolution",
            [
                ResultsAreFloatLike,
                DeclareOpInterfaceMethods<ConvolutionLayerInterface>,
                DeclareOpInterfaceMethods<LayerInterface>
            ]
        > {
    let summary = "InferenceEngine Convolution layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$filter,
        Optional<RankedTensorOf<[F16, F32]>>:$bias,

        I32ArrayAttr:$strides,
        I32ArrayAttr:$pads_begin,
        I32ArrayAttr:$pads_end,
        I32ArrayAttr:$dilations
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
                SingleInputAndResultLayer,
                DeclareOpInterfaceMethods<LayerInterface>
            ]
        > {
    let summary = "InferenceEngine AvgPool layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        I32ArrayAttr:$kernel_size,
        I32ArrayAttr:$strides,
        I32ArrayAttr:$pads_begin,
        I32ArrayAttr:$pads_end,
        IE_RoundingType:$rounding_type
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
                ResultsAreFloatLike,
                SingleInputAndResultLayer,
                DeclareOpInterfaceMethods<LayerInterface>
            ]
        > {
    let summary = "InferenceEngine MaxPool layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        I32ArrayAttr:$kernel_size,
        I32ArrayAttr:$strides,
        I32ArrayAttr:$pads_begin,
        I32ArrayAttr:$pads_end,
        IE_RoundingType:$rounding_type
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
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
                SameOperandsAndResultType,
                SingleInputAndResultLayer,
                DeclareOpInterfaceMethods<LayerInterface>
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
                SameOperandsAndResultType,
                SingleInputAndResultLayer,
                DeclareOpInterfaceMethods<LayerInterface>
            ]
        > {
    let summary = "InferenceEngine Elu layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        F64Attr:$alpha
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
            "Reshape"
        > {
    let summary = "InferenceEngine Reshape layer";

    let arguments = (ins
        AnyRankedTensor:$input1,
        1DTensorOf<[AnyInteger]>:$input2,

        UnitAttr:$special_zero
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// SqueezeOp
//

def IE_SqueezeOp :
        IE_LayerOp<
            "Squeeze"
        > {
    let summary = "InferenceEngine Squeeze layer";

    let arguments = (ins
        AnyRankedTensor:$input1,
        1DTensorOf<[AnyInteger]>:$input2
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// SigmoidOp
//

def IE_SigmoidOp :
        IE_LayerOp<
            "Sigmoid",
            [
                ResultsAreFloatLike,
                SameOperandsAndResultType,
                SingleInputAndResultLayer,
                DeclareOpInterfaceMethods<LayerInterface>
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
// ClampOp
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
        I32Attr:$size
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// UnsqueezeOp
//

def IE_UnsqueezeOp :
        IE_LayerOp<
            "Unsqueeze"
        > {

    let summary = "InferenceEngine Unsqueeze layer";

    let arguments = (ins
        AnyRankedTensor:$input1,
        1DTensorOf<[AnyInteger]>:$input2
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// MinimumOp
//

def IE_MinimumOp :
        IE_LayerOp<
            "Minimum"
        > {
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
        IE_LayerOp<
            "Maximum"
        > {
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

        I32Attr:$levels,
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
    let summary = "InferenceEngine Power layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input1,
        RankedTensorOf<[F16, F32]>:$input2,

        UnitAttr:$transpose_a,
        UnitAttr:$transpose_b
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
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
// TransposeOp
//

def IE_TransposeOp :
        IE_LayerOp<
            "Transpose"
        > {
    let summary = "InferenceEngine Transpose layer";

    let arguments = (ins
        AnyRankedTensor:$input1,
        RankedTensorOf<[AnyInteger]>:$input2
    );

    let results = (outs
        AnyRankedTensor:$output
    );
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
            "Interpolate"
        > {
    let summary = "InferenceEngine Interpolate layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        1DTensorOf<[AnyInteger]>:$target_shape,

        IE_InterpolateAttr:$attr
    );

    let results = (outs
        AnyRankedTensor:$output
    );
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

        I64Attr:$axis,
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

        I32Attr:$coord,
        I32Attr:$classes,
        I32Attr:$regions,
        BoolAttr:$do_softmax,
        I64ArrayAttr:$mask,
        I32Attr:$axis,
        I32Attr:$end_axis,
        F32ArrayAttr:$anchors
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

        I32Attr:$stride
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
        2DTensorOf<[AnyFloat]>:$box_logits,
        2DTensorOf<[AnyFloat]>:$class_preds,
        3DTensorOf<[AnyFloat]>:$proposals,
        Optional<2DTensorOf<[AnyFloat]>>:$additional_preds,
        Optional<2DTensorOf<[AnyFloat]>>:$additional_poposals,

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

        F32Attr:$eps,
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
        Variadic<AnyRankedTensor>:$input,

        I32Attr:$axis
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

#endif
