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
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/IE/dialect.td"

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
                    DeclareOpInterfaceMethods<InferShapedTypeOpInterface>,
                    DeclareOpInterfaceMethods<LayerInterface>
                ]
            )
        >;

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

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];

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

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];

    let hasCanonicalizer = 1;
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

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
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

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
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
        Variadic<AnyRankedTensor>:$output_tensors
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
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

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
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

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
}

//
// MaxPoolOp
//
def IE_MaxPoolOp :
        IE_LayerOp<
            "MaxPool",
            [
                ResultsAreFloatLike,
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

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
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

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
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

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
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

        F64Attr:$alpha    
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
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

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
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
        TensorOf<[AnyInteger]>:$input2
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
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

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
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

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
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
        TensorOf<[AnyInteger]>:$input2
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
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

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
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

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
}

#endif
