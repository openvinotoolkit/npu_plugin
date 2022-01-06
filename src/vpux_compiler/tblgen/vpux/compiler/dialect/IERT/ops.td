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

#ifndef VPUX_COMPILER_DIALECT_IERT_OPS
#define VPUX_COMPILER_DIALECT_IERT_OPS

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/core/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/IERT/dialect.td"
include "vpux/compiler/dialect/IERT/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"

include "mlir/Dialect/Quant/QuantOpsBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/OpAsmInterface.td"
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

class IERT_LayerOp<int numResult, string mnemonic, list<OpTrait> traits = []> :
        IERT_Op<
            mnemonic,
            !listconcat(
                traits,
                [
                    InferTypeOpInterface,
                    DeclareOpInterfaceMethods<IERT_LayerOpInterface>,
                    MemoryEffectsOpInterface
                ]
            )
        > {
    let extraClassDeclaration = IERT_InferLayerReturnTypes<numResult>.impl;
}

//
// StaticAllocOp
//

def IERT_StaticAllocOp :
        IERT_Op<
            "StaticAlloc",
            [
                NoSideEffect,
                DeclarationOp,
                DotInterface
            ]
        > {
    let summary = "InferenceEngine run-time static buffer allocation";

    let arguments = (ins
        IntAttr:$offset
    );

    let results = (outs
        AnyMemRef:$memory
    );

    let extraClassDeclaration = [{
        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::ORANGE;
        }
    }];

    let assemblyFormat = [{
        `<` $offset `>` attr-dict `->` type(results)
    }];
}

//
// TimestampOp
//

def IERT_TimestampOp :
        IERT_LayerOp<1, "Timestamp",
            [
                ViewLikeOpInterface,
                SameOperandsAndResultElementType,
                DotInterface
            ]
        > {
    let summary = "Get timer timestamp operation";

    let description = [{
        Get timer timestamp operation
    }];

    let arguments = (ins
        MemRefOf<[UI32, UI64]>:$output_buff
    );

    let results = (outs
        MemRefOf<[UI32, UI64]>:$output
    );

    let extraClassDeclaration = [{
        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::BLUE;
        }
    }] # IERT_InferLayerReturnTypes<1>.impl;

    let assemblyFormat = [{
        attr-dict
        `(` $output_buff `:` type($output_buff) `)`
        `->` type($output)
    }];
}

//
// ConvertOp
//

def IERT_ConvertOp :
        IERT_LayerOp<1, "Convert",
            [
                SameOperandsShape,
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time Convert layer";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output_buff
    );

    let results = (outs
        AnyMemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// CopyOp
//

def IERT_CopyOp :
        IERT_LayerOp<1, "Copy",
            [
                ViewLikeOpInterface,
                DotInterface
            ]
        > {
    let summary = "InferenceEngine run-time Copy layer";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output_buff
    );

    let results = (outs
        AnyMemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::GREEN;
        }
    }] # IERT_InferLayerReturnTypes<1>.impl;
}

//
// SoftMaxOp
//

def IERT_SoftMaxOp :
        IERT_LayerOp<1, "SoftMax",
            [
                SameOperandsShape,
                SameOperandsElementType,
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time SoftMax layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output_buff,

        IntAttr:$axisInd
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// AvgPoolOp
//

def IERT_AvgPoolOp :
        IERT_LayerOp<1, "AvgPool",
            [
                ViewLikeOpInterface,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time AvgPool layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output_buff,

        I64ArrayAttr:$kernel_size,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        UnitAttr:$exclude_pads
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// MaxPoolOp
//

def IERT_MaxPoolOp :
        IERT_LayerOp<1, "MaxPool",
            [
                ViewLikeOpInterface,
            ]
        > {
    let summary = "InferenceEngine run-time MaxPool layer";

    let arguments = (ins
        MemRefOf<[F16, F32, quant_QuantizedType]>:$input,
        MemRefOf<[F16, F32, quant_QuantizedType]>:$output_buff,

        I64ArrayAttr:$kernel_size,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,

        OptionalAttr<IE_PostOp>:$post_op
    );

    let results = (outs
        MemRefOf<[F16, F32, quant_QuantizedType]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// ConvolutionOp
//

def IERT_ConvolutionOp :
        IERT_LayerOp<1, "Convolution",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time Convolution layer";

    let arguments = (ins
        MemRefOf<[F16, F32, quant_QuantizedType]>:$input,
        MemRefOf<[F16, F32, quant_QuantizedType]>:$filter,
        Optional<MemRefOf<[F16, F32]>>:$bias,
        MemRefOf<[F16, F32, quant_QuantizedType]>:$output_buff,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        I64ArrayAttr:$dilations,

        OptionalAttr<IE_PostOp>:$post_op
    );

    let results = (outs
        MemRefOf<[F16, F32, quant_QuantizedType]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $filter `:` type($filter) (`,` $bias^ `:` type($bias))? `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// GroupConvolutionOp
//

def IERT_GroupConvolutionOp :
        IERT_LayerOp<1, "GroupConvolution",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time GroupConvolution layer";

    let arguments = (ins
        MemRefOf<[F16, F32, quant_QuantizedType]>:$input,
        MemRefOf<[F16, F32, quant_QuantizedType]>:$filter,
        Optional<MemRefOf<[F16, F32]>>:$bias,
        MemRefOf<[F16, F32, quant_QuantizedType]>:$output_buff,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        I64ArrayAttr:$dilations,
        OptionalAttr<IntAttr>:$groups,

        OptionalAttr<IE_PostOp>:$post_op
    );

    let results = (outs
        MemRefOf<[F16, F32, quant_QuantizedType]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $filter `:` type($filter) (`,` $bias^ `:` type($bias))? `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// ReLUOp
//

def IERT_ReLUOp :
        IERT_LayerOp<1, "ReLU",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time ReLU layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// SigmoidOp
//

def IERT_SigmoidOp :
        IERT_LayerOp<1, "Sigmoid",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time Sigmoid layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// LRN_IEOp
//

def IERT_LRN_IEOp :
        IERT_LayerOp<1, "LRN_IE",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time LRN_IE layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,

        MemRefOf<[F16, F32]>:$output_buff,

        F64Attr:$alpha,
        F64Attr:$beta,
        F64Attr:$bias,
        IntAttr:$size,
        IE_LRN_IERegion:$region
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `ouputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// ReduceMaxOp
//

def IERT_ReduceMaxOp :
        IERT_LayerOp<1, "ReduceMax",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time reduceMax layer";

    let arguments = (ins
        AnyMemRef:$input,
        MemRefOf<[SI64, SI32]>:$axes,

        AnyMemRef:$output_buff,

        UnitAttr:$keep_dims
    );

    let results = (outs
        AnyMemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $axes `:` type($axes) `)`
        `ouputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// ReduceMeanOp
//

def IERT_ReduceMeanOp :
    IERT_LayerOp<1, "ReduceMean",
        [
            ViewLikeOpInterface
        ]
    > {
    let summary = "InferenceEngine run-time ReduceMean layer";

    let arguments = (ins
        AnyMemRef:$input,
        MemRefOf<[SI32, SI64]>:$axes,
        AnyMemRef:$output_buff,

        UnitAttr:$keep_dims
    );

    let results = (outs
        AnyMemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $axes `:` type($axes) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// ReduceSumOp
//

def IERT_ReduceSumOp :
        IERT_LayerOp<1, "ReduceSum",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time ReduceSum layer";

    let arguments = (ins
        AnyMemRef:$input,
        MemRefOf<[SI64, SI32]>:$axes,

        AnyMemRef:$output_buff,

        UnitAttr:$keep_dims
    );

    let results = (outs
        AnyMemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $axes `:` type($axes) `)`
        `ouputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// ClampOp
//

def IERT_ClampOp :
        IERT_LayerOp<1, "Clamp",
            [
                ViewLikeOpInterface,
                SameOperandsShape,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time Clamp layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output_buff,

        F64Attr:$min,
        F64Attr:$max
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// EluOp
//

def IERT_EluOp :
        IERT_LayerOp<1, "Elu",
            [
                ViewLikeOpInterface,
                SameOperandsShape,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time Elu layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output_buff,

        F64Attr:$x
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// HSwishOp
//

def IERT_HSwishOp :
        IERT_LayerOp<1, "HSwish",
            [
                ViewLikeOpInterface,
                SameOperandsShape,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time HSwish layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// FloorOp
//

def IERT_FloorOp :
        IERT_LayerOp<1, "Floor",
            [
                ViewLikeOpInterface,
                SameOperandsShape,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time Floor layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// RoundOp
//

def IERT_RoundOp :
        IERT_LayerOp<1, "Round",
            [
                ViewLikeOpInterface,
                SameOperandsShape,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time Round layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output_buff,

        IE_RoundMode:$mode
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// MishOp
//

def IERT_MishOp :
        IERT_LayerOp<1, "Mish",
            [
                ViewLikeOpInterface,
                SameOperandsShape,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time Mish layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// ErfOp
//

def IERT_ErfOp :
        IERT_LayerOp<1, "Erf",
            [
                ViewLikeOpInterface,
                SameOperandsShape,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngie run-time Erf layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// BroadcastOp
//

def IERT_BroadcastOp :
        IERT_LayerOp<1, "Broadcast",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time Broadcast layer";

    let arguments = (ins
        AnyMemRef:$input,
        MemRefOf<[SI64, SI32]>:$target_shape,
        Optional<MemRefOf<[SI64, SI32]>>:$axes_mapping,

        AnyMemRef:$output_buff,

        OptionalAttr<IE_BroadcastType>:$mode
    );

    let results = (outs
        AnyMemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $target_shape `:` type($target_shape) (`,` $axes_mapping^ `:` type($axes_mapping))? `)`
        `ouputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// TanhOp
//

def IERT_TanhOp :
        IERT_LayerOp<1, "Tanh",
            [
                ViewLikeOpInterface,
                SameOperandsShape,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time Tanh layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// SqrtOp
//

def IERT_SqrtOp :
        IERT_LayerOp<1, "Sqrt",
            [
                ViewLikeOpInterface,
                SameOperandsShape,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time Sqrt layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// SinhOp
//

def IERT_SinhOp :
        IERT_LayerOp<1, "Sinh",
            [
                ViewLikeOpInterface,
                SameOperandsShape,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time Sinh layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// CoshOp
//

def IERT_CoshOp :
        IERT_LayerOp<1, "Cosh",
            [
                ViewLikeOpInterface,
                SameOperandsShape,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time Cosh layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// AsinhOp
//

def IERT_AsinhOp :
        IERT_LayerOp<1, "Asinh",
            [
                ViewLikeOpInterface,
                SameOperandsShape,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time Asinh layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// AcoshOp
//

def IERT_AcoshOp :
        IERT_LayerOp<1, "Acosh",
            [
                ViewLikeOpInterface,
                SameOperandsShape,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time Acosh layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// LogOp
//

def IERT_LogOp :
        IERT_LayerOp<1, "Log",
            [
                ViewLikeOpInterface,
                SameOperandsShape,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time Log layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// GeluOp
//

def IERT_GeluOp :
        IERT_LayerOp<1, "Gelu",
            [
                ViewLikeOpInterface,
                SameOperandsShape,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time Gelu layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// ExpOp
//

def IERT_ExpOp :
        IERT_LayerOp<1, "Exp",
            [
                ViewLikeOpInterface,
                SameOperandsShape,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time Exp layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// FakeQuantizeOp
//

def IERT_FakeQuantizeOp :
        IERT_LayerOp<1, "FakeQuantize",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine FakeQuantize layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$input_low,
        MemRefOf<[F16, F32]>:$input_high,
        MemRefOf<[F16, F32]>:$output_low,
        MemRefOf<[F16, F32]>:$output_high,
        MemRefOf<[F16, F32]>:$output_buff,

        IntAttr:$levels
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $input_low `:` type($input_low) `,` $input_high `:` type($input_high) `,` $output_low `:` type($output_low) `,` $output_high `:` type($output_high) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// QuantizeOp
//

def IERT_QuantizeOp :
        IERT_LayerOp<1, "Quantize",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time Quantize layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[quant_QuantizedType]>:$output_buff
    );

    let results = (outs
        MemRefOf<[quant_QuantizedType]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// DequantizeOp
//

def IERT_DequantizeOp :
        IERT_LayerOp<1, "Dequantize",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time Dequantize layer";

    let arguments = (ins
        MemRefOf<[quant_QuantizedType]>:$input,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// QuantizeCastOp
//

def IERT_QuantizeCastOp :
        IERT_Op<
            "QuantizeCast",
            [
                NoSideEffect,
                DeclareOpInterfaceMethods<ViewLikeOpInterface>
            ]
        > {
    let summary = "InferenceEngine run-time Quantize Cast layer";

    let arguments = (ins
        MemRefOf<[SI8, UI8, quant_QuantizedType]>:$input
    );

    let results = (outs
        MemRefOf<[SI8, UI8, quant_QuantizedType]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];

    let hasFolder = 1;
}

//
// PReluOp
//

def IERT_PReluOp :
        IERT_LayerOp<1, "PRelu",
            [
                ViewLikeOpInterface,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time PRelu layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$negative_slope,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $negative_slope `:` type($negative_slope) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// LeakyReluOp
//

def IERT_LeakyReluOp :
        IERT_LayerOp<1, "LeakyRelu",
            [
                ViewLikeOpInterface,
                SameOperandsElementType,
                SameOperandsShape
            ]
        > {
    let summary = "InferenceEngine run-time LeakyRelu layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output_buff,

        F64Attr:$negative_slope
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// AddOp
//

def IERT_AddOp :
        IERT_LayerOp<1, "Add",
            [
                ViewLikeOpInterface,
            ]
        > {
    let summary = "InferenceEngine run-time Add layer";

    let arguments = (ins
        MemRefOf<[F16, F32, quant_QuantizedType]>:$input1,
        MemRefOf<[F16, F32, quant_QuantizedType]>:$input2,
        MemRefOf<[F16, F32, quant_QuantizedType]>:$output_buff,

        OptionalAttr<IE_PostOp>:$post_op
    );

    let results = (outs
        MemRefOf<[F16, F32, quant_QuantizedType]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input1 `:` type($input1) `,` $input2 `:` type($input2) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// MultiplyOp
//

def IERT_MultiplyOp :
        IERT_LayerOp<1, "Multiply",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time Multiply layer";

    let arguments = (ins
        MemRefOf<[F16, F32, quant_QuantizedType]>:$input1,
        MemRefOf<[F16, F32, quant_QuantizedType]>:$input2,
        MemRefOf<[F16, F32, quant_QuantizedType]>:$output_buff,

        OptionalAttr<IE_PostOp>:$post_op
    );

    let results = (outs
        MemRefOf<[F16, F32, quant_QuantizedType]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input1 `:` type($input1) `,` $input2 `:` type($input2) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// AndOp
//

def IERT_AndOp :
        IERT_LayerOp<1, "And",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time And layer";

    let arguments = (ins
        MemRefOf<[F16, F32, quant_QuantizedType]>:$input1,
        MemRefOf<[F16, F32, quant_QuantizedType]>:$input2,
        MemRefOf<[F16, F32, quant_QuantizedType]>:$output_buff,

        OptionalAttr<IE_PostOp>:$post_op
    );

    let results = (outs
        MemRefOf<[F16, F32, quant_QuantizedType]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input1 `:` type($input1) `,` $input2 `:` type($input2) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// DivideOp
//

def IERT_DivideOp :
        IERT_LayerOp<1, "Divide",
            [
                ViewLikeOpInterface,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time Divide layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input1,
        MemRefOf<[F16, F32]>:$input2,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input1 `:` type($input1) `,` $input2 `:` type($input2) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// SquaredDifferenceOp
//

def IERT_SquaredDifferenceOp :
        IERT_LayerOp<1, "SquaredDifference",
            [
                ViewLikeOpInterface,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time SquaredDifference layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input1,
        MemRefOf<[F16, F32]>:$input2,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input1 `:` type($input1) `,` $input2 `:` type($input2) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// PowerOp
//

def IERT_PowerOp :
        IERT_LayerOp<1, "Power",
            [
                ViewLikeOpInterface,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time Power layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input1,
        MemRefOf<[F16, F32]>:$input2,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input1 `:` type($input1) `,` $input2 `:` type($input2) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// FloorModOp
//

def IERT_FloorModOp :
        IERT_LayerOp<1, "FloorMod",
            [
                ViewLikeOpInterface,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time FloorMod layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input1,
        MemRefOf<[F16, F32]>:$input2,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input1 `:` type($input1) `,` $input2 `:` type($input2) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// LessOp
//

def IERT_LessOp :
        IERT_LayerOp<1, "Less",
            [
                ViewLikeOpInterface,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time Less layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input1,
        MemRefOf<[F16, F32]>:$input2,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input1 `:` type($input1) `,` $input2 `:` type($input2) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// LessEqualOp
//

def IERT_LessEqualOp :
        IERT_LayerOp<1, "LessEqual",
            [
                ViewLikeOpInterface,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time LessEqual layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input1,
        MemRefOf<[F16, F32]>:$input2,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input1 `:` type($input1) `,` $input2 `:` type($input2) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// GreaterOp
//

def IERT_GreaterOp :
        IERT_LayerOp<1, "Greater",
            [
                ViewLikeOpInterface,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time Greater layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input1,
        MemRefOf<[F16, F32]>:$input2,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input1 `:` type($input1) `,` $input2 `:` type($input2) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// GreaterEqualOp
//

def IERT_GreaterEqualOp :
        IERT_LayerOp<1, "GreaterEqual",
            [
                ViewLikeOpInterface,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time GreaterEqual layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input1,
        MemRefOf<[F16, F32]>:$input2,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input1 `:` type($input1) `,` $input2 `:` type($input2) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// LogicalOrOp
//

def IERT_LogicalOrOp :
        IERT_LayerOp<1, "LogicalOr",
        [
                ViewLikeOpInterface,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time LogicalOr layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input1,
        MemRefOf<[F16, F32]>:$input2,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input1 `:` type($input1) `,` $input2 `:` type($input2) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// LogicalXorOp
//

def IERT_LogicalXorOp :
        IERT_LayerOp<1, "LogicalXor",
        [
                ViewLikeOpInterface,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time LogicalXor layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input1,
        MemRefOf<[F16, F32]>:$input2,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input1 `:` type($input1) `,` $input2 `:` type($input2) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// MinimumOp
//

def IERT_MinimumOp :
        IERT_LayerOp<1, "Minimum",
            [
                ViewLikeOpInterface,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time Minimum layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input1,
        MemRefOf<[F16, F32]>:$input2,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input1 `:` type($input1) `,` $input2 `:` type($input2) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// MaximumOp
//

def IERT_MaximumOp :
        IERT_LayerOp<1, "Maximum",
            [
                ViewLikeOpInterface,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time Maximum layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input1,
        MemRefOf<[F16, F32]>:$input2,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input1 `:` type($input1) `,` $input2 `:` type($input2) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
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
        AnyMemRef:$input
    );

    let results = (outs
        AnyMemRef:$output
    );

    let verifier = [{
        return vpux::IERT::verifyOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];

    let hasFolder = 1;
}

//
// DetectionOutputOp
//

def IERT_DetectionOutputOp :
        IERT_LayerOp<1, "DetectionOutput",
            [
                ViewLikeOpInterface,
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine run-time DetectionOutput layer";

    let arguments = (ins
        MemRefOf<[AnyFloat]>:$in_box_logits,
        MemRefOf<[AnyFloat]>:$in_class_preds,
        MemRefOf<[AnyFloat]>:$in_proposals,
        Optional<MemRefOf<[AnyFloat]>>:$in_additional_preds,
        Optional<MemRefOf<[AnyFloat]>>:$in_additional_proposals,

        MemRefOf<[AnyFloat]>:$output_buff,

        IE_DetectionOutputAttrs:$attr
    );

    let results = (outs
        MemRefOf<[AnyFloat]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $in_box_logits `:` type($in_box_logits) `,` $in_class_preds `:` type($in_class_preds) `,` $in_proposals `:` type($in_proposals) (`,` $in_additional_preds^ `:` type($in_additional_preds))? (`,` $in_additional_proposals^ `:` type($in_additional_proposals))? `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// SwishOp
//

def IERT_SwishOp :
        IERT_LayerOp<1, "Swish",
            [
                ViewLikeOpInterface,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time Swish layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        Optional<MemRefOf<[F16, F32]>>:$beta,
        MemRefOf<[F16, F32]>:$output_buff,

        OptionalAttr<F64Attr>:$beta_value
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) (`,` $beta^ `:` type($beta))? `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// ScaleShiftOp
//

def IERT_ScaleShiftOp :
        IERT_LayerOp<1, "ScaleShift",
            [
                ViewLikeOpInterface,
                AttrSizedOperandSegments,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time ScaleShift layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        Optional<MemRefOf<[F16, F32]>>:$weights,
        Optional<MemRefOf<[F16, F32]>>:$biases,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) (`,` $weights^ `:` type($weights))? (`,` $biases^ `:` type($biases))? `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// UpsamplingOp
//

def IERT_UpsamplingOp :
        IERT_LayerOp<1, "Upsampling",
            [
                ViewLikeOpInterface,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time Upsampling layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output_buff,
        I64ArrayAttr:$upsampling_factor,
        I64ArrayAttr:$pad_l,
        I64ArrayAttr:$pad_r
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// GRNOp
//

def IERT_GRNOp :
        IERT_LayerOp<1, "GRN",
            [
                ViewLikeOpInterface,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time GRN layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output_buff,

        F64Attr:$bias
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// TileOp
//

def IERT_TileOp :
        IERT_LayerOp<1, "Tile",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time Tile layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[SI64]>:$repeats,

        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $repeats `:` type($repeats) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// PerAxisTileOp
//

def IERT_PerAxisTileOp :
        IERT_LayerOp<1, "PerAxisTile",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time per axis Tile layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output_buff,

        IntAttr:$axis,
        IntAttr:$tiles
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// NegativeOp
//

def IERT_NegativeOp :
        IERT_LayerOp<1, "Negative",
            [
                ViewLikeOpInterface,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time Negative layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// SignOp
//

def IERT_SignOp :
        IERT_LayerOp<1, "Sign",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time Sign layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// ProposalOp
//

def IERT_ProposalOp :
        IERT_LayerOp<1, "Proposal",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time Proposal layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$class_probs,
        MemRefOf<[F16, F32]>:$bbox_deltas,
        MemRefOf<[F16, F32]>:$image_shape,
        MemRefOf<[F16, F32]>:$output_buff,

        IE_ProposalAttrs:$proposal_attrs
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $class_probs `:` type($class_probs) `,` $bbox_deltas `:` type($bbox_deltas) `,` $image_shape `:` type($image_shape) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// ROIPoolingOp
//

def IERT_ROIPoolingOp :
        IERT_LayerOp<1, "ROIPooling",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time ROIPooling layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$coords,
        MemRefOf<[F16, F32]>:$output_buff,

        I64ArrayAttr:$output_size,
        F64Attr:$spatial_scale,
        IE_ROIPoolingMethod:$method
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $coords `:` type($coords) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// ROIAlignOp
//

def IERT_ROIAlignOp :
        IERT_LayerOp<1, "ROIAlign",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time ROIAlign layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$coords,
        MemRefOf<[AnyInteger]>:$roisIdx,
        MemRefOf<[F16, F32]>:$output_buff,

        IntAttr:$pooled_h,
        IntAttr:$pooled_w,
        IntAttr:$sampling_ratio,
        F64Attr:$spatial_scale,
        IE_ROIAlignMethod:$poolingMode
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $coords `:` type($coords) `,` $roisIdx `:` type($roisIdx) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// FullyConnectedOp
//

def IERT_FullyConnectedOp:
        IERT_LayerOp<1, "FullyConnected",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time FullyConnected layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$weights,
        Optional<MemRefOf<[F16, F32]>>:$bias,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $weights `:` type($weights) (`,` $bias^ `:` type($bias))? `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// ConcatViewOp
//

def IERT_ConcatViewOp :
        IERT_Op<"ConcatView",
            [
                ViewLikeOpInterface,
                InferTypeOpInterface,
                NoSideEffect
            ]
        > {
    let summary = "InferenceEngine run-time ConcatView layer. Dummy operation to maintain use-def chains.";

    let arguments = (ins
        Variadic<AnyMemRef>:$inputs,
        AnyMemRef:$output_buff
    );

    let results = (outs
        AnyMemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $inputs `:` type($inputs) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        mlir::Value getViewSource() {
            return output_buff();
        }
    }] # IERT_InferLayerReturnTypes<1>.impl;
}

//
// CTCGreedyDecoderOp
//

def IERT_CTCGreedyDecoderOp :
        IERT_LayerOp<1, "CTCGreedyDecoder",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time CTCGreedyDecoder layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$sequenceLengths,
        MemRefOf<[F16, F32]>:$output_buff,

        UnitAttr:$mergeRepeated
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $sequenceLengths `:` type($sequenceLengths) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// CTCGreedyDecoderSeqLenOp
//

def IERT_CTCGreedyDecoderSeqLenOp :
        IERT_LayerOp<2,"CTCGreedyDecoderSeqLen",
            [
                MultiViewOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time CTCGreedyDecoderSeqLen layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[SI32]>:$sequenceLength,
        Optional<MemRefOf<[SI32]>>:$blankIndex,
        MemRefOf<[SI32]>:$output_buff,
        MemRefOf<[SI32]>:$outputLength_buff,

        UnitAttr:$mergeRepeated
    );

    let results = (outs
        MemRefOf<[SI32]>:$output,
        MemRefOf<[SI32]>:$outputLength
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $sequenceLength `:` type($sequenceLength) (`,` $blankIndex^ `:` type($blankIndex))? `)`
        `outputs` `(` $output_buff `:` type($output_buff) `,` $outputLength_buff `:` type($outputLength_buff) `)`
        `->` type(results)
    }];
}

//
// PadOp
//

def IERT_PadOp :
        IERT_LayerOp<1,"Pad",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time Pad layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output_buff,

        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        OptionalAttr<F64Attr>:$pad_value,
        IE_PadMode:$mode
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// InterpolateOp
//

def IERT_InterpolateOp :
        IERT_LayerOp<1, "Interpolate",
        [
            ViewLikeOpInterface
        ]
        > {
    let summary = "InferenceEngine run-time Interpolate layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output_buff,

        IE_InterpolateMode:$mode,
        IE_InterpolateCoordMode:$coord_mode,
        IE_InterpolateNearestMode:$nearest_mode,
        UnitAttr:$antialias
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// StridedSliceOp
//

def IERT_StridedSliceOp :
        IERT_LayerOp<1, "StridedSlice",
            [
                ViewLikeOpInterface,
            ]
        > {
    let summary = "InferenceEngine run-time StridedSlice layer";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output_buff,

        I64ArrayAttr:$begins,
        I64ArrayAttr:$ends,
        I64ArrayAttr:$strides
    );

    let results = (outs
        AnyMemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// GatherOp
//

def IERT_GatherOp :
        IERT_LayerOp<1, "Gather",
            [
                ViewLikeOpInterface,
            ]
        > {
    let summary = "InferenceEngine run-time Gather layer";

    let arguments = (ins
        AnyMemRef:$input,
        MemRefOf<[SI64, SI32]>:$indices,
        Optional<AnyMemRef>:$axis,
        AnyMemRef:$output_buff,

        OptionalAttr<IntAttr>:$axis_value,
        IntAttr:$batch_dims
    );

    let results = (outs
        AnyMemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $indices `:` type($indices) (`,` $axis^ `:` type($axis))? `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// GatherElementsOp
//

def IERT_GatherElementsOp :
        IERT_LayerOp<1, "GatherElements",
            [
                ViewLikeOpInterface,
            ]
        > {
        let summary = "InferenceEngine run-time GatherElements layer";

        let arguments = (ins
            AnyMemRef:$input,
            MemRefOf<[SI64, SI32]>:$indices,
            AnyMemRef:$output_buff,

            IntAttr:$axis
        );

        let results = (outs
            AnyMemRef:$output
        );

        let assemblyFormat = [{
            attr-dict
            `inputs` `(` $input `:` type($input) `,` $indices `:` type($indices) `)`
            `outputs` `(` $output_buff `:` type($output_buff) `)`
            `->` type(results)
        }];
    }

//
// ScatterNDUpdateOp
//

def IERT_ScatterNDUpdateOp :
        IERT_LayerOp<1, "ScatterNDUpdate",
            [
                ViewLikeOpInterface,
            ]
        > {
    let summary = "InferenceEngine run-time ScatterNDUpdate layer";

    let arguments = (ins
        AnyMemRef:$input,
        MemRefOf<[SI64, SI32]>:$indices,
        AnyMemRef:$updates,
        AnyMemRef:$output_buff
    );

    let results = (outs
        AnyMemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $indices `:` type($indices) `,` $updates `:` type($updates) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// LSTMCellOp
//

def IERT_LSTMCellOp :
        IERT_LayerOp<2,"LSTMCell",
            [
                MultiViewOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time LSTMCell layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$inputData,
        MemRefOf<[F16, F32]>:$initialHiddenState,
        MemRefOf<[F16, F32]>:$initialCellState,
        MemRefOf<[F16, F32]>:$weights,
        MemRefOf<[F16, F32]>:$biases,

        MemRefOf<[F16, F32]>:$outputHiddenState_buff,
        MemRefOf<[F16, F32]>:$outputCellState_buff,

        IntAttr:$hiddenSize
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$outputHiddenState,
        MemRefOf<[F16, F32]>:$outputCellState
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $inputData `:` type($inputData) `,` $initialHiddenState `:` type($initialHiddenState) `,` $initialCellState `:` type($initialCellState)
                    `,` $weights `:` type($weights) `,` $biases `:` type($biases) `)`
        `outputs` `(` $outputHiddenState_buff `:` type($outputHiddenState_buff) `,` $outputCellState_buff `:` type($outputCellState_buff) `)`
        `->` type(results)
    }];
}

//
// RegionYoloOp
//

def IERT_RegionYoloOp :
        IERT_LayerOp<1, "RegionYolo",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time RegionYolo layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output_buff,

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
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// SpaceToDepthOp
//

def IERT_SpaceToDepthOp :
        IERT_LayerOp<1,
            "SpaceToDepth",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine Run-Time SpaceToDepth layer";

    let arguments = (ins
        MemRefOf<[F16]>:$input,
        MemRefOf<[F16]>:$output_buff,

        IntAttr:$block_size,
        IE_SpaceToDepthMode:$mode
    );

    let results = (outs
        MemRefOf<[F16]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// LSTMSequenceOp
//

def IERT_LSTMSequenceOp :
        IERT_LayerOp<3,"LSTMSequence",
            [
                MultiViewOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time LSTMSequence layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$inputData,
        MemRefOf<[F16, F32]>:$initialHiddenState,
        MemRefOf<[F16, F32]>:$initialCellState,
        MemRefOf<[F16, F32]>:$weights,
        MemRefOf<[F16, F32]>:$biases,

        MemRefOf<[F16, F32]>:$outputHiddenValues_buff,
        MemRefOf<[F16, F32]>:$outputHiddenState_buff,
        MemRefOf<[F16, F32]>:$outputCellState_buff,

        IntAttr:$sequenceLength,
        IE_RNNSequenceDirection:$direction
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$outputHiddenValues,
        MemRefOf<[F16, F32]>:$outputHiddenState,
        MemRefOf<[F16, F32]>:$outputCellState
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $inputData `:` type($inputData) `,` $initialHiddenState `:` type($initialHiddenState) `,` $initialCellState `:` type($initialCellState)
                    `,` $weights `:` type($weights) `,` $biases `:` type($biases) `)`
        `outputs` `(` $outputHiddenValues_buff `:` type($outputHiddenValues_buff) `,`
                    $outputHiddenState_buff `:` type($outputHiddenState_buff) `,`
                    $outputCellState_buff `:` type($outputCellState_buff) `)`
        `->` type(results)
    }];
}

//
// SubViewOp
//

def IERT_SubViewOp :
        IERT_Op<
            "SubView",
            [
                NoSideEffect,
                DeclareOpInterfaceMethods<ViewLikeOpInterface>,
                DeclareOpInterfaceMethods<InferTypeOpInterface>,
                DotInterface
            ]
        > {
    let summary = "Extract single subview from buffer";

    let arguments = (ins
        AnyMemRef:$source,

        I64ArrayAttr:$static_offsets,
        I64ArrayAttr:$static_sizes,
        OptionalAttr<I64ArrayAttr>:$static_strides
    );

    let results = (outs
        AnyMemRef:$result
    );

    let assemblyFormat = [{
        $source $static_offsets $static_sizes ($static_strides^)?
        attr-dict `:` type($source) `to` type(results)
    }];

    let extraClassDeclaration = [{
        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::ORANGE;
        }
    }];

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$source, "vpux::ShapeRef":$static_offsets, "vpux::ShapeRef":$static_sizes)
        >,
        OpBuilder<
            (ins "mlir::Value":$source, "vpux::ArrayRef<int64_t>":$static_offsets, "vpux::ArrayRef<int64_t>":$static_sizes)
        >,
        OpBuilder<
            (ins "mlir::Value":$source, "mlir::ArrayAttr":$static_offsets, "mlir::ArrayAttr":$static_sizes)
        >,

        OpBuilder<
            (ins "mlir::Value":$source, "vpux::ShapeRef":$static_offsets, "vpux::ShapeRef":$static_sizes, "vpux::ShapeRef":$static_strides)
        >,
        OpBuilder<
            (ins "mlir::Value":$source, "vpux::ArrayRef<int64_t>":$static_offsets, "vpux::ArrayRef<int64_t>":$static_sizes, "vpux::ArrayRef<int64_t>":$static_strides)
        >
    ];

    let hasFolder = 1;
    let hasCanonicalizer = 1;
}

//
// MVNOp
//

def IERT_MVNOp :
        IERT_LayerOp<1, "MVN",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine Run-Time MVN layer";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output_buff,
        BoolAttr:$across_channels,
        BoolAttr:$normalize_variance,
        F64Attr:$eps

    );

    let results = (outs
        AnyMemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// SubtractOp
//

def IERT_SubtractOp :
        IERT_LayerOp<1, "Subtract",
            [
                ViewLikeOpInterface,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time Subtract layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input1,
        MemRefOf<[F16, F32]>:$input2,
        MemRefOf<[F16, F32]>:$output_buff,

        OptionalAttr<IE_PostOp>:$post_op
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input1 `:` type($input1) `,` $input2 `:` type($input2) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// MemPermuteOp
//

def IERT_MemPermuteOp :
        IERT_LayerOp<1, "MemPermute",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time MemPermute layer";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output_buff,

        AffineMapAttr:$mem_perm
    );

    let results = (outs
        AnyMemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// PermuteCastOp
//

def IERT_PermuteCastOp :
        IERT_Op<
            "PermuteCast",
            [
                DeclareOpInterfaceMethods<ViewLikeOpInterface>,
                NoSideEffect
            ]
        > {
    let summary = "Compile-time PermuteCast layer";

    let arguments = (ins
        AnyMemRef:$source,

        AffineMapAttr:$dst_order,
        AffineMapAttr:$mem_perm
    );

    let results = (outs
        AnyMemRef:$result
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $source `:` type($source) `)`
        `->` type(results)
    }];
}

//
// CeilingOp
//

def IERT_CeilingOp :
        IERT_LayerOp<1, "Ceiling",
            [
                ViewLikeOpInterface,
                SameOperandsShape,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time Ceiling layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// NormalizeIEOp
//

def IERT_NormalizeIEOp :
        IERT_LayerOp<1, "NormalizeIE",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time NormalizeIE layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$data,
        MemRefOf<[F16, F32]>:$weights,
        MemRefOf<[F16, F32]>:$output_buff,

        F64Attr:$eps,
        BoolAttr:$across_spatial,
        BoolAttr:$channel_shared
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $data `:` type($data) `,` $weights `:` type($weights) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// EqualOp
//

def IERT_EqualOp :
        IERT_LayerOp<1, "Equal",
            [
                ViewLikeOpInterface,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time Equal layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input1,
        MemRefOf<[F16, F32]>:$input2,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input1 `:` type($input1) `,` $input2 `:` type($input2) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

// 
// DepthToSpaceOp
//

def IERT_DepthToSpaceOp :
        IERT_LayerOp<1, "DepthToSpace",
            [
                ViewLikeOpInterface,
                IERT_SameDimsOrder
            ]
        > {
    let summary = "InferenceEngine run-time DepthToSpace layer";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,
        IntAttr:$block_size,
        IE_DepthToSpaceMode:$mode
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// NotEqualOp
//

def IERT_NotEqualOp :
        IERT_LayerOp<1, "NotEqual",
            [
                ViewLikeOpInterface,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time NotEqual layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input1,
        MemRefOf<[F16, F32]>:$input2,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input1 `:` type($input1) `,` $input2 `:` type($input2) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// ReverseSequenceOp
//

def IERT_ReverseSequenceOp :
        IERT_LayerOp<1, 
            "ReverseSequence",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine Run-Time ReverseSequence layer";

    let arguments = (ins
        F16MemRef:$data,
        F16MemRef:$seq_length,
        F16MemRef:$output_buff,

        IntAttr:$seq_axis,
        IntAttr:$batch_axis
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $data `:` type($data) `,` $seq_length `:` type($seq_length) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// TopKOp
//

def IERT_TopKOp:
        IERT_LayerOp<2, "TopK",
            [
                MultiViewOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time TopK layer";

    let arguments = (ins
        MemRefOf<[F16]>:$input,
        MemRefOf<[SI32]>:$k,
        MemRefOf<[F16]>:$output_values_buff,
        MemRefOf<[SI32]>:$target_shape_buff,

        IntAttr:$axis,
        IE_TopKMode:$mode,
        IE_TopKSortType:$sort,
        TypeAttr:$element_type
    );

    let results = (outs
        MemRefOf<[F16]>:$output_values,
        MemRefOf<[SI32]>:$target_shape
    );
}

//
// SoftPlusOp
//

def IERT_SoftPlusOp :
        IERT_LayerOp<1, "SoftPlus",
            [
                SameOperandsShape,
                SameOperandsElementType,
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time SoftPlus layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// YuvToRgbOp
//

def IERT_YuvToRgbOp :
        IERT_LayerOp<1, "YuvToRgb",
            [
                ViewLikeOpInterface,
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine run-time YUV-to-RGB layer";

    let arguments = (ins
                 MemRefOf<[I8, F16, F32]> :$input1,
        Optional<MemRefOf<[I8, F16, F32]>>:$input2,
        Optional<MemRefOf<[I8, F16, F32]>>:$input3,
                 MemRefOf<[I8, F16, F32]> :$output_buff,

        IE_ColorFmt:$inFmt,
        IE_ColorFmt:$outFmt
    );

    let results = (outs
        MemRefOf<[I8, F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input1 `:` type($input1) (`,` $input2^ `:` type($input2))? (`,` $input3^ `:` type($input3))? `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

#endif
