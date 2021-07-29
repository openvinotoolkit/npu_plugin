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
include "vpux/compiler/dialect/IERT/dialect.td"
include "vpux/compiler/dialect/IE/attributes.td"

include "mlir/Dialect/Quant/QuantOpsBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
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

class IERT_LayerOp<int numResult, string mnemonic, list<OpTrait> traits = []> :
        IERT_Op<
            mnemonic,
            !listconcat(
                traits,
                [
                    NoSideEffect,
                    InferTypeOpInterface,
                    DeclareOpInterfaceMethods<RTLayerInterface>
                ]
            )
        > {
    let extraClassDeclaration = InferReturnTypesForRTLayer<numResult>.impl;
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
        IntAttr:$byteSize
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
        IntAttr:$count
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
        IntAttr:$offset
    );

    let results = (outs
        AnyMemRef:$memory
    );

    let assemblyFormat = [{
        `<` $offset `>` attr-dict `->` type(results)
    }];
}

//
// TimestampOp
//

def IERT_TimestampOp :
        IERT_Op<
            "Timestamp",
            [
                NoSideEffect,
            ]
        > {
    let summary = "Get timer timestamp operation";

    let description = [{
        Get timer timestamp operation
    }];

    let results = (outs
        AnyStaticShapeMemRef:$output
    );

    let assemblyFormat = [{
        attr-dict `->` type(results)
    }];
}

//
// ConvertOp
//

def IERT_ConvertOp :
        IERT_LayerOp<1, "Convert",
            [
                SameOperandsShape,
                ViewLikeOpInterface,
                DeclareOpInterfaceMethods<ConvertLayerInterface>
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
                ViewLikeOpInterface
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

    let hasFolder = 1;
}

//
// ReorderOp
//

def IERT_ReorderOp :
        IERT_LayerOp<1, "Reorder",
            [
                ViewLikeOpInterface,
                SameOperandsShape
            ]
        > {
    let summary = "InferenceEngine run-time Reorder layer";

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

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

//
// SoftMaxOp
//

def IERT_SoftMaxOp :
        IERT_LayerOp<1, "SoftMax",
            [
                SameOperandsShape,
                ViewLikeOpInterface,
                SameOperandsElementType,
                DeclareOpInterfaceMethods<SoftMaxLayerInterface>
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
                DeclareOpInterfaceMethods<TensorInterface_4D>,
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
                ViewLikeOpInterface,
                DeclareOpInterfaceMethods<TensorInterface_4D>,
                DeclareOpInterfaceMethods<ConvolutionLayerInterface>
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
        MemRefOf<[F16, F32]>:$output
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
                ViewLikeOpInterface,
                SameOperandsElementType,
                DeclareOpInterfaceMethods<ConvolutionLayerInterface>
            ]
        > {
    let summary = "InferenceEngine run-time GroupConvolution layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$filter,
        Optional<MemRefOf<[F16, F32]>>:$bias,
        MemRefOf<[F16, F32]>:$output_buff,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        I64ArrayAttr:$dilations,
        OptionalAttr<IntAttr>:$groups,

        OptionalAttr<IE_PostOp>:$post_op
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
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
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time Add layer";

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
// MultiplyOp
//

def IERT_MultiplyOp :
        IERT_LayerOp<1, "Multiply",
            [
                ViewLikeOpInterface,
                SameOperandsElementType
            ]
        > {
    let summary = "InferenceEngine run-time Multiply layer";

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
        AnyStridedMemRef:$input
    );

    let results = (outs
        AnyStridedMemRef:$output
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
// TransposeOp
//

def IERT_TransposeOp :
        IERT_LayerOp<1, "Transpose",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time Transpose layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        Optional<MemRefOf<[SI64]>>:$order,
        MemRefOf<[F16, F32]>:$output_buff,

        AffineMapAttr:$order_value
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) (`,` $order^ `:` type($order))? `)`
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
                SameOperandsAndResultElementType,
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
    }] # InferReturnTypesForRTLayer<1>.impl;
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

#endif
