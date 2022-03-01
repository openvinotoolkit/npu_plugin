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
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Dialect/Quant/QuantOpsBase.td"
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
            [
                NoSideEffect,
                InferTypeOpInterface,
                DeclareOpInterfaceMethods<InferShapedTypeOpInterface, ["inferReturnTypeComponents"]>,
                DeclareOpInterfaceMethods<IE_LayerOpInterface>
            ] # traits
        > {
    list<string> elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    bit checkInferredDimsOrder = 0;
    bit checkInferredMemSpace = 0;
    bit checkInferredSparsity = 0;

    let extraClassDeclaration = [{
        static bool isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
            return vpux::IE::isCompatibleTensorTypes(lhs, rhs,
                }] # !interleave(elemComparisonModes, "|") # [{,
                static_cast<bool>(}] # checkInferredDimsOrder # [{),
                static_cast<bool>(}] # checkInferredMemSpace # [{),
                static_cast<bool>(}] # checkInferredSparsity # [{)
            );
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
                DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                OpAsmOpInterface
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "InferenceEngine CNN Network description";

    let description = [{
        This operation is bound to MLIR Module and holds extra information about InferenceEngine CNN Network:

          * Precision and layout for user-provided inputs.
          * Precision and layout for user-provided outputs.
          * Layout for output profiling data(optional).
          * Entry point (Function name) for the network inference.
    }];

    let arguments = (ins
        FlatSymbolRefAttr:$entryPoint
    );

    let regions = (region
        SizedRegion<1>:$inputsInfo,
        SizedRegion<1>:$outputsInfo,
        VariadicRegion<SizedRegion<1>>:$profilingOutputsInfo
    );

    let extraClassDeclaration = [{
        size_t getNetInputsCount();
        vpux::SmallVector<vpux::IE::DataInfoOp, 1> getInputsInfo();

        size_t getNetOutputsCount();
        vpux::SmallVector<vpux::IE::DataInfoOp, 1> getOutputsInfo();

        size_t getProfilingOutputsCount();
        vpux::SmallVector<vpux::IE::DataInfoOp, 1> getProfilingOutputsInfo();

        static void getFromModule(
                mlir::ModuleOp module,
                vpux::IE::CNNNetworkOp& netInfo,
                mlir::FuncOp& netFunc);

        static mlir::StringRef getDefaultDialect() {
            return "IE";
        }
    }];

    let builders = [
        OpBuilder<
            (ins "mlir::FlatSymbolRefAttr":$entryPoint, "bool":$withProfiling)
        >
    ];

    let assemblyFormat = [{
        attr-dict
        `entryPoint` `:` $entryPoint
        `inputsInfo` `:` $inputsInfo
        `outputsInfo` `:` $outputsInfo
        (`profilingOutputsInfo` `:` $profilingOutputsInfo^)?
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
// MemoryResourceOp
//

def IE_MemoryResourceOp :
        IE_Op<
            "MemoryResource",
            [
                IERT_ResourceOpInterface,
                Symbol
            ]
        > {
    let summary = "Information about memory resource";

    let description = [{
        The memory resource is defined by the following attributes:

          * sym_name - kind of memory space.
          * Size - size in bytes of memory space.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        IntAttr:$byteSize
    );

    let extraClassDeclaration = [{
        vpux::Byte size() { return vpux::Byte(byteSize()); }

        static mlir::StringRef getDefaultDialect() {
            return "IE";
        }
    }];

    let assemblyFormat = [{
        $byteSize `bytes` `of` $sym_name
        attr-dict
    }];
}

//
// ExecutorResourceOp
//

def IE_ExecutorResourceOp :
        IE_Op<
            "ExecutorResource",
            [
                IERT_ResourceOpInterface,
                SymbolTable,
                Symbol
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Information about executor resource";

    let description = [{
        The executor resource is defined by the following attributes:

          * sym_name - kind of the executor.
          * Count - number of executor units.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        IntAttr:$count
    );

    let regions = (region
        SizedRegion<1>:$subExecutors
    );

    let extraClassDeclaration = [{
        vpux::IE::ExecutorResourceOp addSubExecutor(mlir::StringAttr executor, uint32_t count);
        vpux::IE::ExecutorResourceOp getSubExecutor(mlir::StringAttr executor);

        template <typename Enum, typename OutT = vpux::IE::ExecutorResourceOp>
        using exec_resource_if = enable_t<OutT, std::is_enum<Enum>, details::HasStringifyEnum<Enum>>;

        template <typename Enum>
        exec_resource_if<Enum> addSubExecutor(Enum kind, uint32_t count) {
            return addSubExecutor(mlir::StringAttr::get(getContext(), stringifyEnum(kind)), count);
        }

        template <typename Enum>
        exec_resource_if<Enum> getSubExecutor(Enum kind) {
            return getSubExecutor(mlir::StringAttr::get(getContext(), stringifyEnum(kind)));
        }

        static mlir::StringRef getDefaultDialect() {
            return "IE";
        }
    }];

    let assemblyFormat = [{
        attr-dict
        $count `of` $sym_name
        custom<OptionalBlockRegion>($subExecutors)
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
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Convert layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        TypeAttr:$dstElemType
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
                DeclareOpInterfaceMethods<IE_ElemTypeInfoOpInterface>
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

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 0;
}

//
// SoftMaxOp
//

def IE_SoftMaxOp :
        IE_LayerOp<
            "SoftMax"
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
        RankedTensorOf<[AnyInteger]>:$repeats
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
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
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

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
}

//
// PowerOp
//

def IE_PowerOp :
        IE_LayerOp<
            "Power",
            [
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
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
                IE_TilingBuilderOpInterface,
                Commutative,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Add layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$input1,
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$input2,

        IE_AutoBroadcastType:$auto_broadcast,
        OptionalAttr<IE_PostOp>:$post_op
    );

    let results = (outs
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_QUANT_MIXED_PRECISION];
}

//
// DivideOp
//

def IE_DivideOp :
        IE_LayerOp<
            "Divide",
            [
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
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
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
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
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
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
// LessOp
//

def IE_LessOp :
        IE_LayerOp<
            "Less",
            [
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Less layer";

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
// LessEqualOp
//

def IE_LessEqualOp :
        IE_LayerOp<
            "LessEqual",
            [
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine LessEqual layer";

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
// GreaterOp
//

def IE_GreaterOp :
        IE_LayerOp<
            "Greater",
            [
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Greater layer";

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
// GreaterEqualOp
//

def IE_GreaterEqualOp :
        IE_LayerOp<
            "GreaterEqual",
            [
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine GreaterEqual layer";

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
// LogicalOrOp
//

def IE_LogicalOrOp :
        IE_LayerOp<
            "LogicalOr",
            [
                IE_TilingBuilderOpInterface,
                Commutative,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine LogicalOr layer";

    let arguments = (ins
        RankedTensorOf<[I8, F16, F32]>:$input1,
        RankedTensorOf<[I8, F16, F32]>:$input2,

        IE_AutoBroadcastType:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[I8, F16, F32]>:$output
    );
}

//
// LogicalNotOp
//

def IE_LogicalNotOp :
        IE_LayerOp<
            "LogicalNot",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Logical Not layer";

    let arguments = (ins
        RankedTensorOf<[I8, F16, F32]>:$input1
    );

    let results = (outs
        RankedTensorOf<[I8, F16, F32]>:$output
    );
}

//
// LogicalXorOp
//

def IE_LogicalXorOp :
        IE_LayerOp<
            "LogicalXor",
            [
                IE_TilingBuilderOpInterface,
                Commutative,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine LogicalXor layer";

    let arguments = (ins
        RankedTensorOf<[I8, F16, F32]>:$input1,
        RankedTensorOf<[I8, F16, F32]>:$input2,

        IE_AutoBroadcastType:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[I8, F16, F32]>:$output
    );
}

//
// MultiplyOp
//

def IE_MultiplyOp :
        IE_LayerOp<
            "Multiply",
            [
                IE_TilingBuilderOpInterface,
                Commutative,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Multiply layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$input1,
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$input2,

        IE_AutoBroadcastType:$auto_broadcast,
        OptionalAttr<IE_PostOp>:$post_op
    );

    let results = (outs
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_QUANT_MIXED_PRECISION];
}

//
// AndOp
//

def IE_AndOp :
        IE_LayerOp<
            "And",
            [
                IE_TilingBuilderOpInterface,
                Commutative,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine And layer";

    let arguments = (ins
        RankedTensorOf<[I8, F16, F32, quant_QuantizedType]>:$input1,
        RankedTensorOf<[I8, F16, F32, quant_QuantizedType]>:$input2,

        IE_AutoBroadcastType:$auto_broadcast,
        OptionalAttr<IE_PostOp>:$post_op
    );

    let results = (outs
        RankedTensorOf<[I8, F16, F32, quant_QuantizedType]>:$output
    );

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_QUANT_MIXED_PRECISION];
}

//
// ConvolutionOp
//

def IE_ConvolutionOp :
        IE_LayerOp<
            "Convolution",
            [
                DeclareOpInterfaceMethods<IE_TilingBuilderOpInterface>
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

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_QUANT_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT];
}

//
// GroupConvolutionOp
//

def IE_GroupConvolutionOp :
        IE_LayerOp<
            "GroupConvolution",
            [
                DeclareOpInterfaceMethods<IE_TilingBuilderOpInterface>
            ]
        > {
    let summary = "InferenceEngine GroupConvolution layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$input,
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$filter,
        Optional<RankedTensorOf<[F16, F32]>>:$bias,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        I64ArrayAttr:$dilations,
        OptionalAttr<IntAttr>:$groups,

        OptionalAttr<IE_PostOp>:$post_op
    );

    let results = (outs
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$output
    );

    let hasCanonicalizer = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_QUANT_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT];
}

//
// AvgPoolOp
//

def IE_AvgPoolOp :
        IE_LayerOp<
            "AvgPool"
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
                DeclareOpInterfaceMethods<IE_TilingBuilderOpInterface>
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

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_QUANT_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT];
}


//
// AdaptiveAvgPoolOp
//

def IE_AdaptiveAvgPoolOp :
        IE_LayerOp<
            "AdaptiveAvgPoolOp"
        > {
    let summary = "InferenceEngine AdaptiveAvgPool layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input1,
        RankedTensorOf<[F16, F32]>:$input2,

        OptionalAttr<IE_AdaptivePoolMode>:$mode
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// ShuffleChannelsOp
//

def IE_ShuffleChannelsOp :
        IE_LayerOp<
            "ShuffleChannels",
            [
                SameOperandsAndResultType,
                SameOperandsAndResultShape
            ]
        > {
    let summary = "InferenceEngine ShuffleChannels layer";

    let arguments = (ins
        4DTensorOf<[F16, F32]>:$input,

        IntAttr:$axis,
        IntAttr:$group
    );

    let results = (outs
        4DTensorOf<[F16, F32]>:$output
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
        RankedTensorOf<[AnyInteger]>:$indices,
        Optional<AnyRankedTensor>:$axis,
        OptionalAttr<IntAttr>:$axis_value,
        IntAttr:$batch_dims
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
}

//
// GatherElementsOp
//

def IE_GatherElementsOp :
        IE_LayerOp<
              "GatherElements"
        > {
    let summary = "InferenceEngine GatherElements layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[AnyInteger]>:$indices,

        IntAttr:$axis
    );

    let results = (outs
        AnyRankedTensor:$output
    );

}


//
// ScatterNDUpdateOp
//

def IE_ScatterNDUpdateOp :
        IE_LayerOp<
            "ScatterNDUpdate"
        > {
    let summary = "InferenceEngine ScatterNDUpdate layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[AnyInteger]>:$indices,
        AnyRankedTensor:$updates
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
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
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
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
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
                DeclareOpInterfaceMethods<IE_ElemTypeInfoOpInterface>
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

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
    let checkInferredSparsity = 1;
}

//
// SqueezeOp
//

def IE_SqueezeOp :
        IE_LayerOp<
            "Squeeze",
            [
                DeclareOpInterfaceMethods<IE_LayoutInfoOpInterface>
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

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
    let checkInferredSparsity = 1;
}

//
// UnsqueezeOp
//

def IE_UnsqueezeOp :
        IE_LayerOp<
            "Unsqueeze",
            [
                DeclareOpInterfaceMethods<IE_LayoutInfoOpInterface>
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

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
    let checkInferredSparsity = 1;
}

//
// SigmoidOp
//

def IE_SigmoidOp :
        IE_LayerOp<
            "Sigmoid",
            [
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
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
// LRN_IEOp
//

def IE_LRN_IEOp :
        IE_LayerOp<
            "LRN_IE"
        > {
    let summary = "InferenceEngine LRN_IE layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        F64Attr:$alpha,
        F64Attr:$beta,
        F64Attr:$bias,
        IntAttr:$size,
        IE_LRN_IERegion:$region
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// BroadcastOp
//

def IE_BroadcastOp :
        IE_LayerOp<
            "Broadcast"
        > {
    let summary = "InferenceEngine Broadcast layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        1DTensorOf<[AnyInteger]>:$target_shape,
        Optional<1DTensorOf<[AnyInteger]>>:$axes_mapping,

        OptionalAttr<IE_BroadcastType>:$mode
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// ReduceMaxOp
//

def IE_ReduceMaxOp :
        IE_LayerOp<
            "ReduceMax"
        > {
    let summary = "InferenceEngine ReduceMax layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        1DTensorOf<[AnyInteger]>:$axes,

        UnitAttr:$keep_dims
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// ReduceMeanOp
//

def IE_ReduceMeanOp :
        IE_LayerOp<
            "ReduceMean"
        > {
    let summary = "InferenceEngine ReduceMean Layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        1DTensorOf<[AnyInteger]>:$axes,

        UnitAttr:$keep_dims
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// ReduceSumOp
//

def IE_ReduceSumOp :
        IE_LayerOp<
            "ReduceSum"
        > {
    let summary = "InferenceEngine ReduceSum layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        1DTensorOf<[AnyInteger]>:$axes,

        UnitAttr:$keep_dims
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
            "Minimum",
            [
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
            ]
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
            "Maximum",
            [
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
            ]
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
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
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
// QuantizeOp
//

def IE_QuantizeOp :
        IE_LayerOp<"Quantize",
            [
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Quantize layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        TypeAttr:$dstElemType
    );

    let results = (outs
        RankedTensorOf<[quant_QuantizedType]>:$output
    );

    let hasFolder = 1;
}

//
// DequantizeOp
//

def IE_DequantizeOp :
        IE_LayerOp<"Dequantize",
            [
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Dequantize layer";

    let arguments = (ins
        RankedTensorOf<[quant_QuantizedType]>:$input,

        TypeAttr:$dstElemType
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// QuantizeCastOp
//

def IE_QuantizeCastOp :
        IE_LayerOp<
            "QuantizeCast"
        > {
    let summary = "InferenceEngine Quantize Cast layer";

    let arguments = (ins
        RankedTensorOf<[SI8, UI8, quant_QuantizedType]>:$input,

        TypeAttr:$dstElemType
    );

    let results = (outs
        RankedTensorOf<[quant_QuantizedType]>:$output
    );

    let hasFolder = 1;
}

//
// MatMul
//

def IE_MatMulOp:
        IE_LayerOp<
            "MatMul"
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
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
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
// SqrtOp
//

def IE_SqrtOp :
        IE_LayerOp<
            "Sqrt",
            [
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Sqrt layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// SinhOp
//

def IE_SinhOp :
        IE_LayerOp<
            "Sinh",
            [
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Sinh layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// CoshOp
//

def IE_CoshOp :
        IE_LayerOp<
            "Cosh",
            [
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Cosh layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// AsinhOp
//

def IE_AsinhOp :
        IE_LayerOp<
            "Asinh",
            [
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Asinh layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// AcoshOp
//

def IE_AcoshOp :
        IE_LayerOp<
            "Acosh",
            [
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Acosh layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// AbsOp
//

def IE_AbsOp :
        IE_LayerOp<
            "Abs",
            [
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Abs layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// AtanOp
//

def IE_AtanOp :
        IE_LayerOp<
            "Atan",
            [
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Atan layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// AsinOp
//

def IE_AsinOp :
        IE_LayerOp<
            "Asin",
            [
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Asin layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// AcosOp
//

def IE_AcosOp :
        IE_LayerOp<
            "Acos",
            [
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Acos layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// AtanhOp
//

def IE_AtanhOp :
        IE_LayerOp<
            "Atanh",
            [
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Atanh layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// LogOp
//

def IE_LogOp :
        IE_LayerOp<
            "Log",
            [
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Log layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// GeluOp
//

def IE_GeluOp :
        IE_LayerOp<
            "Gelu",
            [
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Gelu layer";

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
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
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
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
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
// FloorOp
//

def IE_FloorOp :
        IE_LayerOp<
            "Floor",
            [
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Floor layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// RoundOp
//

def IE_RoundOp :
        IE_LayerOp<
            "Round",
            [
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Round layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        IE_RoundMode:$mode
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
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
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
// ErfOp
//

def IE_ErfOp :
        IE_LayerOp<
            "Erf",
            [
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Erf layer";

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
            "Transpose",
            [
                DeclareOpInterfaceMethods<IE_ElemTypeInfoOpInterface>
            ]
        > {
    let summary = "InferenceEngine Transpose layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$order,

        OptionalAttr<AffineMapAttr>:$order_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 0;
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
// NormalizeIEOp
//

def IE_NormalizeIEOp :
        IE_LayerOp<
            "NormalizeIE"
        > {
    let summary = "InferenceEngine NormalizeIE layer";

    let arguments = (ins
        AnyRankedTensor:$data,
        AnyRankedTensor:$weights,

        F64Attr:$eps,
        BoolAttr:$across_spatial,
        BoolAttr:$channel_shared
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// MVNOp
//

def IE_MVNOp :
        IE_LayerOp<
            "MVN"
        > {
    let summary = "InferenceEngine MVN layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        BoolAttr:$across_channels,
        BoolAttr:$normalize_variance,
        F64Attr:$eps
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

        OptionalAttr<IE_ConcatAttrs>:$per_axis,
        OptionalAttr<I64ArrayOfArraysAttr>:$static_offsets
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let builders = [
        OpBuilder<
            (ins "mlir::ValueRange":$inputs, "vpux::IE::ConcatAttrs":$per_axis)
        >,
        OpBuilder<
            (ins "mlir::ValueRange":$inputs, "mlir::IntegerAttr":$axis,
                 CArg<"mlir::IntegerAttr", "{}">:$offset, CArg<"mlir::IntegerAttr", "{}">:$stride)
        >,
        OpBuilder<
            (ins "mlir::ValueRange":$inputs, "int64_t":$axis, CArg<"int64_t", "0">:$offset, CArg<"int64_t", "1">:$stride)
        >,
        OpBuilder<
            (ins "mlir::ValueRange":$inputs, "vpux::Dim":$axis, CArg<"int64_t", "0">:$offset, CArg<"int64_t", "1">:$stride)
        >,

        OpBuilder<
            (ins "mlir::Type":$outType, "mlir::ValueRange":$inputs, "mlir::ArrayAttr":$static_offsets)
        >,
        OpBuilder<
            (ins "mlir::Type":$outType, "mlir::ValueRange":$inputs, "vpux::ArrayRef<vpux::Shape>":$static_offsets)
        >,
        OpBuilder<
            (ins "mlir::Type":$outType, "mlir::ValueRange":$inputs, "vpux::ArrayRef<vpux::ShapeRef>":$static_offsets)
        >,
    ];

    let hasCanonicalizer = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
}

//
// ROIPoolingOp
//

def IE_ROIPoolingOp :
        IE_LayerOp<
            "ROIPooling"
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
// ROIAlignOp
//

def IE_ROIAlignOp :
        IE_LayerOp<
            "ROIAlign",
            [
               ResultsAreFloatLike
            ]
        > {
    let summary = "InferenceEngine ROIAlign layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$coords,
        1DTensorOf<[AnyInteger]>:$roisIdx,

        IntAttr:$pooled_h,
        IntAttr:$pooled_w,
        IntAttr:$sampling_ratio,
        F64Attr:$spatial_scale,
        IE_ROIAlignMethod:$poolingMode
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

    let hasFolder = 1;
    let hasCanonicalizer = 1;

    let extraClassDeclaration = [{
        bool isSimplified();
    }];
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
            "PRelu",
            [
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
            ]
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
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
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
            "Swish",
            [
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
            ]
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
                IE_TilingBuilderOpInterface,
                AttrSizedOperandSegments,
                IE_EltwiseOp
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
// UpsamplingOp
//

def IE_UpsamplingOp :
        IE_LayerOp<
            "Upsampling"
        > {
    let summary = "InferenceEngine Upsampling layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        I64ArrayAttr:$upsampling_factor,
        I64ArrayAttr:$pad_l,
        I64ArrayAttr:$pad_r
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}


//
// GRNOp
//

def IE_GRNOp :
        IE_LayerOp<
            "GRN"
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
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
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
// SignOp
//

def IE_SignOp :
        IE_LayerOp<
            "Sign",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Sign layer";

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
            "FullyConnected"
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
            "CTCGreedyDecoder"
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

        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$input, "vpux::Optional<vpux::ShapeRef>":$pads_begin, "vpux::Optional<vpux::ShapeRef>":$pads_end)
        >
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
}

//
// LSTMCellOp
//

def IE_LSTMCellOp :
        IE_LayerOp<
            "LSTMCell"
        > {
    let summary = "InferenceEngine LSTMCell layer";

    let arguments = (ins
        2DTensorOf<[F16, F32]>:$inputData,
        2DTensorOf<[F16, F32]>:$initialHiddenState,
        2DTensorOf<[F16, F32]>:$initialCellState,
        2DTensorOf<[F16, F32]>:$weights,
        2DTensorOf<[F16, F32]>:$recurrenceWeights,
        1DTensorOf<[F16, F32]>:$biases,

        IntAttr:$hiddenSize
    );

    let results = (outs
        2DTensorOf<[F16, F32]>:$outputHiddenState,
        2DTensorOf<[F16, F32]>:$outputCellState
    );
}

//
// SliceOp
//

def IE_SliceOp :
        IE_LayerOp<
            "Slice"
        > {
    let summary = "Extract single slice from tensor";

    let arguments = (ins
        AnyRankedTensor:$source,

        I64ArrayAttr:$static_offsets,
        I64ArrayAttr:$static_sizes
    );

    let results = (outs
        AnyRankedTensor:$result
    );

    let assemblyFormat = [{
        $source $static_offsets $static_sizes
        attr-dict `:` type($source) `to` type(results)
    }];

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$source, "vpux::ShapeRef":$static_offsets, "vpux::ShapeRef":$static_sizes)
        >,
        OpBuilder<
            (ins "mlir::Value":$source, "vpux::ArrayRef<int64_t>":$static_offsets, "vpux::ArrayRef<int64_t>":$static_sizes)
        >
    ];

    let hasFolder = 1;
    let hasCanonicalizer = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
}

//
// SubtractOp
//

def IE_SubtractOp :
        IE_LayerOp<
            "Subtract",
            [
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Subtract layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input1,
        RankedTensorOf<[F16, F32]>:$input2,

        IE_AutoBroadcastType:$auto_broadcast,
        OptionalAttr<IE_PostOp>:$post_op
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

//
// LSTMSequenceOp
//

def IE_LSTMSequenceOp :
        IE_LayerOp<
            "LSTMSequence"
        > {
    let summary = "InferenceEngine LSTMSequence layer";

    let arguments = (ins
        3DTensorOf<[F16, F32]>:$inputData,
        3DTensorOf<[F16, F32]>:$initialHiddenState,
        3DTensorOf<[F16, F32]>:$initialCellState,
        3DTensorOf<[F16, F32]>:$weights,
        3DTensorOf<[F16, F32]>:$reccurenceWeights,
        2DTensorOf<[F16, F32]>:$biases,

        IntAttr:$sequenceLength,
        IE_RNNSequenceDirection:$direction
    );

    let results = (outs
        4DTensorOf<[F16, F32]>:$outputHiddenValues,
        3DTensorOf<[F16, F32]>:$outputHiddenState,
        3DTensorOf<[F16, F32]>:$outputCellState
    );
}

//
// MemPermuteOp
//

def IE_MemPermuteOp :
        IE_LayerOp<
            "MemPermute"
        > {
    let summary = "InferenceEngine MemPermute layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        AffineMapAttr:$dst_order,
        AffineMapAttr:$mem_perm
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
    let hasCanonicalizer = 1;

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 0;
}

//
// PermuteCastOp
//

def IE_PermuteCastOp :
        IE_LayerOp<
            "PermuteCast"
        > {
    let summary = "InferenceEngine PermuteCast layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        AffineMapAttr:$dst_order,
        AffineMapAttr:$mem_perm
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
    let hasCanonicalizer = 1;

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
    let checkInferredSparsity = 1;
}

//
// CeilingOp
//

def IE_CeilingOp :
        IE_LayerOp<
            "Ceiling",
            [
                IE_TilingBuilderOpInterface,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Ceiling layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// EqualOp
//

def IE_EqualOp :
        IE_LayerOp<
            "Equal",
            [
                IE_TilingBuilderOpInterface,
                Commutative,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Equal layer";

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
// SpaceToDepthOp
//

def IE_SpaceToDepthOp :
        IE_LayerOp<
            "SpaceToDepthOp"
        > {
    let summary = "InferenceEngine SpaceToDepthOp layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        DefaultValuedAttr<IntAttr, "1">:$block_size,
        IE_SpaceToDepthMode:$mode
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
    let hasCanonicalizer = 1;
}

//
// AffineReshapeOp
//

def IE_AffineReshapeOp :
        IE_LayerOp<
            "AffineReshape",
            [
                DeclareOpInterfaceMethods<IE_LayoutInfoOpInterface>,
                DeclareOpInterfaceMethods<IE_ElemTypeInfoOpInterface>
            ]
        > {
    let summary = "AffineReshape layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        I64ArrayOfArraysAttr:$dim_mapping,
        I64ArrayAttr:$shape_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
    let checkInferredSparsity = 1;
}

//
// NotEqualOp
//

def IE_NotEqualOp :
        IE_LayerOp<
            "NotEqual",
            [
                IE_TilingBuilderOpInterface,
                Commutative,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine NotEqual layer";

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
// ReverseSequenceOp
//

def IE_ReverseSequenceOp :
        IE_LayerOp<
            "ReverseSequence"
        > {
    let summary = "Reverse variable length sequence operation";

    let arguments = (ins
        AnyRankedTensor:$data,
        1DTensorOf<[AnyInteger]>:$seq_length,

        IntAttr:$seq_axis,
        IntAttr:$batch_axis
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
    let hasCanonicalizer = 1;
}

//
// DepthToSpaceOp
//

def IE_DepthToSpaceOp :
        IE_LayerOp<
            "DepthToSpace"
        > {
    let summary = "InferenceEngine DepthToSpace layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        IntAttr:$block_size,
        IE_DepthToSpaceMode:$mode
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
    let hasCanonicalizer = 1;
}

//
// SoftPlusOp
//

def IE_SoftPlusOp :
        IE_LayerOp<
            "SoftPlus",
            [
                IE_EltwiseOp,
                IE_TilingBuilderOpInterface
            ]
        > {
    let summary = "InferenceEngine SoftPlus layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// CopyOp
//

def IE_CopyOp :
        IE_LayerOp<
            "Copy"
        > {
    let summary = "InferenceEngine Copy layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        OptionalAttr<IndexedSymbolAttr>:$out_mem_space
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
    let hasCanonicalizer = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
    let checkInferredSparsity = 1;
}

// YuvToRgbOp
//  Conversions:
//   NV12toRGB, NV12toBGR,
//   I420toRGB, I420toBGR
//

def IE_YuvToRgbOp :
        IE_LayerOp<
            "YuvToRgb",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine NV12/I420 to RGB/BGR layer";

    let arguments = (ins
                 4DTensorOf<[SI8, F16, F32]> :$input1,
        Optional<4DTensorOf<[SI8, F16, F32]>>:$input2,
        Optional<4DTensorOf<[SI8, F16, F32]>>:$input3,

        IE_ColorFmt:$inFmt,
        IE_ColorFmt:$outFmt
    );

    let results = (outs
        4DTensorOf<[SI8, F16, F32]>:$output
    );
}

//
// ExpandDilatedOp
//

def IE_ExpandDilatedOp :
        IE_LayerOp<
            "ExpandDilated",
            [
                DeclareOpInterfaceMethods<IE_ElemTypeInfoOpInterface>
            ]
        > {
    let summary = "Expand tensor with uninitialized values according to dilations";

    let arguments = (ins
        AnyRankedTensor:$input,

        I64ArrayAttr:$dilations
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
}

#endif
