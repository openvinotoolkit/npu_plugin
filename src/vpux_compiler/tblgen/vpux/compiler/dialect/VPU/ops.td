//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_VPU_OPS
#define VPUX_COMPILER_DIALECT_VPU_OPS

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/core/ops_interfaces.td"
include "vpux/compiler/core/types.td"
include "vpux/compiler/dialect/const/attributes.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/EMU/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPU/dialect.td"
include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/types.td"

include "mlir/Dialect/Quant/QuantOpsBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"

//
// Base classes
//

class VPU_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<
            VPU_Dialect,
            mnemonic,
            traits
        >;

class VPU_LayerOp<string mnemonic, list<OpTrait> traits = []> :
        VPU_Op<
            mnemonic,
            [
                NoSideEffect,
                DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
                DeclareOpInterfaceMethods<VPU_LayerOpInterface>,
                DeclareOpInterfaceMethods<EMU_SerializeInterface>,
                DeclareOpInterfaceMethods<VPU_EMUUPAOpInterface>
            ] # traits
        > {
    list<string> elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    bit checkInferredDimsOrder = 0;
    bit checkInferredMemSpace = 0;

    code baseExtraClassDeclaration = [{
        static bool isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
            return vpux::IE::areTypesCompatible(lhs, rhs,
                }] # !interleave(elemComparisonModes, "|") # [{,
                static_cast<bool>(}] # checkInferredDimsOrder # [{),
                static_cast<bool>(}] # checkInferredMemSpace # [{)
            );
        }
    }];
    let extraClassDeclaration = baseExtraClassDeclaration;

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
}

//
// DPU.Workload
//

def VPU_DPUWorkloadOp :
        VPU_Op<
            "DPU.Workload",
            [
                ParentOneOf<[
                    "vpux::VPU::NCEConvolutionOp",
                    "vpux::VPU::NCEDepthConvolutionOp",
                    "vpux::VPU::NCEMaxPoolOp",
                    "vpux::VPU::NCEAveragePoolOp",
                    "vpux::VPU::NCEEltwiseOp"
                ]>
            ]
        > {
    let summary = "Workload for a single DPU tile";

    let arguments = (ins
        Confined<I64ArrayAttr, [ArrayCount<4>]>:$offsets,
        Confined<I64ArrayAttr, [ArrayCount<4>]>:$sizes,
        VPU_PaddingAttr:$pad,
        VPU_MPEMode:$mpe_mode,
        OptionalAttr<IntAttr>:$cluster_id
    );

    let builders = [
        OpBuilder<(ins
            "mlir::Value":$offsets,
            "mlir::Value":$sizes,
            "vpux::VPU::PaddingAttr":$kernelFunction,
            "vpux::VPU::MPEMode":$mpe_mode
        )>
    ];

    let assemblyFormat = [{
        $offsets $sizes $pad $mpe_mode attr-dict-with-keyword
    }];
}

//
// NCE.Convolution
//

def VPU_NCEConvolutionOp :
        VPU_LayerOp<
            "NCE.Convolution",
            [
                NoRegionArguments,
                NoTerminator,
                SingleBlock,
                DeclareOpInterfaceMethods<IE_LayoutInfoOpInterface>,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_NCEOpInterface>,
                AttrSizedOperandSegments,
                IE_AlignedChannelsOpInterface
            ]
        > {
    let summary = "NCE version of Convolution layer";

    let arguments = (ins
        4DTensorOf<[F16, BF16, quant_QuantizedType]>:$input,
        4DTensorOf<[F16, BF16, quant_QuantizedType]>:$filter,
        4DTensorOf<[SI32]>:$weightsTable,
        Optional<4DTensorOf<[UI8]>>:$activationWindow,
        Optional<4DTensorOf<[SI32]>>:$instructionListTable,

        Confined<I64ArrayAttr, [ArrayCount<2>]>:$strides,
        VPU_PaddingAttr:$pad,

        OptionalAttr<VPU_PPETaskAttr>:$ppe,

        Confined<I64ArrayAttr, [ArrayCount<4>]>:$rawFilterShape,
        OptionalAttr<IntAttr>:$activation_window_channel_length
    );

    let results = (outs
        4DTensorOf<[F16, BF16, quant_QuantizedType]>:$output
    );

    let regions = (region
        AnyRegion:$workloads
    );

    let assemblyFormat = [{
        `(` $input `,` $filter `,` $weightsTable (`,` $activationWindow^ custom<OptionalTypes>(type($activationWindow)) ``)?
         (`,` $instructionListTable^ custom<OptionalTypes>(type($instructionListTable)) ``)? `)`
        attr-dict
        custom<OptionalTypes>(type($input), type($filter), type($weightsTable)) ``
        `->` type(results)
        custom<OptionalRegion>($workloads)
    }];

    let verifier = [{
        return vpux::VPU::verifyOp(*this);
    }];

    let extraClassDeclaration = [{
        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter, vpux::NDTypeInterface output);

        static bool isSupported(vpux::IE::ConvolutionOp origOp, vpux::LogCb logCb);

        vpux::Shape inferAlignedFilterShape(vpux::NDTypeInterface input, vpux::NDTypeInterface output);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_QUANT_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT];
}

//
// NCE.DepthConvolution
//

def VPU_NCEDepthConvolutionOp :
        VPU_LayerOp<
            "NCE.DepthConvolution",
            [
                NoRegionArguments,
                NoTerminator,
                SingleBlock,
                DeclareOpInterfaceMethods<IE_LayoutInfoOpInterface>,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_NCEOpInterface>,
                IE_AlignedChannelsOpInterface
            ]
        > {
    let summary = "NCE version of Depthwise Convolution layer";

    let arguments = (ins
        4DTensorOf<[F16, BF16, quant_QuantizedType]>:$input,
        4DTensorOf<[F16, BF16, quant_QuantizedType]>:$filter,
        4DTensorOf<[SI32]>:$weightsTable,
        4DTensorOf<[UI8]>:$activationWindow,
        Optional<4DTensorOf<[SI32]>>:$instructionListTable,

        Confined<I64ArrayAttr, [ArrayCount<2>]>:$strides,
        VPU_PaddingAttr:$pad,

        OptionalAttr<VPU_PPETaskAttr>:$ppe,

        Confined<I64ArrayAttr, [ArrayCount<4>]>:$rawFilterShape,
        IntAttr:$activation_window_channel_length
    );

    let results = (outs
        4DTensorOf<[F16, BF16, quant_QuantizedType]>:$output
    );

    let regions = (region
        AnyRegion:$workloads
    );

    let assemblyFormat = [{
        `(` $input `,` $filter `,` $weightsTable `,` $activationWindow
         (`,` $instructionListTable^ custom<OptionalTypes>(type($instructionListTable)) ``)? `)`
        attr-dict
        custom<OptionalTypes>(type($input), type($filter), type($weightsTable), type($activationWindow)) ``
        `->` type(results)
        custom<OptionalRegion>($workloads)
    }];

    let verifier = [{
        return vpux::VPU::verifyOp(*this);
    }];

    let extraClassDeclaration = [{
        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter, vpux::NDTypeInterface output);

        static bool isSupported(vpux::IE::GroupConvolutionOp origOp, vpux::LogCb logCb);

        vpux::Shape inferAlignedFilterShape(vpux::NDTypeInterface output);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_QUANT_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT];
}

//
// NCE.MaxPool
//

def VPU_NCEMaxPoolOp :
        VPU_LayerOp<
            "NCE.MaxPool",
            [
                NoRegionArguments,
                NoTerminator,
                SingleBlock,
                DeclareOpInterfaceMethods<IE_LayoutInfoOpInterface>,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_NCEOpInterface>,
                IE_AlignedChannelsOpInterface
            ]
        > {
    let summary = "NCE version of MaxPool layer";

    let arguments = (ins
        4DTensorOf<[F16, BF16, quant_QuantizedType]>:$input,
        4DTensorOf<[SI32]>:$weightsTable,
        4DTensorOf<[UI8]>:$activationWindow,

        Confined<I64ArrayAttr, [ArrayCount<2>]>:$kernel_size,
        Confined<I64ArrayAttr, [ArrayCount<2>]>:$strides,
        VPU_PaddingAttr:$pad,

        OptionalAttr<VPU_PPETaskAttr>:$ppe,
        IntAttr:$activation_window_channel_length
    );

    let results = (outs
        4DTensorOf<[F16, BF16, quant_QuantizedType]>:$output
    );

    let regions = (region
        AnyRegion:$workloads
    );

    let assemblyFormat = [{
        `(` $input `,` $weightsTable `,` $activationWindow `)`
        attr-dict
        custom<OptionalTypes>(type($input), type($weightsTable), type($activationWindow)) ``
        `->` type(results)
        custom<OptionalRegion>($workloads)
    }];

    let verifier = [{
        return vpux::VPU::verifyOp(*this);
    }];

    let extraClassDeclaration = [{
        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output);

        static bool isSupported(vpux::IE::MaxPoolOp origOp, vpux::LogCb logCb);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_QUANT_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT];
}

//
// NCE.AveragePool
//

def VPU_NCEAveragePoolOp :
        VPU_LayerOp<
            "NCE.AveragePool",
            [
                NoRegionArguments,
                NoTerminator,
                SingleBlock,
                DeclareOpInterfaceMethods<IE_LayoutInfoOpInterface>,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_NCEOpInterface>,
                IE_AlignedChannelsOpInterface
            ]
        > {
    let summary = "NCE version of AveragePool layer";

    let arguments = (ins
        4DTensorOf<[F16, BF16, quant_QuantizedType]>:$input,

        Confined<I64ArrayAttr, [ArrayCount<2>]>:$kernel_size,
        Confined<I64ArrayAttr, [ArrayCount<2>]>:$strides,
        VPU_PaddingAttr:$pad,

        OptionalAttr<VPU_PPETaskAttr>:$ppe
    );

    let results = (outs
        4DTensorOf<[F16, BF16, quant_QuantizedType]>:$output
    );

    let regions = (region
        AnyRegion:$workloads
    );

    let assemblyFormat = [{
        `(` $input `)`
        attr-dict
        custom<OptionalTypes>(type($input)) ``
        `->` type(results)
        custom<OptionalRegion>($workloads)
    }];

    let verifier = [{
        return vpux::VPU::verifyOp(*this);
    }];

    let extraClassDeclaration = [{
        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output);

        static bool isSupported(vpux::IE::AvgPoolOp origOp, vpux::LogCb logCb, bool checkLayout = true,
                                bool checkChannelAlignment = true);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_QUANT_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT];
}

//
// NCE.Eltwise
//

def VPU_NCEEltwiseOp :
        VPU_LayerOp<
            "NCE.Eltwise",
            [
                NoRegionArguments,
                NoTerminator,
                SingleBlock,
                VPU_EltwiseOp,
                VPU_TilingBuilderOpInterface,
                DeclareOpInterfaceMethods<IE_LayoutInfoOpInterface>,
                DeclareOpInterfaceMethods<VPU_NCEOpInterface>,
                IE_AlignedChannelsOpInterface
            ]
        > {
    let summary = "NCE version of Eltwise layer";

    let arguments = (ins
        4DTensorOf<[F16, BF16, quant_QuantizedType]>:$input1,
        4DTensorOf<[F16, BF16, quant_QuantizedType]>:$input2,

        VPU_EltwiseType:$op_type,

        OptionalAttr<VPU_PPETaskAttr>:$ppe
    );

    let results = (outs
        4DTensorOf<[F16, BF16, quant_QuantizedType]>:$output
    );

    let regions = (region
        AnyRegion:$workloads
    );

    let assemblyFormat = [{
        `(` $input1 `,` $input2 `)`
        attr-dict
        custom<OptionalTypes>(type($input1), type($input2)) ``
        `->` type(results)
        custom<OptionalRegion>($workloads)
    }];

    let extraClassDeclaration = [{
        bool fitIntoCMX(vpux::NDTypeInterface input1, vpux::NDTypeInterface input2, vpux::NDTypeInterface output);

        static bool isSupported(mlir::Operation* op, bool allowDifferentScales, bool allowDifferentZp,
                                vpux::LogCb logCb);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_QUANT_MIXED_PRECISION];
}

//
// NCE.ClusterTiling
//

def VPU_NCEClusterTilingOp :
        VPU_Op<
            "NCE.ClusterTiling",
            [
                NoSideEffect,
                IsolatedFromAbove,
                DeclareOpInterfaceMethods<RegionBranchOpInterface,
                    ["getSuccessorEntryOperands", "getNumRegionInvocations"]>,
                SingleBlockImplicitTerminator<"YieldOp">
            ]
        > {
    let summary = "Operation that encapsulates details of tiling operation between clusters";

    let arguments = (ins
        Variadic<AnyTypeOf<[4DTensorOf<[F16, BF16, SI32, UI8, quant_QuantizedType]>, VPU_DistributedTensor]>>:$operands
    );

    let results = (outs
        Variadic<AnyTypeOf<[4DTensorOf<[F16, BF16, SI32, UI8, quant_QuantizedType]>, VPU_DistributedTensor]>>:$results
    );

    let regions = (region SizedRegion<1>:$body);

    let printer = [{ vpux::VPU::print(p, *this); }];
    let parser = [{ return vpux::VPU::parse$cppClass(parser, result); }];
    let verifier = [{ return vpux::VPU::verifyOp(*this); }];

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "mlir::TypeRange":$resultTypes, "mlir::ValueRange":$operands,
            "llvm::function_ref<void(mlir::OpBuilder&, mlir::Location, mlir::ValueRange)>":$bodyBuilder)>,
    ];

    let extraClassDeclaration = [{
      using BodyBuilderFn =
          llvm::function_ref<void(mlir::OpBuilder&, mlir::Location, mlir::ValueRange)>;

          template <typename T>
          T getInnerTaskOpOfType();
    }];

    let hasCanonicalizer = 1;
}

//
// YieldOp
//

def VPU_YieldOp :
        VPU_Op<
            "Yield",
            [
                HasParent<"NCEClusterTilingOp">,
                DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>,
                NoSideEffect,
                Terminator
            ]
        > {
    let summary = "Terminator for NCE.ClusterTiling operation";

    let arguments = (ins Variadic<4DTensorOf<[F16, BF16, SI32, UI8, quant_QuantizedType]>>:$operands);

    let assemblyFormat = [{
        $operands
        custom<OptionalTypes>(type($operands)) ``
        attr-dict
    }];

    let verifier = [{ return vpux::VPU::verifyOp(*this); }];
}

//
// DistributedCastOp
//

def VPU_DistributedCastOp :
        VPU_Op<
            "DistributedCast"
        > {
    let summary = "Operation that casts one DistributedTensor type to another.";

    let description = [{
        Used to cast one DistributedTensor type to another and help with NNCMX retention
        of data.

        Currently following distribution mode pairs are compatible:

        DUPLICATED|SEGMENTED -> DUPLICATED ## needed for K cluster tiling
    }];

    let arguments = (ins
        VPU_DistributedTensor:$input
    );

    let results = (outs
        VPU_DistributedTensor:$output
    );

    let assemblyFormat = [{
        `(` $input `:` type($input) `)`
        attr-dict
        `->` type($output)
    }];

    let hasFolder = 1;

    let verifier = [{ return vpux::VPU::verifyOp(*this); }];
}

//
// GroupSparseTensor
//

def VPU_GroupSparseTensorOp :
        VPU_Op<
            "GroupSparseTensor",
            [
                DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
                AttrSizedOperandSegments
            ]
        > {
    let summary = "Groups sparse data and metadata into a single value";

    let arguments = (ins
        4DTensorOf<[F16, BF16, quant_QuantizedType]>:$data,
        Optional<I1Tensor>:$sparsityMap,
        Optional<I32Tensor>:$storageElementTable
    );

    let results = (outs
        VPU_SparseTensor:$output
    );

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$data)
        >,
        OpBuilder<
            (ins "mlir::Value":$data, "mlir::Value":$sparsityMap)
        >
    ];

    let assemblyFormat = [{
        `(` $data `:` type($data) (`,` $sparsityMap^ `:` type($sparsityMap))? (`,` $storageElementTable^ `:` type($storageElementTable))? `)`
        attr-dict `->` type(results)
    }];
}

//
// SliceOp
//

def VPU_SliceOp :
        VPU_LayerOp<
            "Slice"
        > {
    let summary = "Extract single slice from ranked tensor or distributed tensor";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$source,
        I64ArrayAttr:$static_offsets,
        I64ArrayAttr:$static_sizes
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$result
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
}

//
// ConcatOp
//

def VPU_ConcatOp :
        VPU_LayerOp<
            "Concat"
        > {
    let summary = "VPU Concat layer";

    let arguments = (ins
        Variadic<AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>>:$inputs,

        OptionalAttr<IE_ConcatAttrs>:$per_axis,
        OptionalAttr<I64ArrayOfArraysAttr>:$static_offsets
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];

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
    let hasFolder = 1;
}

//
// Tanh
//

def VPU_TanhOp :
        VPU_LayerOp<
            "Tanh",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Tanh VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Sin
//

def VPU_SinOp :
        VPU_LayerOp<
            "Sin",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Sin VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Cos
//

def VPU_CosOp :
        VPU_LayerOp<
            "Cos",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Cos VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Sqrt
//

def VPU_SqrtOp :
        VPU_LayerOp<
            "Sqrt",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Sqrt VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Sinh
//

def VPU_SinhOp :
        VPU_LayerOp<
            "Sinh",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Sinh VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Cosh
//

def VPU_CoshOp :
        VPU_LayerOp<
            "Cosh",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Cosh VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Asinh
//

def VPU_AsinhOp :
        VPU_LayerOp<
            "Asinh",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Asinh VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Acosh
//

def VPU_AcoshOp :
        VPU_LayerOp<
            "Acosh",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Acosh VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Abs
//

def VPU_AbsOp :
        VPU_LayerOp<
            "Abs",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Abs VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Atan
//

def VPU_AtanOp :
        VPU_LayerOp<
            "Atan",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Atan VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Asin
//

def VPU_AsinOp :
        VPU_LayerOp<
            "Asin",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Asin VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Acos
//

def VPU_AcosOp :
        VPU_LayerOp<
            "Acos",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Acos VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Atanh
//

def VPU_AtanhOp :
        VPU_LayerOp<
            "Atanh",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Atanh VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// HSigmoidOp
//

def VPU_HSigmoidOp :
        VPU_LayerOp<
            "HSigmoid",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "HSigmoid VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Log
//

def VPU_LogOp :
        VPU_LayerOp<
            "Log",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Log VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Gelu
//

def VPU_GeluOp :
        VPU_LayerOp<
            "Gelu",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Gelu VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Exp
//

def VPU_ExpOp :
        VPU_LayerOp<
            "Exp",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Exp VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// HSwish
//

def VPU_HSwishOp :
        VPU_LayerOp<
            "HSwish",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "HSwish VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Floor
//

def VPU_FloorOp :
        VPU_LayerOp<
            "Floor",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Floor VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Round
//

def VPU_RoundOp :
        VPU_LayerOp<
            "Round",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Round VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        IE_RoundMode:$mode
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Mish
//

def VPU_MishOp :
        VPU_LayerOp<
            "Mish",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Mish VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Erf
//

def VPU_ErfOp :
        VPU_LayerOp<
            "Erf",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Erf VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Clamp
//

def VPU_ClampOp :
        VPU_LayerOp<
            "Clamp",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Clamp VPU layer";

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
// Elu
//

def VPU_EluOp :
        VPU_LayerOp<
            "Elu",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Elu VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        F64Attr:$x
    );


    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Sigmoid
//

def VPU_SigmoidOp :
        VPU_LayerOp<
            "Sigmoid",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Sigmoid VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// HardSigmoidOp
//

def VPU_HardSigmoidOp :
        VPU_LayerOp<
            "HardSigmoid",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "HardSigmoid VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        F64Attr:$alpha_value,
        F64Attr:$beta_value
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// SeluOp
//

def VPU_SeluOp :
        VPU_LayerOp<
            "Selu",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Selu VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        F64Attr:$alpha_value,
        F64Attr:$lambda_value
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// NormalizeIE
//

def VPU_NormalizeIEOp :
        VPU_LayerOp<
            "NormalizeIE"
        > {
    let summary = "NormalizeIE VPU layer";

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
// Ceiling
//

def VPU_CeilingOp :
        VPU_LayerOp<
            "Ceiling",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Ceiling VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// SoftPlus
//

def VPU_SoftPlusOp :
        VPU_LayerOp<
            "SoftPlus",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "SoftPlus VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Convert
//

def VPU_ConvertOp :
        VPU_LayerOp<
            "Convert",
            [
                DeclareOpInterfaceMethods<CastOpInterface>,
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Convert VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        TypeAttr:$dstElemType
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// SoftMax
//

def VPU_SoftMaxOp :
        VPU_LayerOp<
            "SoftMax"
        > {
    let summary = "SoftMax VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        IntAttr:$axisInd
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// PerAxisTile
//

def VPU_PerAxisTileOp :
        VPU_LayerOp<
            "PerAxisTile"
        > {
    let summary = "Per axis Tile VPU layer";

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
// ReLU
//

def VPU_ReLUOp :
        VPU_LayerOp<
            "ReLU",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "ReLU VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// LogicalNot
//

def VPU_LogicalNotOp :
        VPU_LayerOp<
            "LogicalNot",
            [
                VPU_EltwiseOp
            ]
        > {
    let summary = "Logical Not VPU layer";

    let arguments = (ins
        RankedTensorOf<[I8, F16, F32]>:$input1
    );

    let results = (outs
        RankedTensorOf<[I8, F16, F32]>:$output
    );
}

//
// Convolution
//

def VPU_ConvolutionOp :
        VPU_LayerOp<
            "Convolution",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "Convolution VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$filter,
        Optional<RankedTensorOf<[F16, F32]>>:$bias,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        I64ArrayAttr:$dilations,

        OptionalAttr<IE_PostOp>:$post_op
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Gather
//

def VPU_GatherOp :
        VPU_LayerOp<
            "Gather"
        > {
    let summary = "Gather VPU layer";

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
}

//
// GatherElements
//

def VPU_GatherElementsOp :
        VPU_LayerOp<
              "GatherElements"
        > {
    let summary = "GatherElements VPU layer";

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
// ScatterNDUpdate
//

def VPU_ScatterNDUpdateOp :
        VPU_LayerOp<
            "ScatterNDUpdate"
        > {
    let summary = "ScatterNDUpdate VPU layer";

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
// Broadcast
//

def VPU_BroadcastOp :
        VPU_LayerOp<
            "Broadcast"
        > {
    let summary = "Broadcast VPU layer";

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
// BucketizeOp
//

def VPU_BucketizeOp :
        VPU_LayerOp<
            "Bucketize"
        > {
    let summary = "Bucketize VPU layer";

    let arguments = (ins
        AnyRankedTensor:$data,
        1DTensorOf<[AnyInteger, AnyFloat]>:$buckets,

        TypeAttr:$output_type,
        UnitAttr:$with_right_bound
    );

    let results = (outs
        RankedTensorOf<[SI32, SI64]>:$output
    );

    let verifier = [{ return vpux::VPU::verifyOp(*this); }];
}

//
// FakeQuantize
//

def VPU_FakeQuantizeOp :
        VPU_LayerOp<
            "FakeQuantize",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "FakeQuantize VPU layer";

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
// Proposal
//

def VPU_ProposalOp :
        VPU_LayerOp<
            "Proposal"
        > {
    let summary = "Proposal VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$class_probs,
        RankedTensorOf<[F16, F32]>:$bbox_deltas,
        RankedTensorOf<[F16, F32]>:$image_shape,

        IE_ProposalAttrs:$proposal_attrs
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output,
        RankedTensorOf<[F16, F32]>:$probs
    );
}

//
// Interpolate
//

def VPU_InterpolateOp :
        VPU_LayerOp<
            "Interpolate",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                AttrSizedOperandSegments
            ]
        > {
    let summary = "Interpolate VPU layer";

    let arguments = (ins
        RankedTensorOf<[UI8, F16, F32]>:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$sizes,
        Optional<RankedTensorOf<[F16, F32]>>:$scales,
        Optional<RankedTensorOf<[AnyInteger]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$sizes_attr,
        OptionalAttr<F64ArrayAttr>:$scales_attr,
        OptionalAttr<I64ArrayAttr>:$axes_attr,

        IE_InterpolateAttr:$attr
    );

    let results = (outs
        RankedTensorOf<[UI8, F16, F32]>:$output
    );
}

//
// TopK
//

def VPU_TopKOp :
        VPU_LayerOp<
            "TopK"
        > {
    let summary = "TopK VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[AnyInteger]>:$k,

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
// AdaptiveAvgPoolOp
//

def VPU_AdaptiveAvgPoolOp :
        VPU_LayerOp<
            "AdaptiveAvgPool"
        > {
    let summary = "AdaptiveAvgPool VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        1DTensorOf<[SI32, SI64]>:$pooled_spatial_shape
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// AdaptiveMaxPoolOp
//

def VPU_AdaptiveMaxPoolOp :
        VPU_LayerOp<
            "AdaptiveMaxPool"
        > {
    let summary = "AdaptiveMaxPool VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        1DTensorOf<[SI32, SI64]>:$pooled_spatial_shape,
        TypeAttr:$index_element_type
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output,
        RankedTensorOf<[SI32, SI64]>:$output_index
    );
}

//
// RegionYolo
//

def VPU_RegionYoloOp :
        VPU_LayerOp<
            "RegionYolo"
        > {
    let summary = "RegionYolo VPU layer";

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
// ReorgYolo
//

def VPU_ReorgYoloOp :
        VPU_LayerOp<
            "ReorgYolo"
        > {
    let summary = "ReorgYolo VPU layer";

    let arguments = (ins
        4DTensorOf<[AnyInteger, AnyFloat]>:$input,

        IntAttr:$stride
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// DetectionOutput
//

def VPU_DetectionOutputOp :
        VPU_LayerOp<
            "DetectionOutput",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "DetectionOutput VPU layer";

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
// MVN
//

def VPU_MVNOp :
        VPU_LayerOp<
            "MVN"
        > {
    let summary = "MVN VPU layer";

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
// ROIPooling
//

def VPU_ROIPoolingOp :
        VPU_LayerOp<
            "ROIPooling"
        > {
    let summary = "ROIPooling VPU layer";

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
// PSROIPooling
//

def VPU_PSROIPoolingOp :
        VPU_LayerOp<
            "PSROIPooling"
        > {
    let summary = "PSROIPooling VPU layer";

    let arguments = (ins
        4DTensorOf<[F16, F32]>:$input,
        2DTensorOf<[F16, F32]>:$coords,

        IntAttr:$output_dim,
        F64Attr:$spatial_scale,
        IntAttr:$group_size,
        OptionalAttr<IntAttr>:$spatial_bins_x,
        OptionalAttr<IntAttr>:$spatial_bins_y,
        OptionalAttr<IE_PSROIPoolingMode>:$mode
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// ROIAlign
//

def VPU_ROIAlignOp :
        VPU_LayerOp<
            "ROIAlign",
            [
               ResultsAreFloatLike
            ]
        > {
    let summary = "ROIAlign VPU layer";

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
// StridedSlice
//

def VPU_StridedSliceOp :
        VPU_LayerOp<
            "StridedSlice",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "StridedSlice VPU layer";

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

    let extraClassDeclaration = [{
        bool isSimplified();
    }];
}

//
// PRelu
//

def VPU_PReluOp :
        VPU_LayerOp<
            "PRelu",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "PRelu VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$negative_slope
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// LeakyRelu
//

def VPU_LeakyReluOp :
        VPU_LayerOp<
            "LeakyRelu",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "LeakyRelu VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        F64Attr:$negative_slope
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Swish
//

def VPU_SwishOp :
        VPU_LayerOp<
            "Swish",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Swish VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        Optional<RankedTensorOf<[F16, F32]>>:$beta,

        OptionalAttr<F64Attr>:$beta_value
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// ScaleShift
//

def VPU_ScaleShiftOp :
        VPU_LayerOp<
            "ScaleShift",
            [
                VPU_TilingBuilderOpInterface,
                AttrSizedOperandSegments,
                VPU_EltwiseOp
            ]
        > {
    let summary = "ScaleShift VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        Optional<RankedTensorOf<[F16, F32]>>:$weights,
        Optional<RankedTensorOf<[F16, F32]>>:$biases
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Upsampling
//

def VPU_UpsamplingOp :
        VPU_LayerOp<
            "Upsampling"
        > {
    let summary = "Upsampling VPU layer";

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
// GRN
//

def VPU_GRNOp :
        VPU_LayerOp<
            "GRN"
        > {
    let summary = "GRN VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        F64Attr:$bias
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Negative
//

def VPU_NegativeOp :
        VPU_LayerOp<
            "Negative",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Negative VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Sign
//

def VPU_SignOp :
        VPU_LayerOp<
            "Sign",
            [
                VPU_EltwiseOp
            ]
        > {
    let summary = "Sign VPU layer";

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

def VPU_FullyConnectedOp:
        VPU_LayerOp<
            "FullyConnected"
        > {
    let summary = "FullyConnected VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$weights,
        Optional<RankedTensorOf<[F16, F32]>>:$bias
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// CTCGreedyDecoder
//

def VPU_CTCGreedyDecoderOp :
        VPU_LayerOp<
            "CTCGreedyDecoder"
        > {
    let summary = "CTCGreedyDecoder VPU layer";

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
// CTCGreedyDecoderSeqLen
//

def VPU_CTCGreedyDecoderSeqLenOp :
        VPU_LayerOp<
            "CTCGreedyDecoderSeqLen"
        > {
    let summary = "CTCGreedyDecoderSeqLen VPU layer";

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
// Pad
//

def VPU_PadOp :
        VPU_LayerOp<
            "Pad",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "Pad VPU layer";

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

    let assemblyFormat = [{
        `(` $input `)` (`[` $pads_begin^ `,` $pads_end (`,` $pad_value^)? `]`)? attr-dict `:` type(operands) `->` type(results)
    }];
}

//
// LSTMCell
//

def VPU_LSTMCellOp :
        VPU_LayerOp<
            "LSTMCell"
        > {
    let summary = "LSTMCell VPU layer";

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
// LSTMSequence
//

def VPU_LSTMSequenceOp :
        VPU_LayerOp<
            "LSTMSequence"
        > {
    let summary = "LSTMSequence VPU layer";

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
// Select
//

def VPU_SelectOp :
        VPU_LayerOp<
            "Select",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Select VPU layer";

    let arguments = (ins
        RankedTensorOf<[Bool8, SI32, F16]>:$input1,
        RankedTensorOf<[SI32, F16]>:$input2,
        RankedTensorOf<[SI32, F16]>:$input3,
        IE_AutoBroadcastType:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[SI32, F16]>:$output
    );
}

//
// SpaceToDepth
//

def VPU_SpaceToDepthOp :
        VPU_LayerOp<
            "SpaceToDepthOp"
        > {
    let summary = "SpaceToDepthOp VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        DefaultValuedAttr<IntAttr, "1">:$block_size,
        IE_SpaceToDepthMode:$mode
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// ReverseSequence
//

def VPU_ReverseSequenceOp :
        VPU_LayerOp<
            "ReverseSequence"
        > {
    let summary = "Reverse variable length sequence  VPU operation";

    let arguments = (ins
        AnyRankedTensor:$data,
        1DTensorOf<[AnyInteger]>:$seq_length,

        IntAttr:$seq_axis,
        IntAttr:$batch_axis
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// DepthToSpace
//

def VPU_DepthToSpaceOp :
        VPU_LayerOp<
            "DepthToSpace"
        > {
    let summary = "DepthToSpace VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        IntAttr:$block_size,
        IE_DepthToSpaceMode:$mode
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// ExtractImagePatches
//

def VPU_ExtractImagePatchesOp :
        VPU_LayerOp<
            "ExtractImagePatches"
        > {
    let summary = "InferenceEngine ExtractImagePatches layer";

    let arguments = (ins
        4DTensorOf<[AnyType]>:$data,

        I64ArrayAttr:$sizes,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$rates,
        IE_PadType:$autoPad
    );

    let results = (outs
        4DTensorOf<[AnyType]>:$output
    );
}

//
// YuvToRgb
//  Conversions:
//   NV12toRGB, NV12toBGR,
//   I420toRGB, I420toBGR
//

def VPU_YuvToRgbOp :
        VPU_LayerOp<
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
// M2I.ColorConvert
//

def VPU_M2IColorConvertOp :
        VPU_LayerOp<
            "M2I.ColorConvert"
        > {
    let summary = "M2I version for color-convert operations";

    let arguments = (ins
        4DTensorOf<[UI8, F16]>:$input,
        IE_ColorFmt:$inFmt,
        IE_ColorFmt:$outFmt
    );

    let results = (outs
         4DTensorOf<[UI8, F16]>:$output
    );

    let assemblyFormat = [{
        `(` $input `)`
        attr-dict
        custom<OptionalTypes>(type($input)) ``
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        static bool fitIntoCMX(mlir::Operation* op, vpux::NDTypeInterface input, vpux::NDTypeInterface output);
        static bool isSupported(vpux::IE::YuvToRgbOp origOp, vpux::LogCb logCb);
    }] # baseExtraClassDeclaration;
}

//
// M2I.Resize
//

def VPU_M2IResizeOp :
        VPU_LayerOp<
            "M2I.Resize"
        > {
    let summary = "M2I version for resize operations";

    let arguments = (ins
        4DTensorOf<[UI8, F16]>:$input,

        I64ArrayAttr:$sizes,
        I64ArrayAttr:$axes
    );

    let results = (outs
         4DTensorOf<[UI8, F16]>:$output
    );

    let assemblyFormat = [{
        `(` $input `)`
        attr-dict
        custom<OptionalTypes>(type($input)) ``
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        static bool fitIntoCMX(mlir::Operation* op, vpux::NDTypeInterface input, vpux::NDTypeInterface output);
        static bool isSupported(vpux::IE::InterpolateOp origOp, vpux::LogCb logCb);
    }] # baseExtraClassDeclaration;
}

//
// M2I.Norm
//

def VPU_M2INormOp :
        VPU_LayerOp<
            "M2I.Norm"
        > {
    let summary = "M2I version for BatchNormInference";

    let arguments = (ins
        4DTensorOf<[F16]>:$input,

        F64ArrayAttr:$gamma_value,
        F64ArrayAttr:$beta_value,
        F64ArrayAttr:$mean_value,
        F64ArrayAttr:$variance_value,

        F64Attr:$eps
    );

    let results = (outs
         4DTensorOf<[F16]>:$output
    );

    let assemblyFormat = [{
        `(` $input `)`
        attr-dict
        custom<OptionalTypes>(type($input)) ``
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        static bool fitIntoCMX(mlir::Operation* op, vpux::NDTypeInterface input, vpux::NDTypeInterface output);
        static bool isSupported(vpux::IE::BatchNormInferenceOp origOp, vpux::LogCb logCb);
    }] # baseExtraClassDeclaration;
}

//
// M2I.Task
//

def VPU_M2ITaskOp :
        VPU_LayerOp<
            "M2I.Task"
        > {
    let summary = "M2I full task op";

    let arguments = (ins
        4DTensorOf<[UI8, F16]>:$input,

        BoolAttr:$do_csc,
        BoolAttr:$do_norm,
        VPU_M2iColorFmt:$inFmt,
        VPU_M2iColorFmt:$outFmt,
        OptionalAttr<I64ArrayAttr>:$sizes,
        OptionalAttr<I64ArrayAttr>:$axes,
        OptionalAttr<F64ArrayAttr>:$norm
    );

    let results = (outs
         4DTensorOf<[UI8, F16]>:$output
    );

    let assemblyFormat = [{
        `(` $input `)`
        attr-dict
        custom<OptionalTypes>(type($input)) ``
        `->` type(results)
    }];
}

//
// Tile
//

def VPU_TileOp :
        VPU_LayerOp<
            "Tile"
        > {
    let summary = "Tile VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[AnyInteger]>:$repeats
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// Split
//

def VPU_SplitOp :
        VPU_LayerOp<
            "Split"
        > {
    let summary = "Split VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<AnyRankedTensor>:$axis,

        IntAttr:$num_splits,
        OptionalAttr<IntAttr>:$axis_value
    );

    let results = (outs
        Variadic<AnyRankedTensor>:$outputs
    );

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
}

//
// Power
//

def VPU_PowerOp :
        VPU_LayerOp<
            "Power",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Power VPU layer";

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
// Add
//

def VPU_AddOp :
        VPU_LayerOp<
            "Add",
            [
                VPU_TilingBuilderOpInterface,
                Commutative,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Add VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input1,
        RankedTensorOf<[F16, F32]>:$input2,

        IE_AutoBroadcastType:$auto_broadcast,
        OptionalAttr<IE_PostOp>:$post_op
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Divide
//

def VPU_DivideOp :
        VPU_LayerOp<
            "Divide",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Divide VPU layer";

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
// SquaredDiff
//

def VPU_SquaredDifferenceOp :
        VPU_LayerOp<
            "SquaredDiff",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "SquaredDiff VPU layer";

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
// FloorMod
//

def VPU_FloorModOp :
        VPU_LayerOp<
            "FloorMod",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "FloorMod VPU layer";

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
// Less
//

def VPU_LessOp :
        VPU_LayerOp<
            "Less",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Less VPU layer";

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
// LessEqual
//

def VPU_LessEqualOp :
        VPU_LayerOp<
            "LessEqual",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "LessEqual VPU layer";

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
// Greater
//

def VPU_GreaterOp :
        VPU_LayerOp<
            "Greater",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Greater VPU layer";

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
// GreaterEqual
//

def VPU_GreaterEqualOp :
        VPU_LayerOp<
            "GreaterEqual",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "GreaterEqual VPU layer";

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
// LogicalOr
//

def VPU_LogicalOrOp :
        VPU_LayerOp<
            "LogicalOr",
            [
                VPU_TilingBuilderOpInterface,
                Commutative,
                VPU_EltwiseOp
            ]
        > {
    let summary = "LogicalOr VPU layer";

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
// LogicalXor
//

def VPU_LogicalXorOp :
        VPU_LayerOp<
            "LogicalXor",
            [
                VPU_TilingBuilderOpInterface,
                Commutative,
                VPU_EltwiseOp
            ]
        > {
    let summary = "LogicalXor VPU layer";

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
// Multiply
//

def VPU_MultiplyOp :
        VPU_LayerOp<
            "Multiply",
            [
                VPU_TilingBuilderOpInterface,
                Commutative,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Multiply VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input1,
        RankedTensorOf<[F16, F32]>:$input2,

        IE_AutoBroadcastType:$auto_broadcast,
        OptionalAttr<IE_PostOp>:$post_op
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// And
//

def VPU_AndOp :
        VPU_LayerOp<
            "And",
            [
                VPU_TilingBuilderOpInterface,
                Commutative,
                VPU_EltwiseOp
            ]
        > {
    let summary = "And VPU layer";

    let arguments = (ins
        RankedTensorOf<[I8, F16, F32]>:$input1,
        RankedTensorOf<[I8, F16, F32]>:$input2,

        IE_AutoBroadcastType:$auto_broadcast,
        OptionalAttr<IE_PostOp>:$post_op
    );

    let results = (outs
        RankedTensorOf<[I8, F16, F32]>:$output
    );
}

//
// GroupConvolution
//

def VPU_GroupConvolutionOp :
        VPU_LayerOp<
            "GroupConvolution",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "GroupConvolution VPU layer";

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
}

//
// AvgPool
//

def VPU_AvgPoolOp :
        VPU_LayerOp<
            "AvgPool",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "AvgPool VPU layer";

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
// MaxPool
//

def VPU_MaxPoolOp :
        VPU_LayerOp<
            "MaxPool",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "MaxPool VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        I64ArrayAttr:$kernel_size,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        IE_RoundingType:$rounding_type,

        OptionalAttr<IE_PostOp>:$post_op
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Reshape
//

def VPU_ReshapeOp :
        VPU_LayerOp<
            "Reshape",
            [
                DeclareOpInterfaceMethods<IE_ElemTypeInfoOpInterface>
            ]
        > {
    let summary = "Reshape VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$shape,

        UnitAttr:$special_zero,
        OptionalAttr<I64ArrayAttr>:$shape_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
}

//
// Squeeze
//

def VPU_SqueezeOp :
        VPU_LayerOp<
            "Squeeze",
            [
                DeclareOpInterfaceMethods<IE_LayoutInfoOpInterface>
            ]
        > {
    let summary = "Squeeze VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$axes_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
}

//
// Unsqueeze
//

def VPU_UnsqueezeOp :
        VPU_LayerOp<
            "Unsqueeze",
            [
                DeclareOpInterfaceMethods<IE_LayoutInfoOpInterface>
            ]
        > {

    let summary = "Unsqueeze VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$axes_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
}

//
// LRN
//

def VPU_LRNOp :
        VPU_LayerOp<
            "LRN"
        > {
    let summary = "LRN VPU layer";

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
// LRN_IE
//

def VPU_LRN_IEOp :
        VPU_LayerOp<
            "LRN_IE"
        > {
    let summary = "LRN_IE VPU layer";

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
// ReduceMax
//

def VPU_ReduceMaxOp :
        VPU_LayerOp<
            "ReduceMax"
        > {
    let summary = "ReduceMax VPU layer";

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
// ReduceMean
//

def VPU_ReduceMeanOp :
        VPU_LayerOp<
            "ReduceMean"
        > {
    let summary = "ReduceMean VPU Layer";

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
// ReduceSum
//

def VPU_ReduceSumOp :
        VPU_LayerOp<
            "ReduceSum"
        > {
    let summary = "ReduceSum VPU layer";

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
// ReduceL1
//

def VPU_ReduceL1Op :
        VPU_LayerOp<
            "ReduceL1"
        > {
    let summary = "ReduceL1 VPU layer";

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
// ReduceLogicalOr
//

def VPU_ReduceLogicalOrOp :
        VPU_LayerOp<
            "ReduceLogicalOr"
        > {
    let summary = "ReduceLogicalOr VPU layer";

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
// ReduceLogicalAnd
//

def VPU_ReduceLogicalAndOp :
        VPU_LayerOp<
            "ReduceLogicalAnd"
        > {
    let summary = "ReduceLogicalAnd VPU layer";

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
// ReduceProd
//

def VPU_ReduceProdOp :
        VPU_LayerOp<
            "ReduceProd"
        > {
    let summary = "ReduceProd VPU layer";

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
// ReduceMin
//

def VPU_ReduceMinOp :
        VPU_LayerOp<
            "ReduceMin"
        > {
    let summary = "ReduceMin VPU layer";

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
// ReduceL2
//

def VPU_ReduceL2Op :
        VPU_LayerOp<
            "ReduceL2"
        > {
    let summary = "ReduceL2 VPU layer";

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
// Minimum
//

def VPU_MinimumOp :
        VPU_LayerOp<
            "Minimum",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
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
// Maximum
//

def VPU_MaximumOp :
        VPU_LayerOp<
            "Maximum",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Maximum VPU layer";

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
// Quantize
//

def VPU_QuantizeOp :
        VPU_LayerOp<"Quantize",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Quantize VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        TypeAttr:$dstElemType
    );

    let results = (outs
        RankedTensorOf<[quant_QuantizedType]>:$output
    );
}

//
// Dequantize
//

def VPU_DequantizeOp :
        VPU_LayerOp<"Dequantize",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Dequantize VPU layer";

    let arguments = (ins
        RankedTensorOf<[quant_QuantizedType]>:$input,

        TypeAttr:$dstElemType
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// QuantizeCast
//

def VPU_QuantizeCastOp :
        VPU_LayerOp<
            "QuantizeCast"
        > {
    let summary = "Quantize Cast VPU layer";

    let arguments = (ins
        RankedTensorOf<[SI8, UI8, quant_QuantizedType]>:$input,

        TypeAttr:$dstElemType
    );

    let results = (outs
        RankedTensorOf<[SI8, UI8, quant_QuantizedType]>:$output
    );
}

//
// Deconvolution (ConvolutionBackprop in ngraph)
//

def VPU_DeconvolutionOp:
        VPU_LayerOp<
            "Deconvolution"
        > {
    let summary = "Deconvolution VPU layer";

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
// Expand
//

def VPU_ExpandOp :
        VPU_LayerOp<
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

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$input, "vpux::Optional<vpux::ShapeRef>":$pads_begin, "vpux::Optional<vpux::ShapeRef>":$pads_end)
        >
    ];

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
}

//
// Subtract
//

def VPU_SubtractOp :
        VPU_LayerOp<
            "Subtract",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Subtract VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input1,
        RankedTensorOf<[F16, F32]>:$input2,

        IE_AutoBroadcastType:$auto_broadcast,
        OptionalAttr<IE_PostOp>:$post_op
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// MemPermute
//

def VPU_MemPermuteOp :
        VPU_LayerOp<
            "MemPermute",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "MemPermute VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        AffineMapAttr:$dst_order,
        AffineMapAttr:$mem_perm
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 0;
}

//
// PermuteCast
//

def VPU_PermuteCastOp :
        VPU_LayerOp<
            "PermuteCast"
        > {
    let summary = "PermuteCast VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        AffineMapAttr:$dst_order,
        AffineMapAttr:$mem_perm
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
}

//
// Equal
//

def VPU_EqualOp :
        VPU_LayerOp<
            "Equal",
            [
                VPU_TilingBuilderOpInterface,
                Commutative,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Equal VPU layer";

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
// AffineReshape
//

def VPU_AffineReshapeOp :
        VPU_LayerOp<
            "AffineReshape",
            [
                DeclareOpInterfaceMethods<IE_LayoutInfoOpInterface>,
                DeclareOpInterfaceMethods<IE_ElemTypeInfoOpInterface>
            ]
        > {
    let summary = "AffineReshape VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        I64ArrayOfArraysAttr:$dim_mapping,
        I64ArrayAttr:$shape_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
}

//
// NotEqual
//

def VPU_NotEqualOp :
        VPU_LayerOp<
            "NotEqual",
            [
                VPU_TilingBuilderOpInterface,
                Commutative,
                VPU_EltwiseOp
            ]
        > {
    let summary = "NotEqual VPU layer";

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
// Copy
//

def VPU_CopyOp :
        VPU_LayerOp<
            "Copy"
        > {
    let summary = "Copy VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        OptionalAttr<IndexedSymbolAttr>:$out_mem_space
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
    let hasCanonicalizer = 1;

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
}

//
// ExpandDilated
//

def VPU_ExpandDilatedOp :
        VPU_LayerOp<
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
}

//
// StorageElementTable
//

def VPU_StorageElementTableOp :
        VPU_Op<
            "StorageElementTable",
            [
                NoSideEffect,
                DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>
            ]
        > {
    let summary = "Declares Storage Element Pointers table";

    let description = [{
        Declares Storage Element Pointers table with defined width, 
        height, number of output pixels each SE contains(seSize), amount of SE 
        per tensor depth(seDepth) and not empty list of base pointers for each 
        element.
    }];

    let arguments = (ins
        IntAttr:$seDepth,
        IntAttr:$seSize,
        IntAttr:$height,
        IntAttr:$width,
        I32ArrayAttr:$base_ptrs
    );

    let results = (outs
        RankedTensorOf<[UI32]>:$output
    );

    let verifier = [{ return vpux::VPU::verifyOp(*this); }];

    let assemblyFormat = [{
         attr-dict `->` type(results)
    }];

    let builders = [
        OpBuilder<(ins
            CArg<"int64_t">:$seDepth,
            CArg<"int64_t">:$seSize,
            CArg<"int64_t">:$height,
            CArg<"int64_t">:$width,
            CArg<"int32_t">:$base_ptr
        )>
    ];

}

#endif
