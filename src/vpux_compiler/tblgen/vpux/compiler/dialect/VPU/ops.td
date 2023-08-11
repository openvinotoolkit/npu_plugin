//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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

class VPU_Op<string mnemonic, list<Trait> traits = []> :
        Op<
            VPU_Dialect,
            mnemonic,
            traits
        >;

class VPU_LayerOp<string mnemonic, list<Trait> traits = []> :
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
            return vpux::areTypesCompatible(lhs, rhs,
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
                    "vpux::VPU::NCEEltwiseOp",
                    "vpux::VPU::NCEPermuteQuantizeOp",
                    "vpux::VPU::NCECompressConvolutionOp",
                    "vpux::VPU::NCEInterpolateOp"
                ]>
            ]
        > {
    let summary = "Workload for a single DPU tile";

    let arguments = (ins
        Confined<I64ArrayAttr, [ArrayCount<4>]>:$outOffsets,
        Confined<I64ArrayAttr, [ArrayCount<4>]>:$outSizes,
        OptionalAttr<Confined<I64ArrayAttr, [ArrayCount<4>]>>:$inOffsets,
        OptionalAttr<Confined<I64ArrayAttr, [ArrayCount<4>]>>:$inSizes,
        VPU_PaddingAttr:$pad,
        VPU_MPEMode:$mpe_mode,
        OptionalAttr<IntAttr>:$cluster_id
    );

    let builders = [
        OpBuilder<(ins
            "mlir::ArrayAttr":$outOffsets,
            "mlir::ArrayAttr":$outSizes,
            "vpux::VPU::PaddingAttr":$kernelFunction,
            "vpux::VPU::MPEMode":$mpe_mode
        )>,
        OpBuilder<(ins
            "mlir::ArrayAttr":$outOffsets,
            "mlir::ArrayAttr":$outSizes,
            "vpux::VPU::PaddingAttr":$kernelFunction,
            "vpux::VPU::MPEModeAttr":$mpe_mode,
            "mlir::IntegerAttr":$cluster_id
        )>,
        OpBuilder<(ins
            "mlir::ArrayAttr":$outOffsets,
            "mlir::ArrayAttr":$outSizes,
            "vpux::VPU::PaddingAttr":$kernelFunction,
            "vpux::VPU::MPEMode":$mpe_mode,
            "mlir::IntegerAttr":$cluster_id
        )>
    ];

    let assemblyFormat = [{
        (`inOffsets` $inOffsets^ )? (`inSizes` $inSizes^ )? `outOffsets` $outOffsets `outSizes` $outSizes $pad $mpe_mode attr-dict-with-keyword
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
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_SparseOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                AttrSizedOperandSegments,
                IE_AlignedChannelsOpInterface
            ]
        > {
    let summary = "NCE version of Convolution layer";

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor]>:$input,
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor]>:$filter,
        4DTensorOf<[SI32]>:$weightsTable,
        Optional<4DTensorOf<[UI8]>>:$activationWindow,
        Optional<4DTensorOf<[SI32]>>:$instructionListTable,

        Confined<I64ArrayAttr, [ArrayCount<2>]>:$strides,
        VPU_PaddingAttr:$pad,

        OptionalAttr<VPU_PPETaskAttr>:$ppe,

        Confined<I64ArrayAttr, [ArrayCount<4>]>:$rawFilterShape,
        OptionalAttr<IntAttr>:$activation_window_channel_length,

        OptionalAttr<VPU_MultiClusterStrategy>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor]>:$output
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

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter, vpux::NDTypeInterface output, Byte reservedMem);

        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter, vpux::NDTypeInterface output);

        static bool isSupported(vpux::IE::ConvolutionOp origOp, vpux::LogCb logCb, bool checkLayout = false,
                                bool checkChannelAlignment = false);

        vpux::Shape inferAlignedFilterShape(vpux::NDTypeInterface input, vpux::NDTypeInterface output);

        mlir::LogicalResult verifyChannels();

        mlir::Value getWeightsOperand() {
            return filter();
        }

        mlir::Value getWeightsTableOperand() {
            return weightsTable();
        }

        bool isVFSupported();
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_QUANT_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT,
                               IE_TypeComparisonMode_ALLOW_GROUPED_OUTPUT];
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
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_SparseOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                IE_AlignedChannelsOpInterface
            ]
        > {
    let summary = "NCE version of Depthwise Convolution layer";

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor]>:$input,
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor]>:$filter,
        4DTensorOf<[SI32]>:$weightsTable,
        4DTensorOf<[UI8]>:$activationWindow,
        Optional<4DTensorOf<[SI32]>>:$instructionListTable,

        Confined<I64ArrayAttr, [ArrayCount<2>]>:$strides,
        VPU_PaddingAttr:$pad,

        OptionalAttr<VPU_PPETaskAttr>:$ppe,

        Confined<I64ArrayAttr, [ArrayCount<4>]>:$rawFilterShape,
        IntAttr:$activation_window_channel_length,

        OptionalAttr<VPU_MultiClusterStrategy>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor]>:$output
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

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter, vpux::NDTypeInterface output, Byte reservedMem);

        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter, vpux::NDTypeInterface output);

        static bool isSupported(vpux::IE::GroupConvolutionOp origOp, vpux::LogCb logCb, bool checkLayout = false,
                                bool checkChannelAlignment = false);

        vpux::Shape inferAlignedFilterShape(vpux::NDTypeInterface output);

        mlir::Value getWeightsOperand() {
            return filter();
        }

        mlir::Value getWeightsTableOperand() {
            return weightsTable();
        }

        bool isVFSupported();
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_QUANT_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT,
                               IE_TypeComparisonMode_ALLOW_GROUPED_OUTPUT];
}

//
// NCE.CompressConvolution
//

def VPU_NCECompressConvolutionOp :
        VPU_LayerOp<
            "NCE.CompressConvolution",
            [
                NoRegionArguments,
                NoTerminator,
                SingleBlock,
                DeclareOpInterfaceMethods<IE_LayoutInfoOpInterface>,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_NCEOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                IE_AlignedChannelsOpInterface
            ]
        > {
    let summary = "NCE version of Compressed Convolution layer";

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>]>:$input,
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>]>:$filter,
        4DTensorOf<[SI32]>:$weightsTable,

        Confined<I64ArrayAttr, [ArrayCount<2>]>:$strides,
        VPU_PaddingAttr:$pad,

        OptionalAttr<VPU_PPETaskAttr>:$ppe,

        Confined<I64ArrayAttr, [ArrayCount<4>]>:$rawFilterShape,

        OptionalAttr<VPU_MultiClusterStrategy>:$multiClusterStrategy,
        IntAttr:$cm_sp_pattern
    );

    let results = (outs
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>]>:$output
    );

    let regions = (region
        AnyRegion:$workloads
    );

    let assemblyFormat = [{
        `(` $input `,` $filter `,` $weightsTable `)`
        attr-dict
        custom<OptionalTypes>(type($input), type($filter), type($weightsTable)) ``
        `->` type(results)
        custom<OptionalRegion>($workloads)
    }];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter, vpux::NDTypeInterface output, Byte reservedMem);

        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter, vpux::NDTypeInterface output);

        static bool isSupported(vpux::IE::ConvolutionOp origOp, vpux::LogCb logCb, bool checkLayout = false,
                                bool checkChannelAlignment = false);

        mlir::LogicalResult verifyChannels();

        mlir::Value getWeightsOperand() {
            return filter();
        }

        mlir::Value getWeightsTableOperand() {
            return weightsTable();
        }

        bool isVFSupported();
        int64_t getInputChannelAlignment();
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_QUANT_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT,
                               IE_TypeComparisonMode_ALLOW_GROUPED_OUTPUT];
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
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_SparseOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                IE_AlignedChannelsOpInterface
            ]
        > {
    let summary = "NCE version of MaxPool layer";

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor]>:$input,
        4DTensorOf<[SI32]>:$weightsTable,
        4DTensorOf<[UI8]>:$activationWindow,

        Confined<I64ArrayAttr, [ArrayCount<2>]>:$kernel_size,
        Confined<I64ArrayAttr, [ArrayCount<2>]>:$strides,
        VPU_PaddingAttr:$pad,

        OptionalAttr<VPU_PPETaskAttr>:$ppe,
        IntAttr:$activation_window_channel_length,

        OptionalAttr<VPU_MultiClusterStrategy>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor]>:$output
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

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output, Byte reservedMem);

        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output);

        static bool isSupported(vpux::IE::MaxPoolOp origOp, vpux::LogCb logCb, bool checkLayout = false,
                                bool checkChannelAlignment = false);

        mlir::Value getWeightsTableOperand() {
            return weightsTable();
        }

        bool isVFSupported();
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_QUANT_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT,
                               IE_TypeComparisonMode_ALLOW_GROUPED_OUTPUT];
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
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_SparseOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                IE_AlignedChannelsOpInterface
            ]
        > {
    let summary = "NCE version of AveragePool layer";

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor]>:$input,

        Confined<I64ArrayAttr, [ArrayCount<2>]>:$kernel_size,
        Confined<I64ArrayAttr, [ArrayCount<2>]>:$strides,
        VPU_PaddingAttr:$pad,

        OptionalAttr<VPU_PPETaskAttr>:$ppe,

        OptionalAttr<VPU_MultiClusterStrategy>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor]>:$output
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

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output, Byte reservedMem);

        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output);

        static bool isSupported(vpux::IE::AvgPoolOp origOp, vpux::LogCb logCb, bool checkLayout = false,
                                bool checkChannelAlignment = false);

        bool isVFSupported();
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_QUANT_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT,
                               IE_TypeComparisonMode_ALLOW_GROUPED_OUTPUT];
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
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<IE_LayoutInfoOpInterface>,
                DeclareOpInterfaceMethods<VPU_NCEOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_SparseOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                IE_AlignedChannelsOpInterface
            ]
        > {
    let summary = "NCE version of Eltwise layer";

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor]>:$input1,
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor]>:$input2,

        VPU_EltwiseType:$op_type,

        OptionalAttr<VPU_PPETaskAttr>:$ppe,

        OptionalAttr<VPU_MultiClusterStrategy>:$multiClusterStrategy,
        OptionalAttr<BoolAttr>:$is_inplace
    );

    let results = (outs
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor]>:$output
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
        bool fitIntoCMX(vpux::NDTypeInterface input1, vpux::NDTypeInterface input2, vpux::NDTypeInterface output, Byte reservedMem);

        bool fitIntoCMX(vpux::NDTypeInterface input1, vpux::NDTypeInterface input2, vpux::NDTypeInterface output);

        bool fitIntoCMX(vpux::NDTypeInterface input1, vpux::NDTypeInterface input2, Byte reservedMem);

        static bool isSupported(mlir::Operation* op, bool allowDifferentScales, bool allowDifferentZp,
                                vpux::LogCb logCb, bool checkLayout = false,
                                bool checkChannelAlignment = false);
        bool isVFSupported();

        bool availableSingleMerge();
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_QUANT_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_GROUPED_OUTPUT];
}

//
// NCE.PermuteQuantize
//

def VPU_NCEPermuteQuantizeOp :
        VPU_LayerOp<
            "NCE.PermuteQuantize",
            [
                NoRegionArguments,
                NoTerminator,
                SingleBlock,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<IE_LayoutInfoOpInterface>,
                DeclareOpInterfaceMethods<VPU_NCEOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_SparseOpInterface>,
                IE_AlignedChannelsOpInterface
            ]
        > {
    let summary = "NCE version of combined Permute and Quantization layers";

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor]>:$input,

        VPU_PaddingAttr:$pad,
        TypeAttr:$dstElemType,
        AffineMapAttr:$dstOrder,
        OptionalAttr<VPU_PPETaskAttr>:$ppe,

        OptionalAttr<VPU_MultiClusterStrategy>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor]>:$output
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

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output, Byte reservedMem);
        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output);
        static bool isSupported(vpux::IE::PermuteQuantizeOp origOp, vpux::LogCb logCb, bool checkLayout = true,
                                bool checkChannelAlignment = true);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_QUANT_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_GROUPED_OUTPUT];
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
                DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getSuccessorEntryOperands",
                                                                    "areTypesCompatible"]>,
                SingleBlockImplicitTerminator<"YieldOp">
            ]
        > {
    let summary = "Operation that encapsulates details of tiling operation between clusters";

    let arguments = (ins
        Variadic<AnyTypeOf<[4DTensorOf<[F16, BF16, SI32, UI8, quant_QuantizedType]>, VPU_DistributedTensor, VPU_SparseTensor]>>:$operands
    );

    let results = (outs
        Variadic<AnyTypeOf<[4DTensorOf<[F16, BF16, SI32, UI8, quant_QuantizedType]>, VPU_DistributedTensor, VPU_SparseTensor]>>:$results
    );

    let regions = (region SizedRegion<1>:$body);

    let hasVerifier = 1;

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "mlir::TypeRange":$resultTypes, "mlir::ValueRange":$operands,
            "llvm::function_ref<void(mlir::OpBuilder&, mlir::Location, mlir::ValueRange)>":$bodyBuilder)>,
    ];

    let extraClassDeclaration = [{
        using BodyBuilderFn =
            llvm::function_ref<void(mlir::OpBuilder&, mlir::Location, mlir::ValueRange)>;

            mlir::Operation* getInnerTaskOp();
            template <typename T>
            T getInnerTaskOpOfType();

        void print(::mlir::OpAsmPrinter& p);
        static ::mlir::ParseResult parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result);
    }];

    let hasCanonicalizer = 1;
}

//
// NCE.Interpolate
//

def VPU_NCEInterpolateOp :
        VPU_LayerOp<
            "NCE.Interpolate",
            [
                NoRegionArguments,
                NoTerminator,
                SingleBlock,
                SameVariadicOperandSize,
                IE_AlignedChannelsOpInterface,
                DeclareOpInterfaceMethods<IE_LayoutInfoOpInterface>,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_NCEOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_SparseOpInterface>,
            ]
        > {
    let summary = "NCE version of Interpolate layer";

    let arguments = (ins
        VPU_SparseTensor:$input,
        Optional<AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>>:$weights,
        Optional<4DTensorOf<[SI32]>>:$weightsTable,

        OptionalAttr<VPU_PPETaskAttr>:$ppe,
        Confined<I64ArrayAttr, [ArrayCount<4>]>:$rawFilterShape,
        OptionalAttr<VPU_MultiClusterStrategy>:$multiClusterStrategy,
        OptionalAttr<VPU_NCEInterpolateMode>:$mode
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$output
    );

    let regions = (region
        AnyRegion:$workloads
    );

    let assemblyFormat = [{
        `(` $input
            (`,` $weights^ `` custom<OptionalTypes>(type($weights)))?
            (`,` $weightsTable^ `` custom<OptionalTypes>(type($weightsTable)))?
        `)`
        attr-dict
        custom<OptionalTypes>(type($input)) ``
        `->` type(results)
        custom<OptionalRegion>($workloads)
    }];

    let extraClassDeclaration = [{
        static bool isSupported(vpux::IE::InterpolateOp origOp, vpux::LogCb logCb,
                                bool checkLayout = false, bool checkChannelAlignment = false);
        static bool isSupported(vpux::VPU::InterpolateOp origOp, vpux::LogCb logCb,
                                bool checkLayout = false, bool checkChannelAlignment = false);

        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter, vpux::NDTypeInterface output, Byte reservedMem);
        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter, vpux::NDTypeInterface output);

        mlir::Value getWeightsOperand() {
            return weights();
        }
        mlir::Value getWeightsTableOperand() {
            return weightsTable();
        }
    }] # baseExtraClassDeclaration;
}

//
// VerticalFusion
//

def VPU_VerticalFusionOp :
        VPU_Op<
            "VerticalFusion",
            [
                NoSideEffect,
                IsolatedFromAbove,
                DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getSuccessorEntryOperands",
                                                                    "areTypesCompatible"]>,
                SingleBlockImplicitTerminator<"YieldOp">
            ]
        > {
    let summary = "Operation that encapsulates details of VF subgraph";

    let arguments = (ins
        Variadic<AnyTypeOf<[4DTensorOf<[F16, BF16, SI32, UI8, quant_QuantizedType]>, VPU_DistributedTensor, VPU_SparseTensor]>>:$operands,
        I64ArrayAttr:$tilingStrategy
    );

    let results = (outs
        Variadic<AnyTypeOf<[4DTensorOf<[F16, BF16, SI32, UI8, quant_QuantizedType]>, VPU_DistributedTensor, VPU_SparseTensor]>>:$results
    );

    let regions = (region SizedRegion<1>:$body);

    let hasVerifier = 1;

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "mlir::TypeRange":$resultTypes, "mlir::ValueRange":$operands,
            "llvm::function_ref<void(mlir::OpBuilder&, mlir::Location, mlir::ValueRange)>":$bodyBuilder,
            "mlir::ArrayAttr":$tilingInfo)>,
    ];

    let extraClassDeclaration = [{
        using BodyBuilderFn =
            llvm::function_ref<void(mlir::OpBuilder&, mlir::Location, mlir::ValueRange)>;

        mlir::Operation* getFirstInnerTaskOp();

        void print(::mlir::OpAsmPrinter& p);
        static ::mlir::ParseResult parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result);
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
                HasParent<"NCEClusterTilingOp, VerticalFusionOp">,
                DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>,
                NoSideEffect,
                Terminator
            ]
        > {
    let summary = "Terminator for wrapping operation";

    let arguments = (ins
        Variadic<AnyTypeOf<[4DTensorOf<[F16, BF16, SI32, UI8, quant_QuantizedType]>, VPU_SparseTensor]>>:$operands
    );

    let assemblyFormat = [{
        $operands
        custom<OptionalTypes>(type($operands)) ``
        attr-dict
    }];

    let hasVerifier = 1;
}

//
// DistributedCastOp
//

def VPU_DistributedCastOp :
        VPU_Op<
            "DistributedCast",
            [
                VPU_ViewLikeOpInterface
            ]
        > {
    let summary = "Operation that casts one DistributedTensor type to another.";

    let description = [{
        Used to cast one DistributedTensor type to another and help with NNCMX retention
        of data.

        Currently following distribution mode pairs are compatible:

        DUPLICATED|SEGMENTED -> DUPLICATED ## needed for K cluster tiling
    }];

    let arguments = (ins
        AnyTypeOf<[VPU_DistributedTensor, VPU_SparseTensor]>:$input
    );

    let results = (outs
        AnyTypeOf<[VPU_DistributedTensor, VPU_SparseTensor]>:$output
    );

    let assemblyFormat = [{
        `(` $input `:` qualified(type($input)) `)`
        attr-dict
        `->` qualified(type($output))
    }];

    let hasFolder = 1;

    let hasVerifier = 1;
}

//
// GroupSparseTensor
//

def VPU_GroupSparseTensorOp :
        VPU_Op<
            "GroupSparseTensor",
            [
                NoSideEffect,
                DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
                AttrSizedOperandSegments,
                VPU_GroupedViewLikeOpInterface
            ]
        > {
    let summary = "Groups sparse data and metadata into a single value";

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_DistributedTensor]>:$data,
        Optional<AnyTypeOf<[I1Tensor, VPU_DistributedTensor]>>:$sparsityMap,
        Optional<AnyTypeOf<[I32Tensor, VPU_DistributedTensor]>>:$storageElementTable,

        OptionalAttr<UnitAttr>:$is_weights,
        OptionalAttr<VPU_CompressionSchemeAttr>:$compression_scheme,

        OptionalAttr<VPU_SEAttr>:$seAttr
    );

    let results = (outs
        VPU_SparseTensor:$output
    );

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$data,
                CArg<"bool", "{}">:$is_weights, CArg<"VPU::CompressionSchemeAttr", "{}">:$compression_scheme)
        >,
        OpBuilder<
            (ins "mlir::Value":$data, "mlir::Value":$sparsityMap,
                CArg<"bool", "{}">:$is_weights, CArg<"VPU::CompressionSchemeAttr", "{}">:$compression_scheme)
        >,
        OpBuilder<
            (ins "mlir::Value":$data, "mlir::Value":$sparsityMap, "mlir::Value":$storageElementTable,
                CArg<"bool", "{}">:$is_weights, CArg<"VPU::CompressionSchemeAttr", "{}">:$compression_scheme)
        >,
        OpBuilder<
            (ins "mlir::Value":$data, "mlir::Value":$sparsityMap, "mlir::Value":$storageElementTable,
                CArg<"VPU::SEAttr", "{}">:$seAttr)
        >
    ];

    let assemblyFormat = [{
        `(` $data
            (`,` $sparsityMap^ `` custom<OptionalTypes>(type($sparsityMap)))?
            (`,` $storageElementTable^ `` custom<OptionalTypes>(type($storageElementTable)))?
        `)`
        attr-dict
        `` custom<OptionalTypes>(type($data))
        `->` type(results)
    }];

    let hasCanonicalizer = 1;
}

//
// SliceOp
//

def VPU_SliceOp :
        VPU_LayerOp<
            "Slice",
            [
                VPU_ViewLikeOpInterface
            ]
        > {
    let summary = "Extract single slice from ranked tensor or distributed tensor";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor, VPU_SparseTensor]>:$source,
        I64ArrayAttr:$static_offsets,
        I64ArrayAttr:$static_sizes
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor, VPU_SparseTensor]>:$result
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
            "Concat",
            [
                VPU_ViewLikeOpInterface,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "VPU Concat layer";

    let arguments = (ins
        Variadic<AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor, VPU_SparseTensor]>>:$inputs,

        OptionalAttr<IE_ConcatAttrs>:$per_axis,
        OptionalAttr<I64ArrayOfArraysAttr>:$static_offsets,
        OptionalAttr<VPU_MultiClusterStrategy>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor, VPU_SparseTensor]>:$output
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];

    let builders = [
        OpBuilder<
            (ins "mlir::ValueRange":$inputs, "vpux::IE::ConcatAttrs":$per_axis)
        >,
        OpBuilder<
            (ins "mlir::ValueRange":$inputs, "vpux::IE::ConcatAttrs":$per_axis, "mlir::ArrayAttr":$static_offsets)
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
            (ins "mlir::Type":$outType, "mlir::ValueRange":$inputs, "vpux::IE::ConcatAttrs":$per_axis, "mlir::ArrayAttr":$static_offsets)
        >,
        OpBuilder<
            (ins "mlir::Type":$outType, "mlir::ValueRange":$inputs, "vpux::ArrayRef<vpux::Shape>":$static_offsets)
        >,
        OpBuilder<
            (ins "mlir::Type":$outType, "mlir::ValueRange":$inputs, "vpux::ArrayRef<vpux::ShapeRef>":$static_offsets)
        >,
    ];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        ::mlir::LogicalResult customVerify();
        bool fitIntoCMX(vpux::NDTypeInterface output, Byte reservedMem);
        bool fitIntoCMX(vpux::NDTypeInterface output);
    }] # baseExtraClassDeclaration;

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

//
// RollOp
//

def VPU_RollOp :
        VPU_LayerOp<
            "Roll",
            [
                VPU_SameInOutDefaultDimsOrder
            ]
        > {
    let summary = "Roll VPU layer";

    let arguments = (ins
        AnyRankedTensor:$data,
        1DTensorOf<[SI32, SI64]>:$shift,
        1DTensorOf<[SI32, SI64]>:$axes
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// Tanh
//

def VPU_TanhOp :
        VPU_LayerOp<
            "Tanh",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                VPU_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
            ]
        > {
    let summary = "Tanh VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        OptionalAttr<VPU_MultiClusterStrategy>:$multiClusterStrategy
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);

        bool availableSingleMerge();
    }] # baseExtraClassDeclaration;
}

//
// Sin
//

def VPU_SinOp :
        VPU_LayerOp<
            "Sin",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                VPU_SameInOutDefaultDimsOrder
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
                VPU_EltwiseOp,
                VPU_SameInOutDefaultDimsOrder
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
// Tan
//

def VPU_TanOp :
        VPU_LayerOp<
            "Tan",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                VPU_SameInOutDefaultDimsOrder
            ]
        > {
    let summary = "Tan VPU layer";

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
                VPU_EltwiseOp,
                VPU_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
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
                VPU_EltwiseOp,
                VPU_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
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
                VPU_EltwiseOp,
                VPU_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
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
                VPU_EltwiseOp,
                VPU_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
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
                VPU_EltwiseOp,
                VPU_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
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
                VPU_EltwiseOp,
                VPU_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
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
                VPU_EltwiseOp,
                VPU_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
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
                VPU_EltwiseOp,
                VPU_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
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
                VPU_EltwiseOp,
                VPU_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
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
                VPU_EltwiseOp,
                VPU_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
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
                VPU_EltwiseOp,
                VPU_SameInOutDimsOrder_NC_CHW_HWC_NCHW_NHWC
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
// GridSampleOp
//

def VPU_GridSampleOp :
        VPU_LayerOp<
            "GridSample",
            [
                VPU_SameInOutDefaultDimsOrder,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "GridSample VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        AnyRankedTensor:$grid,

        OptionalAttr<UnitAttr>:$align_corners,
        OptionalAttr<IE_GridSampleModeAttr>:$mode,
        OptionalAttr<IE_GridSamplePaddingModeAttr>:$padding_mode
    );

    let results = (outs
        AnyRankedTensor:$output
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
                VPU_EltwiseOp,
                VPU_SameInOutDefaultDimsOrder
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
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp,
                VPU_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
            ]
        > {
    let summary = "Gelu VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        OptionalAttr<VPU_MultiClusterStrategy>:$multiClusterStrategy
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;
}

//
// Exp
//

def VPU_ExpOp :
        VPU_LayerOp<
            "Exp",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                VPU_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
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
                VPU_EltwiseOp,
                VPU_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "HSwish VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        OptionalAttr<VPU_MultiClusterStrategy>:$multiClusterStrategy
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

}

//
// Floor
//

def VPU_FloorOp :
        VPU_LayerOp<
            "Floor",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                VPU_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
            ]
        > {
    let summary = "Floor VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32]>:$output
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
                VPU_EltwiseOp,
                VPU_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
            ]
        > {
    let summary = "Round VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        IE_RoundModeAttr:$mode
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
                VPU_EltwiseOp,
                VPU_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
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
                VPU_EltwiseOp,
                VPU_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
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
                VPU_EltwiseOp,
                VPU_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
            ]
        > {
    let summary = "Clamp VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$input,

        F64Attr:$min,
        F64Attr:$max
    );

    let results = (outs
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$output
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
                VPU_EltwiseOp,
                VPU_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
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
                VPU_EltwiseOp,
                VPU_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
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
                VPU_EltwiseOp,
                VPU_SameInOutDefaultDimsOrder
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
// EmbeddingBagOffsetsSumOp
//

def VPU_EmbeddingBagOffsetsSumOp :
        VPU_LayerOp<
            "EmbeddingBagOffsetsSum",
            [
                VPU_AnyDimsOrder
            ]
        > {
    let summary = "InferenceEngine EmbeddingBagOffsetsSum layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        I64ArrayAttr:$indices_value,
        I64ArrayAttr:$offsets_value,
        IntAttr:$default_index_value,
        F64ArrayAttr:$weights_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// EmbeddingSegmentsSumOp
//

def VPU_EmbeddingSegmentsSumOp :
        VPU_LayerOp<
            "EmbeddingSegmentsSum",
            [
                AttrSizedOperandSegments,
                VPU_AnyDimsOrder
            ]
        > {
    let summary = "EmbeddingSegmentsSum VPU layer";

    let arguments = (ins
        AnyRankedTensor:$emb_table,
        Optional<1DTensorOf<[SI64, SI32]>>:$indices,
        Optional<1DTensorOf<[SI64, SI32]>>:$segment_ids,
        Optional<1DTensorOf<[AnyInteger, AnyFloat]>>:$per_sample_weights,

        OptionalAttr<I64ArrayAttr>:$indices_value,
        OptionalAttr<I64ArrayAttr>:$segment_ids_value,  
        IntAttr:$num_segments_value,
        IntAttr:$default_index_value,
        OptionalAttr<F64ArrayAttr>:$per_sample_weights_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// EmbeddingBagPackedSumOp
//

def VPU_EmbeddingBagPackedSumOp :
        VPU_LayerOp<
            "EmbeddingBagPackedSum",
            [
                VPU_AnyDimsOrder
            ]
        > {
    let summary = "EmbeddingBagPackedSum VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$emb_table,
        2DTensorOf<[SI32, SI64]>:$indices,
        Optional<2DTensorOf<[F16, F32]>>:$per_sample_weights
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
                VPU_EltwiseOp,
                VPU_SameInOutDefaultDimsOrder
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
// NormalizeL2
//

def VPU_NormalizeL2Op :
        VPU_LayerOp<
            "NormalizeL2",
            [
                VPU_SameInOutDefaultDimsOrder
            ]
        > {
    let summary = "NormalizeL2 VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$data,
        1DTensorOf<[SI32, SI64]>:$axes,

        F64Attr:$eps,
        IE_EpsModeAttr:$eps_mode
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let hasVerifier = 1;
}

//
// NormalizeIE
//

def VPU_NormalizeIEOp :
        VPU_LayerOp<
            "NormalizeIE",
            [
                VPU_SameInOutDefaultDimsOrder
            ]
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
// CumSum
//

def VPU_CumSumOp :
        VPU_LayerOp<
            "CumSum",
            [
                VPU_SameInOutDefaultDimsOrder
            ]
        > {
    let summary = "CumSum VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        OptionalAttr<IntAttr>:$axis_value,
        OptionalAttr<UnitAttr>:$exclusive,
        OptionalAttr<UnitAttr>:$reverse
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
                VPU_EltwiseOp,
                VPU_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
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
                VPU_EltwiseOp,
                VPU_SameInOutDefaultDimsOrder
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
                VPU_EltwiseOp,
                VPU_SameInOutDefaultDimsOrder
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
            "SoftMax",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_SameInOutDefaultDimsOrder
            ]
        > {
    let summary = "SoftMax VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        IntAttr:$axisInd,
        OptionalAttr<VPU_MultiClusterStrategy>:$multiClusterStrategy
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::IntegerAttr":$axisInd
        )>
    ];
}

//
// LogSoftmax
//

def VPU_LogSoftmaxOp :
        VPU_LayerOp<
            "LogSoftmax",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                VPU_SameInOutDefaultDimsOrder
            ]
        > {
    let summary = "LogSoftmax VPU layer";

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
            "PerAxisTile",
            [
                VPU_SameInOutDefaultDimsOrder
            ]
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
                VPU_EltwiseOp,
                VPU_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
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
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                SameInOutDimsOrder_NCHW_CHW_NC_C
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

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # baseExtraClassDeclaration;
}

//
// Gather
//

def VPU_GatherOp :
        VPU_LayerOp<
            "Gather",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                VPU_AnyDimsOrder
            ]
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

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # baseExtraClassDeclaration;
}

//
// GatherNDOp
//

def VPU_GatherNDOp :
        VPU_LayerOp<
            "GatherND",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                VPU_AnyDimsOrder
            ]
        > {
    let summary = "GatherND VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[AnyInteger]>:$indices,

        IntAttr:$batch_dims
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # baseExtraClassDeclaration;

    let hasVerifier = 1;
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
// GatherTree
//

def VPU_GatherTreeOp :
        VPU_LayerOp<
              "GatherTree",
            [
                VPU_SameInOutDefaultDimsOrder
            ]
        > {
    let summary = "GatherTree VPU layer";

    let arguments = (ins
        AnyRankedTensor:$stepIds,
        AnyRankedTensor:$parentIds,
        AnyRankedTensor:$maxSeqLen,
        AnyRankedTensor:$endToken
    );

    let results = (outs
        AnyRankedTensor:$finalIds
    );

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # baseExtraClassDeclaration;
}

//
// ScatterNDUpdate
//

def VPU_ScatterNDUpdateOp :
        VPU_LayerOp<
            "ScatterNDUpdate",
            [
                VPU_SameInOutDefaultDimsOrder
            ]
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
// ScatterUpdate
//

def VPU_ScatterUpdateOp :
        VPU_LayerOp<
            "ScatterUpdate",
            [
                VPU_AnyDimsOrder
            ]
        > {
    let summary = "ScatterUpdate VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[AnyInteger]>:$indices,
        AnyRankedTensor:$updates,
        OptionalAttr<IntAttr>:$axis_value

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
            "Broadcast",
            [
                VPU_AnyDimsOrder
            ]
        > {
    let summary = "Broadcast VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        1DTensorOf<[AnyInteger]>:$target_shape,
        Optional<1DTensorOf<[AnyInteger]>>:$axes_mapping,

        OptionalAttr<IE_BroadcastTypeAttr>:$mode
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
            "Bucketize",
            [
                VPU_SameInOutDefaultDimsOrder
            ]
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

    let hasVerifier = 1;
}

//
// FakeQuantize
//

def VPU_FakeQuantizeOp :
        VPU_LayerOp<
            "FakeQuantize",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                VPU_SameInOutDimsOrder_NCHW_NHWC
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
        IE_AutoBroadcastTypeAttr:$auto_broadcast
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
            "Proposal",
            [
                VPU_AnyDimsOrder
            ]
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
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                AttrSizedOperandSegments
            ]
        > {
    let summary = "Interpolate VPU layer";

    let arguments = (ins
        RankedTensorOf<[UI8, F16, F32, quant_QuantizedType]>:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$sizes,
        Optional<RankedTensorOf<[F16, F32]>>:$scales,
        Optional<RankedTensorOf<[AnyInteger]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$sizes_attr,
        OptionalAttr<F64ArrayAttr>:$scales_attr,
        OptionalAttr<I64ArrayAttr>:$axes_attr,
        OptionalAttr<F64ArrayAttr>:$tile_offset_attr,
        OptionalAttr<I64ArrayAttr>:$initial_input_dims_attr,
        OptionalAttr<I64ArrayAttr>:$initial_output_dims_attr,
        OptionalAttr<I64ArrayAttr>:$initial_input_offset_attr,
        OptionalAttr<I64ArrayAttr>:$initial_output_offset_attr,
        OptionalAttr<VPU_MultiClusterStrategy>:$multiClusterStrategy,

        IE_InterpolateAttr:$attr
    );

    let results = (outs
        RankedTensorOf<[UI8, F16, F32, quant_QuantizedType]>:$output
    );

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::Value":$sizes,
            "::mlir::Value":$scales,
            "::mlir::Value":$axes,
            "::mlir::ArrayAttr":$sizes_attr,
            "::mlir::ArrayAttr":$scales_attr,
            "::mlir::ArrayAttr":$axes_attr,
            "::mlir::ArrayAttr":$tile_offset_attr,
            "::mlir::ArrayAttr":$initial_input_dims_attr,
            "::mlir::ArrayAttr":$initial_output_dims_attr,
            "vpux::IE::InterpolateAttr":$attr
        )>
    ];
}

//
// TopK
//

def VPU_TopKOp :
        VPU_LayerOp<
            "TopK",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "TopK VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[AnyInteger]>:$k,

        IntAttr:$axis,
        IE_TopKModeAttr:$mode,
        IE_TopKSortTypeAttr:$sort,
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
            "AdaptiveAvgPool",
            [
                VPU_SameInOutDefaultDimsOrder
            ]
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
            "AdaptiveMaxPool",
            [
                VPU_SameInOutDefaultDimsOrder
            ]
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

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # baseExtraClassDeclaration;
}

//
// ReorgYolo
//

def VPU_ReorgYoloOp :
        VPU_LayerOp<
            "ReorgYolo",
            [
                VPU_AnyDimsOrder
            ]
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
                AttrSizedOperandSegments,
                VPU_AnyDimsOrder
            ]
        > {
    let summary = "DetectionOutput VPU layer";

    let arguments = (ins
        2DTensorOf<[AnyFloat]>:$in_box_logits,
        2DTensorOf<[AnyFloat]>:$in_class_preds,
        3DTensorOf<[AnyFloat]>:$in_proposals,
        Optional<2DTensorOf<[AnyFloat]>>:$in_additional_preds,
        Optional<2DTensorOf<[AnyFloat]>>:$in_additional_proposals,

        IE_DetectionOutputAttr:$attr
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// DetectionOutputNormalize
//

def VPU_DetectionOutputNormalizeOp:
        VPU_LayerOp<
            "DetectionOutputNormalize",
            [
                VPU_AnyDimsOrder
            ]
        > {
    let summary = "DetectionOutputNormalize VPU layer";

    let arguments = (ins
        3DTensorOf<[AnyFloat]>:$prior_boxes,

        IntAttr:$input_width,
        IntAttr:$input_height
    );

    let results = (outs
        3DTensorOf<[AnyFloat]>:$out_prior_boxes
    );

    let hasVerifier = 1;
}

//
// DetectionOutputDecodeBoxes
//

def VPU_DetectionOutputDecodeBoxesOp:
        VPU_LayerOp<
            "DetectionOutputDecodeBoxes",
            [
                VPU_AnyDimsOrder
            ]
        > {
    let summary = "DetectionOutputDecodeBoxes VPU layer";

    let arguments = (ins
        4DTensorOf<[AnyFloat]>:$box_logits,
        3DTensorOf<[AnyFloat]>:$prior_boxes,

        IE_DetectionOutputCodeTypeAttr:$code_type,
        BoolAttr:$clip_before_nms
    );

    let results = (outs
        4DTensorOf<[AnyFloat]>:$out_decoded_boxes
    );
}

//
// DetectionOutputSortTopK
//

def VPU_DetectionOutputSortTopKOp:
        VPU_LayerOp<
            "DetectionOutputSortTopK",
            [
                VPU_AnyDimsOrder
            ]
        > {
    let summary = "DetectionOutputSortTopK VPU layer";

    let arguments = (ins
        3DTensorOf<[AnyFloat]>:$class_predictions,

        F64Attr:$confidence_threshold,
        IntAttr:$top_k,
        IntAttr:$background_id
    );

    let results = (outs
        3DTensorOf<[AnyFloat]>:$out_top_k_confidence,
        3DTensorOf<[SI32]>:$out_indices,
        2DTensorOf<[SI32]>:$out_sizes
    );
}

//
// DetectionOutputSelectBoxes
//

def VPU_DetectionOutputSelectBoxesOp:
        VPU_LayerOp<
            "DetectionOutputSelectBoxes",
            [
                VPU_AnyDimsOrder
            ]
        > {
    let summary = "DetectionOutputSelectBoxes VPU layer";

    let arguments = (ins
        3DTensorOf<[AnyFloat]>:$decoded_boxes,
        3DTensorOf<[SI32]>:$indices,
        2DTensorOf<[SI32]>:$sizes,

        IntAttr:$top_k
    );

    let results = (outs
        3DTensorOf<[AnyFloat]>:$out_boxes
    );
}

//
// DetectionOutputNmsCaffe
//

def VPU_DetectionOutputNmsCaffeOp:
        VPU_LayerOp<
            "DetectionOutputNmsCaffe",
            [
                VPU_AnyDimsOrder
            ]
        > {
    let summary = "DetectionOutputNmsCaffe VPU layer";

    let arguments = (ins
        3DTensorOf<[AnyFloat]>:$top_k_confidence,
        3DTensorOf<[AnyFloat]>:$boxes,
        2DTensorOf<[SI32]>:$sizes,

        F64Attr:$nms_threshold
    );

    let results = (outs
        3DTensorOf<[AnyFloat]>:$out_confidence,
        3DTensorOf<[AnyFloat]>:$out_boxes,
        2DTensorOf<[SI32]>:$out_sizes
    );
}

//
// DetectionOutputCollectResults
//

def VPU_DetectionOutputCollectResultsOp:
        VPU_LayerOp<
            "DetectionOutputCollectResults",
            [
                VPU_AnyDimsOrder
            ]
        > {
    let summary = "DetectionOutputCollectResults VPU layer";

    let arguments = (ins
        3DTensorOf<[AnyFloat]>:$confidence,
        3DTensorOf<[AnyFloat]>:$boxes,
        2DTensorOf<[SI32]>:$sizes,

        IntAttr:$keep_top_k,
        BoolAttr:$clip_after_nms
    );

    let results = (outs
        4DTensorOf<[AnyFloat]>:$out_detections
    );
}

//
// MVN
//

def VPU_MVNOp :
        VPU_LayerOp<
            "MVN",
            [
                VPU_TilingBuilderOpInterface,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp
            ]

        > {
    let summary = "MVN1 VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        BoolAttr:$across_channels,
        BoolAttr:$normalize_variance,
        F64Attr:$eps,
        OptionalAttr<VPU_MultiClusterStrategy>:$multiClusterStrategy
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;


    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::BoolAttr":$across_channels,
            "::mlir::BoolAttr":$normalize_variance,
            "::mlir::FloatAttr":$eps
        )>
    ];
}

//
// MVN6
//

def VPU_MVN6Op :
        VPU_LayerOp<
            "MVN6",
            [
                VPU_EltwiseOp,
                VPU_TilingBuilderOpInterface
            ]
        > {
    let summary = "MVN6 VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        I64ArrayAttr:$axes,
        BoolAttr:$normalize_variance,
        F64Attr:$eps,
        IE_MvnEpsModeAttr:$eps_mode
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let extraClassDeclaration = [{
        DimArr getNonNormDims();
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # baseExtraClassDeclaration;
}

//
// ROIPooling
//

def VPU_ROIPoolingOp :
        VPU_LayerOp<
            "ROIPooling",
            [
                VPU_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
            ]
        > {
    let summary = "ROIPooling VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$coords,

        I64ArrayAttr:$output_size,
        F64Attr:$spatial_scale,
        IE_ROIPoolingMethodAttr:$method
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
            "PSROIPooling",
            [
                VPU_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
            ]
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
        OptionalAttr<IE_PSROIPoolingModeAttr>:$mode
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
        IE_ROIAlignMethodAttr:$poolingMode
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
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                SameInOutDimsOrder_NCHW_CHW_NC_C
            ]
        > {
    let summary = "StridedSlice VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        I64ArrayAttr:$begins_attr,
        I64ArrayAttr:$ends_attr,
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
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                VPU_EltwiseOp,
                VPU_SameInOutDefaultDimsOrder
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

    let hasVerifier = 1;
}

//
// LeakyRelu
//

def VPU_LeakyReluOp :
        VPU_LayerOp<
            "LeakyRelu",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                VPU_SameInOutDefaultDimsOrder
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
                VPU_EltwiseOp,
                VPU_SameInOutDefaultDimsOrder,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "Swish VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        Optional<RankedTensorOf<[F16, F32]>>:$beta,

        OptionalAttr<F64Attr>:$beta_value,
        OptionalAttr<VPU_MultiClusterStrategy>:$multiClusterStrategy
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::Value":$beta,
            "::mlir::FloatAttr":$beta_value
        )>
    ];
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

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # baseExtraClassDeclaration;
}

//
// Upsampling
//

def VPU_UpsamplingOp :
        VPU_LayerOp<
            "Upsampling",
            [
                VPU_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
            ]
        > {
    let summary = "Upsampling VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        I64ArrayAttr:$upsampling_factor,
        OptionalAttr<IE_UpsamplingPadAttr>:$pad
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
            "GRN",
            [
                VPU_SameInOutDimsOrder_NCHW_NHWC
            ]
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
                VPU_EltwiseOp,
                VPU_SameInOutDefaultDimsOrder
            ]
        > {
    let summary = "Negative VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32]>:$output
    );
}

//
// Sign
//

def VPU_SignOp :
        VPU_LayerOp<
            "Sign",
            [
                VPU_TilingBuilderOpInterface,
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

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # baseExtraClassDeclaration;
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

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # baseExtraClassDeclaration;
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

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # baseExtraClassDeclaration;
}

//
// Pad
//

def VPU_PadOp :
        VPU_LayerOp<
            "Pad",
            [
                AttrSizedOperandSegments,
                VPU_SameInOutDefaultDimsOrder
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

        IE_PadModeAttr:$mode
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
            "LSTMCell",
            [
                VPU_SameInOutDefaultDimsOrder
            ]
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
// LSTMCellOp
//

def VPU_LSTMGatesOp :
        VPU_LayerOp<
            "LSTMGates",
            [
                VPU_SameInOutDefaultDimsOrder
            ]
        > {
    let summary = "Computes LSTM activation functions";

    let description = [{
        This operation is intended to be run as a software stage after computing and adding LSTM matrix multiplications.

        - **gatesInput** - tensor of shape **[batchSize, 4 * hiddenSize]**. Formula:
            ```
            gatesInput = (inputData * weights) + (initialHiddenState * recurrenceWeights) + biases
            * - Matrix multiplication
            + - Element-wise add
            ```
        - The meaning of other operands are identical to those in LSTMCell operation.
    }];

    let arguments = (ins
        2DTensorOf<[F16, F32]>:$gatesInput,
        2DTensorOf<[F16, F32]>:$initialCellState
    );

    let results = (outs
        2DTensorOf<[F16, F32]>:$outputHiddenState,
        2DTensorOf<[F16, F32]>:$outputCellState
    );

    let hasVerifier = 1;
}

//
// LSTMSequence
//

def VPU_LSTMSequenceOp :
        VPU_LayerOp<
            "LSTMSequence",
            [
                VPU_AnyDimsOrder
            ]
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
        IE_RNNSequenceDirectionAttr:$direction
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
                VPU_EltwiseOp,
                VPU_SameInOutDefaultDimsOrder
            ]
        > {
    let summary = "Select VPU layer";

    let arguments = (ins
        RankedTensorOf<[Bool8, SI32, F16]>:$input1,
        RankedTensorOf<[SI32, F16]>:$input2,
        RankedTensorOf<[SI32, F16]>:$input3,
        IE_AutoBroadcastTypeAttr:$auto_broadcast
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
            "SpaceToDepthOp",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                VPU_SameInOutDefaultDimsOrder
            ]
        > {
    let summary = "SpaceToDepthOp VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        DefaultValuedAttr<IntAttr, "1">:$block_size,
        IE_SpaceToDepthModeAttr:$mode
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
            "ReverseSequence",
            [
                VPU_SameInOutDefaultDimsOrder
            ]
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
            "DepthToSpace",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                VPU_SameInOutDefaultDimsOrder
            ]
        > {
    let summary = "DepthToSpace VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        IntAttr:$block_size,
        IE_DepthToSpaceModeAttr:$mode,
        OptionalAttr<IE_ChannelPadding>:$padded_channels
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
            "ExtractImagePatches",
            [
                VPU_SameInOutDimsOrder_NCHW
            ]
        > {
    let summary = "InferenceEngine ExtractImagePatches layer";

    let arguments = (ins
        4DTensorOf<[AnyType]>:$data,

        I64ArrayAttr:$sizes,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$rates,
        IE_PadTypeAttr:$autoPad
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
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                AttrSizedOperandSegments,
                VPU_AnyDimsOrder
            ]
        > {
    let summary = "InferenceEngine NV12/I420 to RGB/BGR layer";

    let arguments = (ins
                 4DTensorOf<[SI8, F16, F32]> :$input1,
        Optional<4DTensorOf<[SI8, F16, F32]>>:$input2,
        Optional<4DTensorOf<[SI8, F16, F32]>>:$input3,

        IE_ColorFmtAttr:$inFmt,
        IE_ColorFmtAttr:$outFmt
    );

    let results = (outs
        4DTensorOf<[SI8, F16, F32]>:$output
    );
}

//
// RandomUniform
//

def VPU_RandomUniformOp :
        VPU_LayerOp<
            "RandomUniform"
        > {
    let summary = "RandomUniform VPU layer";

    let arguments = (ins
        1DTensorOf<[F16, F32, SI32]>:$min,
        1DTensorOf<[F16, F32, SI32]>:$max,

        I64ArrayAttr:$output_shape,
        TypeAttr:$output_type,
        IntAttr:$global_seed,
        IntAttr:$op_seed
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32]>:$output
    );

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # baseExtraClassDeclaration;
}

//
// OneHot
//

def VPU_OneHotOp :
        VPU_LayerOp<
            "OneHot"
        > {
    let summary = "InferenceEngine OneHot layer";

    let arguments = (ins
        RankedTensorOf<[SI32, SI64]> :$input,

        IntAttr:$depth,
        F64Attr:$on_value,
        F64Attr:$off_value,
        IntAttr:$axis,

        TypeAttr:$outElemType
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # baseExtraClassDeclaration;
}

//
// Tile
//

def VPU_TileOp :
        VPU_LayerOp<
            "Tile",
            [
                VPU_SameInOutDefaultDimsOrder,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "Tile VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        I64ArrayAttr:$repeats_values
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
}

//
// Split
//

def VPU_SplitOp :
        VPU_LayerOp<
            "Split",
            [
            VPU_ViewLikeOpInterface
            ]
        > {
    let summary = "Split VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$input,
        Optional<AnyRankedTensor>:$axis,

        IntAttr:$num_splits,
        OptionalAttr<IntAttr>:$axis_value
    );

    let results = (outs
        Variadic<AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>>:$outputs
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
                VPU_EltwiseOp,
                SameInOutDimsOrder_NCHW_CHW_NC_C
            ]
        > {
    let summary = "Power VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32]>:$input1,
        RankedTensorOf<[F16, F32, SI32]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32]>:$output
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
                VPU_EltwiseOp,
                VPU_SameAnyDimsOrder
            ]
        > {
    let summary = "Add VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32]>:$input1,
        RankedTensorOf<[F16, F32, SI32]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<IE_PostOp>:$post_op
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32]>:$output
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
                VPU_EltwiseOp,
                SameInOutDimsOrder_NCHW_CHW_NC_C
            ]
        > {
    let summary = "Divide VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32]>:$input1,
        RankedTensorOf<[F16, F32, SI32]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32]>:$output
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
                VPU_EltwiseOp,
                SameInOutDimsOrder_NCHW_CHW_NC_C
            ]
        > {
    let summary = "SquaredDiff VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32]>:$input1,
        RankedTensorOf<[F16, F32, SI32]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32]>:$output
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
                VPU_EltwiseOp,
                SameInOutDimsOrder_NCHW_CHW_NC_C
            ]
        > {
    let summary = "FloorMod VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32]>:$input1,
        RankedTensorOf<[F16, F32, SI32]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32]>:$output
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
                VPU_EltwiseOp,
                SameInOutDimsOrder_NCHW_CHW_NC_C
            ]
        > {
    let summary = "Less VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32]>:$input1,
        RankedTensorOf<[F16, F32, SI32]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32]>:$output
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
                VPU_EltwiseOp,
                SameInOutDimsOrder_NCHW_CHW_NC_C
            ]
        > {
    let summary = "LessEqual VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32]>:$input1,
        RankedTensorOf<[F16, F32, SI32]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32]>:$output
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
                VPU_EltwiseOp,
                SameInOutDimsOrder_NCHW_CHW_NC_C
            ]
        > {
    let summary = "Greater VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32]>:$input1,
        RankedTensorOf<[F16, F32, SI32]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32]>:$output
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
                VPU_EltwiseOp,
                SameInOutDimsOrder_NCHW_CHW_NC_C
            ]
        > {
    let summary = "GreaterEqual VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32]>:$input1,
        RankedTensorOf<[F16, F32, SI32]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32]>:$output
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
                VPU_EltwiseOp,
                SameInOutDimsOrder_NCHW_CHW_NC_C
            ]
        > {
    let summary = "LogicalOr VPU layer";

    let arguments = (ins
        RankedTensorOf<[I8, F16, F32, SI32]>:$input1,
        RankedTensorOf<[I8, F16, F32, SI32]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[I8, F16, F32, SI32]>:$output
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
                VPU_EltwiseOp,
                SameInOutDimsOrder_NCHW_CHW_NC_C
            ]
        > {
    let summary = "LogicalXor VPU layer";

    let arguments = (ins
        RankedTensorOf<[I8, F16, F32, SI32]>:$input1,
        RankedTensorOf<[I8, F16, F32, SI32]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[I8, F16, F32, SI32]>:$output
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
                VPU_EltwiseOp,
                VPU_SameAnyDimsOrder,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "Multiply VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32]>:$input1,
        RankedTensorOf<[F16, F32, SI32]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<IE_PostOp>:$post_op,
        OptionalAttr<VPU_MultiClusterStrategy>:$multiClusterStrategy
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastTypeAttr":$auto_broadcast,
            "vpux::IE::PostOp":$post_op
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;
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
                VPU_EltwiseOp,
                SameInOutDimsOrder_NCHW_CHW_NC_C
            ]
        > {
    let summary = "And VPU layer";

    let arguments = (ins
        RankedTensorOf<[I8, F16, F32, SI32]>:$input1,
        RankedTensorOf<[I8, F16, F32, SI32]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<IE_PostOp>:$post_op
    );

    let results = (outs
        RankedTensorOf<[I8, F16, F32, SI32]>:$output
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

    let extraClassDeclaration = [{
        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter, vpux::NDTypeInterface output, Byte reservedMem);

        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter, vpux::NDTypeInterface output);
    }] # baseExtraClassDeclaration;
}

//
// AvgPool
//

def VPU_AvgPoolOp :
        VPU_LayerOp<
            "AvgPool",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                VPU_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
            ]
        > {
    let summary = "AvgPool VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        I64ArrayAttr:$kernel_size,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        IE_RoundingTypeAttr:$rounding_type,
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
        IE_RoundingTypeAttr:$rounding_type,

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
                DeclareOpInterfaceMethods<IE_ElemTypeInfoOpInterface>,
                VPU_ViewLikeOpInterface
            ]
        > {
    let summary = "Reshape VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$shape,

        UnitAttr:$special_zero,
        OptionalAttr<I64ArrayAttr>:$shape_value
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$output
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
                DeclareOpInterfaceMethods<IE_LayoutInfoOpInterface>,
                VPU_ViewLikeOpInterface
            ]
        > {
    let summary = "Squeeze VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$axes_value
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$output
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
                DeclareOpInterfaceMethods<IE_LayoutInfoOpInterface>,
                VPU_ViewLikeOpInterface
            ]
        > {

    let summary = "Unsqueeze VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$axes_value
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$output
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
            "LRN_IE",
            [
                VPU_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
            ]
        > {
    let summary = "LRN_IE VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        F64Attr:$alpha,
        F64Attr:$beta,
        F64Attr:$bias,
        IntAttr:$size,
        IE_LRN_IERegionAttr:$region
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
        RankedTensorOf<[AnyInteger]>:$axes,

        UnitAttr:$keep_dims
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # baseExtraClassDeclaration;
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

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # baseExtraClassDeclaration;
}

//
// ReduceSum
//

def VPU_ReduceSumOp :
        VPU_LayerOp<
            "ReduceSum",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
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

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # baseExtraClassDeclaration;
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

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # baseExtraClassDeclaration;
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

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # baseExtraClassDeclaration;
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

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # baseExtraClassDeclaration;
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

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # baseExtraClassDeclaration;
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

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # baseExtraClassDeclaration;
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

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # baseExtraClassDeclaration;
}

//
// Minimum
//

def VPU_MinimumOp :
        VPU_LayerOp<
            "Minimum",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                SameInOutDimsOrder_NCHW_CHW_NC_C
            ]
        > {
    let summary = "InferenceEngine Minimum layer";

    let arguments = (ins
        AnyRankedTensor:$input1,
        AnyRankedTensor:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
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
                VPU_EltwiseOp,
                SameInOutDimsOrder_NCHW_CHW_NC_C
            ]
        > {
    let summary = "Maximum VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input1,
        AnyRankedTensor:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// Sparsify
//

def VPU_SparsifyOp :
        VPU_LayerOp<"Sparsify",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Sparsify VPU layer";

    let arguments = (ins
        4DTensorOf<[quant_QuantizedType, F16, BF16]>:$input
    );

    let results = (outs
        VPU_SparseTensor:$output
    );
}

//
// Desparsify
//

def VPU_DesparsifyOp :
        VPU_LayerOp<"Desparsify",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Desparsify VPU layer";

    let arguments = (ins
        VPU_SparseTensor:$input
    );

    let results = (outs
        4DTensorOf<[quant_QuantizedType, F16, BF16]>:$output
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

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # baseExtraClassDeclaration;
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

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # baseExtraClassDeclaration;
}

//
// DynamicQuantize
//

def VPU_DynamicQuantizeOp :
        VPU_LayerOp<"DynamicQuantize",
            [
                VPU_SameInOutDefaultDimsOrder
            ]
        > {
    let summary = "Dynamic-Quantize VPU layer";

    let arguments = (ins
        RankedTensorOf<[F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[UI8]>:$output,
        1DTensorOf<[F32]>:$scale,
        1DTensorOf<[UI8]>:$zero_point
    );
}

//
// QuantizeCast
//

def VPU_QuantizeCastOp :
        VPU_LayerOp<
            "QuantizeCast",
            [
                VPU_ViewLikeOpInterface
            ]
        > {
    let summary = "Quantize Cast VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[SI8, UI8, quant_QuantizedType]>, VPU_SparseTensor]>:$input,

        TypeAttr:$dstElemType
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[SI8, UI8, quant_QuantizedType]>, VPU_SparseTensor]>:$output
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
                VPU_EltwiseOp,
                SameInOutDimsOrder_NCHW_CHW_NC_C
            ]
        > {
    let summary = "Subtract VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32]>:$input1,
        RankedTensorOf<[F16, F32, SI32]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<IE_PostOp>:$post_op
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32]>:$output
    );
}

//
// MemPermute
//

def VPU_MemPermuteOp :
        VPU_LayerOp<
            "MemPermute",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                VPU_AnyDimsOrder
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
    let hasCanonicalizer = 1;
}

//
// PermuteCast
//

def VPU_PermuteCastOp :
        VPU_LayerOp<
            "PermuteCast",
            [
                VPU_ViewLikeOpInterface
            ]
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
                VPU_EltwiseOp,
                SameInOutDimsOrder_NCHW_CHW_NC_C
            ]
        > {
    let summary = "Equal VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32]>:$input1,
        RankedTensorOf<[F16, F32, SI32]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32]>:$output
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
                DeclareOpInterfaceMethods<IE_ElemTypeInfoOpInterface>,
                VPU_ViewLikeOpInterface
            ]
        > {
    let summary = "AffineReshape VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$input,

        I64ArrayOfArraysAttr:$dim_mapping,
        I64ArrayAttr:$shape_value
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$output
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
                VPU_EltwiseOp,
                SameInOutDimsOrder_NCHW_CHW_NC_C
            ]
        > {
    let summary = "NotEqual VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32]>:$input1,
        RankedTensorOf<[F16, F32, SI32]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32]>:$output
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
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$input,

        OptionalAttr<IndexedSymbolAttr>:$out_mem_space
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$output
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
    let summary = "Declares a Storage Element Pointers table";

    let description = [{
        A Storage Element represents a 1x1xN volume that contains sparse data, where N
        represents the number of channels stored. The Storage Element Table is comprised
        of pointers to these Storage Elements, which have the following structure:

        31-29 28                            9 8         0
        -------------------------------------------------
        | xx |           DATA_PTR            | BASE_PTR |
        -------------------------------------------------

        The DATA_PTR represents the offset to a Storage Element in relation to the start of
        the input data. BASE_PTR is used to decide what base address is added to DATA_PTR
        in order to find the location of the Storage Element in memory during inference.

        This operation represents the Storage Element Table in relation to the input data,
        on top of which transformations can be applied. This operation will later get
        converted to a constant, where the pointers are generated based on the information
        contained in this operation.

        The following information is contained:
        - dataShape, dataElemType, dataStrides: information about the input data that
          is associated with this Storage Element Table
        - seSize: the size of a Storage Element
        - seDepth: the number of Storage Elements per depth
        - seAttr: information on how the input data is transformed
        - basePtrs: base pointers associated with each Storage Element pointer
    }];

    let arguments = (ins
        I64ArrayAttr:$dataShape,
        TypeAttr:$dataElemType,
        IntAttr:$seSize,
        IntAttr:$seDepth,
        OptionalAttr<VPU_SEAttr>:$seAttr,
        OptionalAttr<I64ArrayAttr>:$dataStrides,
        OptionalAttr<I32ElementsAttr>:$basePtrs
    );

    let results = (outs
        RankedTensorOf<[I32]>:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
         attr-dict `->` type(results)
    }];

    let builders = [
        OpBuilder<(ins
            CArg<"llvm::ArrayRef<int64_t>">:$dataShape,
            CArg<"mlir::Type">:$dataElemType,
            CArg<"int64_t">:$seSize,
            CArg<"int64_t">:$seDepth,
            CArg<"VPU::SEAttr">:$seAttr
        )>
    ];

    let hasCanonicalizer = 1;
}

//
// NonMaxSuppression
//

def VPU_NonMaxSuppressionOp :
        VPU_LayerOp<
            "NonMaxSuppression"
        > {
    let summary = "NonMaxSuppression VPU layer";

    let arguments = (ins
        3DTensorOf<[F16, F32]>:$in_box_coords,
        3DTensorOf<[F16, F32]>:$in_box_scores,

        IE_BoxEncodingTypeAttr:$box_encoding,
        UnitAttr:$sort_result_descending,

        OptionalAttr<IntAttr>:$max_output_boxes_per_class_value,
        OptionalAttr<F64Attr>:$iou_threshold_value,
        OptionalAttr<F64Attr>:$score_threshold_value,
        OptionalAttr<F64Attr>:$soft_nms_sigma_value
    );

    let results = (outs
        2DTensorOf<[SI32]>:$out_selected_indices,
        2DTensorOf<[F16, F32]>:$out_selected_scores,
        1DTensorOf<[SI32]>:$out_valid_outputs
    );

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # baseExtraClassDeclaration;
}


//
// StubOp
//

def VPU_StubOp :
        VPU_Op<
            "Stub",
            [
                NoSideEffect
            ]
        > {
    let summary = "Substitute operation for stubbing.";

    let arguments = (ins
        Variadic<AnyRankedTensor>:$inputs
    );

    let results = (outs
        Variadic<AnyRankedTensor>:$outputs
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
}

//
// GRUSequence
//

def VPU_GRUSequenceOp :
        VPU_LayerOp<
            "GRUSequence",
            [
                VPU_AnyDimsOrder,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "GRUSequence VPU layer";

    let arguments = (ins
        3DTensorOf<[F16, F32]>:$input_data,
        3DTensorOf<[F16, F32]>:$initial_hidden_state,
        3DTensorOf<[F16, F32]>:$weights,
        3DTensorOf<[F16, F32]>:$recurrence_weights,
        2DTensorOf<[F16, F32]>:$biases,

        IntAttr:$hidden_size,
        IntAttr:$seq_length,
        IE_RNNSequenceDirectionAttr:$direction,
        UnitAttr:$should_linear_before_reset,
        F64Attr:$clip
    );

    let results = (outs
        4DTensorOf<[F16, F32]>:$middle_hidden_state,
        3DTensorOf<[F16, F32]>:$output_hidden_state
    );

    let extraClassDeclaration = [{
        mlir::LogicalResult applyTileStrategyGRUSequence(VPU::TilingBuilderOpInterface origOp, OutputTiling tilesY,
                                                 mlir::PatternRewriter& rewriter, Logger log);
        SmallVector<mlir::Value> reifyTileGRUSequence(VPU::TilingBuilderOpInterface origOp, const TileInfo& outputYTile,
                                              const TileInfo& outputHoTile, mlir::OpBuilder& builder, Logger log,
                                              bool isInitialState, mlir::Value prevHo);
    }] # baseExtraClassDeclaration;
}

//
// GRUSequenceFirstPart
//

def VPU_GRUSequenceFirstPartOp :
        VPU_LayerOp<
            "GRUSequenceFirstPart",
            [
                VPU_AnyDimsOrder
            ]
        > {
    let summary = "GRUSequenceFirstPart VPU layer";

    let arguments = (ins
        3DTensorOf<[F16, F32]>:$input_data,
        3DTensorOf<[F16, F32]>:$weights,

        IntAttr:$hidden_size,
        IntAttr:$seq_length,
        F64Attr:$clip
    );

    let results = (outs
        4DTensorOf<[F16, F32]>:$output
    );
}

//
// GRUSequenceLastPart
//

def VPU_GRUSequenceLastPartOp :
        VPU_LayerOp<
            "GRUSequenceLastPart",
            [
                VPU_AnyDimsOrder
            ]
        > {
    let summary = "GRUSequenceLastPart VPU layer";

    let arguments = (ins
        4DTensorOf<[F16, F32]>:$first_part_output,
        3DTensorOf<[F16, F32]>:$initial_hidden_state,
        3DTensorOf<[F16, F32]>:$recurrence_weights,
        2DTensorOf<[F16, F32]>:$biases,

        IntAttr:$hidden_size,
        IntAttr:$seq_length,
        IE_RNNSequenceDirectionAttr:$direction,
        UnitAttr:$should_linear_before_reset,
        F64Attr:$clip
    );

    let results = (outs
        4DTensorOf<[F16, F32]>:$middle_hidden_state,
        3DTensorOf<[F16, F32]>:$output_hidden_state
    );
}

//
// DeformablePSROIPoolingOp
//

def VPU_DeformablePSROIPoolingOp :
        VPU_LayerOp<
            "DeformablePSROIPooling",
            [
                VPU_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
            ]
        > {
    let summary = "DeformablePSROIPooling VPU layer";

    let arguments = (ins
        4DTensorOf<[AnyFloat]>:$input_score_maps,
        2DTensorOf<[AnyFloat]>:$input_rois,
        Optional<4DTensorOf<[AnyFloat]>>:$input_transformations,

        IntAttr:$output_dim,
        F64Attr:$spatial_scale,
        OptionalAttr<IntAttr>:$group_size,
        OptionalAttr<IntAttr>:$spatial_bins_x,
        OptionalAttr<IntAttr>:$spatial_bins_y,
        OptionalAttr<F64Attr>:$trans_std,
        OptionalAttr<IntAttr>:$part_size,
        OptionalAttr<IE_DeformablePSROIPoolingModeAttr>:$mode
    );

    let results = (outs
        4DTensorOf<[AnyFloat]>:$output
    );
}

//
// PermuteQuantize
//

def VPU_PermuteQuantizeOp :
        VPU_LayerOp<
            "PermuteQuantize",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "PermuteQuantize VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        AffineMapAttr:$dst_order,
        AffineMapAttr:$mem_perm,
        TypeAttr:$dstElemType,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end
    );

    let results = (outs
        RankedTensorOf<[quant_QuantizedType]>:$output
    );
}

//
// DFTOp
//

def VPU_DFTOp :
        VPU_LayerOp<
            "DFT",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                VPU_SameInOutDefaultDimsOrder
            ]
        > {
    let summary = "InferenceEngine DFT layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        I64ArrayAttr:$axes_attr,
        I64ArrayAttr:$signal_size_attr
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # baseExtraClassDeclaration;
}

//
// RDFTOp
//

def VPU_RDFTOp :
        VPU_LayerOp<
            "RDFT",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "InferenceEngine RDFT layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        I64ArrayAttr:$axes_attr,
        I64ArrayAttr:$signal_size_attr

    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # baseExtraClassDeclaration;
}

//
// IDFTOp
//

def VPU_IDFTOp :
        VPU_LayerOp<
            "IDFT",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                VPU_SameInOutDefaultDimsOrder
            ]
        > {
    let summary = "InferenceEngine IDFT layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        I64ArrayAttr:$axes_attr,
        I64ArrayAttr:$signal_size_attr

    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # baseExtraClassDeclaration;
}

//
// IRDFTOp
//

def VPU_IRDFTOp :
        VPU_LayerOp<
            "IRDFT",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "InferenceEngine IRDFT layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        I64ArrayAttr:$axes_attr,
        I64ArrayAttr:$signal_size_attr

    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # baseExtraClassDeclaration;
}

//
// ShapeCastOp
//

def VPU_ShapeCastOp :
        VPU_LayerOp<
            "ShapeCast"
        > {
    let summary = "ShapeCast VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$source,
        I64ArrayAttr:$shape
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$result
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $source `:` type($source) `)`
        `->` type(results)
    }];
}

//
// LayoutCastOp
//

def VPU_LayoutCastOp :
        VPU_LayerOp<
            "LayoutCast",
            [
                VPU_ViewLikeOpInterface
            ]
        > {
    let summary = "This layer overrides layout of a given tensor.";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$input,
        AffineMapAttr:$dst_order
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$output
    );

    let hasVerifier = 1;
}

//
// WorkloadCastOp
//

def VPU_WorkloadCastOp :
        VPU_Op<
            "WorkloadCast",
            [
                VPU_ViewLikeOpInterface
            ]
        > {
    let summary = "Operation that casts one DistributedTensor type to another.";

    let description = [{
        This operation is required in order to support cluster tiling for VPU.NCE.PermuteQuantize.
        PermuteQuantize operates on workloads split over width, while input DMA must tile over height.
        For example consider the following chain of operations:
        ```
            Input 1x3x16x32 -> Reshape 1x32x3x16 -> VPU.NCE.PermuteQuantize -> Reshape 1x3x16x32
        ```
        VPU.NCE.PermuteQuantize operates with 1x32x3x16 workload while original input has 1x3x16x32 shape.
        Original input must be split over height, PermuteQuantize workload must be split over width:
        ```
            Tile 1: Copy 1x3x8x32 -> Reshape 1x32x3x8 -> VPU.NCE.PermuteQuantize -> Reshape 1x3x8x32
            Tile 2: Copy 1x3x8x32 -> Reshape 1x32x3x8 -> VPU.NCE.PermuteQuantize -> Reshape 1x3x8x32
        ```
        However, Reshape prohibits such tiling because split axis differs in input and output.
        VPU.WorkloadCast solves this problem because it doesn't have strict checks:
        ```
            Copy 1x3x16x32, SOH -> WorkloadCast 1x32x3x8, SOW -> PermuteQuantize -> WorkloadCast 1x32x3x8, SOH
            SOH = [1, 1, 2, 1]
            SOW = [1, 1, 1, 2]
        ```
    }];

    let arguments = (ins
        AnyTypeOf<[VPU_DistributedTensor, VPU_SparseTensor]>:$input
    );

    let results = (outs
        AnyTypeOf<[VPU_DistributedTensor, VPU_SparseTensor]>:$output
    );

    let assemblyFormat = [{
        `(` $input `:` qualified(type($input)) `)`
        attr-dict
        `->` qualified(type($output))
    }];
}
#endif
