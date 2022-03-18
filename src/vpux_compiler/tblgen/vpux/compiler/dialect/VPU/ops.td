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

#ifndef VPUX_COMPILER_DIALECT_VPU_OPS
#define VPUX_COMPILER_DIALECT_VPU_OPS

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/core/ops_interfaces.td"
include "vpux/compiler/dialect/const/attributes.td"
include "vpux/compiler/dialect/const/attributes.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPU/dialect.td"
include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/types.td"

include "mlir/Dialect/Quant/QuantOpsBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/RegionKindInterface.td"

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
                InferTypeOpInterface,
                DeclareOpInterfaceMethods<InferShapedTypeOpInterface, ["inferReturnTypeComponents"]>,
                DeclareOpInterfaceMethods<IE_LayerOpInterface>
            ] # traits
        > {
    list<string> elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    bit checkInferredDimsOrder = 0;
    bit checkInferredMemSpace = 0;
    bit checkInferredSparsity = 0;

    code baseExtraClassDeclaration = [{
        static bool isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
            return vpux::IE::isCompatibleTensorTypes(lhs, rhs,
                }] # !interleave(elemComparisonModes, "|") # [{,
                static_cast<bool>(}] # checkInferredDimsOrder # [{),
                static_cast<bool>(}] # checkInferredMemSpace # [{),
                static_cast<bool>(}] # checkInferredSparsity # [{)
            );
        }
    }];
    let extraClassDeclaration = baseExtraClassDeclaration;
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
                DeclareOpInterfaceMethods<IE_TilingBuilderOpInterface>,
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
        Optional<4DTensorOf<[SI32]>>:$instructionListTable,
        Optional<4DTensorOf<[UI8]>>:$activationWindow,

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

        static bool isSupported(vpux::IE::ConvolutionOp origOp, vpux::VPU::NCEInvariant::LogCb logCb);

        static int64_t getInputChannelAlignmentImpl(vpux::NDTypeInterface inputType);

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
                DeclareOpInterfaceMethods<IE_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_NCEOpInterface>,
                IE_AlignedChannelsOpInterface
            ]
        > {
    let summary = "NCE version of Depthwise Convolution layer";

    let arguments = (ins
        4DTensorOf<[F16, BF16, quant_QuantizedType]>:$input,
        4DTensorOf<[F16, BF16, quant_QuantizedType]>:$filter,
        4DTensorOf<[SI32]>:$weightsTable,
        Optional<4DTensorOf<[SI32]>>:$instructionListTable,
        4DTensorOf<[UI8]>:$activationWindow,

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

        static bool isSupported(vpux::IE::GroupConvolutionOp origOp, vpux::VPU::NCEInvariant::LogCb logCb);

        bool checkChannelRestrictions(int64_t channels);

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
                DeclareOpInterfaceMethods<IE_TilingBuilderOpInterface>,
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

        static bool isSupported(vpux::IE::MaxPoolOp origOp, vpux::VPU::NCEInvariant::LogCb logCb);

        bool checkChannelRestrictions(int64_t channels);
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
                IE_EltwiseOp,
                IE_TilingBuilderOpInterface,
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
                                vpux::VPU::NCEInvariant::LogCb logCb);
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

    let printer = [{ return vpux::VPU::print(p, *this); }];
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

#endif
