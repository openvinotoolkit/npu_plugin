//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_EMU_OPS
#define VPUX_COMPILER_DIALECT_EMU_OPS

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/core/ops_interfaces.td"
include "vpux/compiler/dialect/const/attributes.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/EMU/dialect.td"
include "vpux/compiler/dialect/EMU/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPUIP/attributes.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Dialect/Quant/QuantOpsBase.td"

//
// Base classes
//

class EMU_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<
            EMU_Dialect,
            mnemonic,
            traits
        >;

class EMU_TaskOp<string mnemonic, list<OpTrait> traits = []> :
        EMU_Op<
            mnemonic,
            [
                NoSideEffect,
                DeclareOpInterfaceMethods<EMU_SerializeInterface>
            ] # traits
        > {

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
}

//
// PPETaskOp
//

def EMU_PPETaskOp :
        EMU_Op<
            "PPETask",
            [
                HasParent<"vpux::EMU::NCEClusterTaskOp">
            ]
        > {
    let summary = "PPE Type for NCE Task";

    let arguments = (ins
        VPU_PPEMode:$ppe_layer_type,

        OptionalAttr<IntAttr>:$clamp_low,
        OptionalAttr<IntAttr>:$clamp_high,
        OptionalAttr<IntAttr>:$lrelu_mult,
        OptionalAttr<IntAttr>:$lrelu_shift,
        OptionalAttr<I64ArrayAttr>:$quant_mult,
        OptionalAttr<I64ArrayAttr>:$quant_shift,
        OptionalAttr<IntAttr>:$quant_post_shift
    );

    let builders = [
        OpBuilder<(ins
            CArg<"vpux::VPU::PPEMode", "vpux::VPU::PPEMode::NOOP">:$ppe_layer_type,
            CArg<"int64_t", "std::numeric_limits<int32_t>::min()">:$clamp_low,
            CArg<"int64_t", "std::numeric_limits<int32_t>::max()">:$clamp_high,
            CArg<"int64_t", "1">:$lrelu_mult,
            CArg<"int64_t", "0">:$lrelu_shift
        )>,
        OpBuilder<(ins
            "vpux::VPU::PPEMode":$ppe_layer_type,
            "int64_t":$clamp_low,
            "int64_t":$clamp_high,
            "int64_t":$lrelu_mult,
            "int64_t":$lrelu_shift,
            "int64_t":$quant_mult,
            "int64_t":$quant_shift,
            "int64_t":$quant_post_shift
        )>
    ];

    let assemblyFormat = [{
        $ppe_layer_type attr-dict
    }];
}

//
// NCEClusterTaskOp
//

def EMU_NCEClusterTaskOp :
        EMU_TaskOp<"NCEClusterTask",
            [
                NoRegionArguments,
                OpAsmOpInterface,
                AttrSizedOperandSegments
            ]
            # GraphRegionNoTerminator.traits
        > {

    let summary = "NCE Cluster Task Operation";

    let description = [{
        This operation defines an NCE task.

        It supports fixed PPE functions as well as generic PPE instruction
        lists. The generic PPE instruction list argument needs to be described as a region of PPE
        supported ops. Single fixed PPE functions and generic PPE instruciton list usage is
        mutually exclusive.
    }];

    let arguments = (ins
        RankedTensorOf<[F16, quant_QuantizedType]>:$input,
        Optional<RankedTensorOf<[F16, quant_QuantizedType]>>:$weights,
        Optional<RankedTensorOf<[SI32]>>:$weight_table,

        VPUIP_NCETaskType:$task_type,
        OptionalAttr<I64ArrayAttr>:$kernel_size,
        OptionalAttr<I64ArrayAttr>:$kernel_strides,
        OptionalAttr<I64ArrayAttr>:$kernel_padding
    );

    let results = (outs
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$output
    );

    let regions = (region
        SizedRegion<1>:$ppe
    );

    let assemblyFormat = [{
        attr-dict
        `input` `(` $input  `:` type($input) `)`
        (`weights` `(` $weights^  `:` type($weights) `)`)?
        (`weight_table` `(` $weight_table^  `:` type($weight_table) `)`)?
        `->` type(results)
        `PPE` `:` $ppe
    }];

    let verifier = [{
        return vpux::EMU::verifyOp(*this);
    }];

    let extraClassDeclaration = [{
        template <typename... Args>
        vpux::EMU::PPETaskOp addPPETask(mlir::OpBuilder& builder, Args&&... args);

        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::NCE;
        }

        static mlir::StringRef getDefaultDialect() {
            return "EMU";
        }
    }];
}

#endif
