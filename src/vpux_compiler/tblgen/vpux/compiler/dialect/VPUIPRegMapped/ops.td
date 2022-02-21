//
// Copyright (C) 2022 Intel Corporation.
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

#ifndef VPUX_COMPILER_DIALECT_VPUIPREGMAPPED_OPS
#define VPUX_COMPILER_DIALECT_VPUIPREGMAPPED_OPS

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/core/ops_interfaces.td"
include "vpux/compiler/dialect/const/attributes.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/IERT/ops_interfaces.td"
include "vpux/compiler/dialect/VPUIPRegMapped/attributes.td"
include "vpux/compiler/dialect/VPUIPRegMapped/dialect.td"
include "vpux/compiler/dialect/VPUIPRegMapped/ops_interfaces.td"
include "vpux/compiler/dialect/VPURT/types.td"
include "vpux/compiler/dialect/ELF/ops_interfaces.td"

include "mlir/Dialect/Quant/QuantOpsBase.td"
include "mlir/Interfaces/CopyOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

//
// Base classes
//

class VPUIPRegMapped_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<
            VPUIPRegMapped_Dialect,
            mnemonic,
            traits
        >;

// TaskOp is used as a parent Op for NNDMAOp and NCEClusterTaskOp
class VPUIPRegMapped_TaskOp<int numResult, string mnemonic, list<OpTrait> traits = []> :
        VPUIPRegMapped_Op<
            mnemonic,
            !listconcat(
                traits,
                [
                    AttrSizedOperandSegments,
                    InferTypeOpInterface,
                    MemoryEffectsOpInterface,
                    DeclareOpInterfaceMethods<IERT_LayerOpInterface>,
                    IERT_AsyncLayerOpInterface,
                    DeclareOpInterfaceMethods<VPUIPRegMapped_TaskOpInterface>,
                    DeclareOpInterfaceMethods<ELF_BinaryOpInterface>
                ]
            )
        > {
    code commonExtraClassDeclaration = IERT_InferLayerReturnTypes<numResult>.impl;

    let extraClassDeclaration = commonExtraClassDeclaration;
}

//
// GraphOp
//

def VPUIPRegMapped_GraphOp :
        VPUIPRegMapped_Op<
            "Graph",
            [
                HasParent<"mlir::ModuleOp">
            ]
        > {
    let summary = "The root object for the VPUIPRegMapped Execution Graph";

    let arguments = (ins
        VPUIPRegMapped_ExecutionFlag:$options,
        VPUIPRegMapped_Version:$version
    );

    let extraClassDeclaration = [{
        static vpux::VPUIPRegMapped::GraphOp getFromModule(mlir::ModuleOp module);
    }];

    let assemblyFormat = [{
        attr-dict
        `options` `:` $options
        `version` `:` $version
    }];
}

//
// DeclareVirtualBarrierOp
//

def VPUIPRegMapped_DeclareVirtualBarrierOp :
        VPUIPRegMapped_Op<
            "DeclareVirtualBarrier",
            [
                NoSideEffect,
                DeclarationOp
            ]
        > {
    let summary = "VPUIPRegMapped virtual Barrier declaration";

    let results = (outs
        VPURT_Barrier:$barrier
    );

    let assemblyFormat = [{
        attr-dict `->` type(results)
    }];
}

//
// ConfigureBarrierOp
//

def VPUIPRegMapped_ConfigureBarrierOp :
        VPUIPRegMapped_Op<"ConfigureBarrier",
            [
                MemoryEffectsOpInterface,
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<VPUIPRegMapped_TaskOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface>,
                DeclarationOp,
                DotInterface
            ]
        > {
    let summary = "A task to configure the setup for a barrier";

    let arguments = (ins
        I8Attr:$id,
        I32Attr:$next_same_id,
        OptionalAttr<UI8Attr>:$producer_count,
        OptionalAttr<UI8Attr>:$consumer_count,

        Variadic<VPURT_Barrier>:$waitBarriers,
        Variadic<VPURT_Barrier>:$updateBarriers
    );

    let results = (outs
        VPURT_Barrier:$barrier
    );

    let builders = [
        OpBuilder<(ins "int64_t":$id)>,
        OpBuilder<(ins "int64_t":$id, "int16_t":$next_same_id)>
    ];

    let extraClassDeclaration = [{
        static vpux::VPUIPRegMapped::TaskType getTaskType() {
            return vpux::VPUIPRegMapped::TaskType::Controller;
        }

        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::AQUA;
        }

        bool printAttributes(llvm::raw_ostream& os) { // 2021_10_14: after source repo rebase
            os << "id: " << idAttr().getInt();
            return true;
        }
    }];

    let assemblyFormat = [{
        attr-dict
        `<` $id `,` $next_same_id `>`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];
}

//
// EmptyOp
//

def VPUIPRegMapped_EmptyOp :
        VPUIPRegMapped_Op<"Empty",
            [
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<VPUIPRegMapped_TaskOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface>
            ]
        > {
    let summary = "Empty op";

    let arguments = (ins
        Variadic<VPURT_Barrier>:$waitBarriers,
        Variadic<VPURT_Barrier>:$updateBarriers
    );

    let extraClassDeclaration = [{
        static vpux::VPUIPRegMapped::TaskType getTaskType() {
            return vpux::VPUIPRegMapped::TaskType::Controller;
        }
    }];

    let assemblyFormat = [{
        attr-dict
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
    }];
}

//
// MappedInferenceOp
//

def VPUIPRegMapped_MappedInferenceOp :
        VPUIPRegMapped_Op<"MappedInference",
            [
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface>
            ]
        > {
    let summary = "Task representing the single mappedInference op";

    let arguments = (ins
        Optional<AnyMemRef>:$dmaList,
        Optional<AnyMemRef>:$invariantList,
        Optional<AnyMemRef>:$variantList,
        Optional<AnyMemRef>:$actInvocations,
        Optional<VPURT_Barrier>:$barrierList,
        UI32Attr:$dmaCount,
        UI32Attr:$invariantCount,
        UI32Attr:$variantCount,
        UI32Attr:$actInvocationsCount,
        UI32Attr:$barrierCount
    );

    let results = (outs
        Index:$output);

    let assemblyFormat = [{
        attr-dict
        (`dmas` `(` $dmaList^ `:` type($dmaList) `)`)?
        (`invariants` `(` $invariantList^ `:` type($invariantList) `)`)?
        (`variants` `(` $variantList^ `:` type($variantList) `)`)?
        (`actInvocations` `(` $actInvocations^ `:` type($actInvocations) `)`)?
        (`barriers` `(` $barrierList^ `:` type($barrierList) `)` )?
        `dmaCount` `(` $dmaCount `)`
        `invariantCount` `(` $invariantCount `)`
        `variantCount` `(` $variantCount `)`
        `actInvocationsCount` `(` $actInvocationsCount `)`
        `barrierCount` `(` $barrierCount `)`
    }];
}

//
// NNDMAOp
//

def VPUIPRegMapped_NNDMAOp :
        VPUIPRegMapped_TaskOp<1, "NNDMA",
            [
                ViewLikeOpInterface,
                IERT_SameShape,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder,
                DotInterface
            ]
        > {
    let summary = "NN DMA task";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output_buff,

        Variadic<VPURT_Barrier>:$waitBarriers,
        Variadic<VPURT_Barrier>:$updateBarriers,
        UnitAttr:$compression,
        DefaultValuedAttr<IntAttr, "0">:$port,
        UI64Attr:$start_after
    );

    let results = (outs
        AnyMemRef:$output
    );

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$src, "mlir::Value":$dst)
        >
    ];

    let extraClassDeclaration = [{
        static vpux::VPUIPRegMapped::TaskType getTaskType() {
            return vpux::VPUIPRegMapped::TaskType::NNDMA;
        }

        vpux::DotNodeColor getNodeColor() {
            const auto memKind = getType().cast<vpux::NDTypeInterface>().getMemoryKind();
            return memKind == vpux::VPU::MemoryKind::DDR ? vpux::DotNodeColor::RED : vpux::DotNodeColor::GREEN;
        }

        void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects) {
            vpux::VPUIPRegMapped::getTaskEffects(this->getOperation(), effects);
        }
    }] # IERT_InferLayerReturnTypes<1>.impl;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `start_after` `(` $start_after `)`
        `->` type(results)
    }];
}


//
// WeightsTableOp
//

def VPUIPRegMapped_WeightsTableOp :
        VPUIPRegMapped_Op<"WeightsTableOp",
            [
                AttrSizedOperandSegments,
            ]
        > {
    let summary = "Intermediate task for creating weights table based on the addresses of CMX buffers";

    let arguments = (ins
        Optional<MemRefOf<[F16, quant_QuantizedType]>>:$op_input,
        Optional<MemRefOf<[F16, quant_QuantizedType]>>:$op_output,
        Optional<MemRefOf<[F16, quant_QuantizedType]>>:$weights,
        Optional<MemRefOf<[F16, F32]>>:$bias,
        Optional<MemRefOf<[UI8]>>:$activation_window
    );

    let results = (outs
        MemRefOf<[SI32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        (`op_input` `(` $op_input^  `:` type($op_input) `)`)?
        (`op_output` `(` $op_output^  `:` type($op_output) `)`)?
        (`weights` `(` $weights^  `:` type($weights) `)`)?
        (`bias` `(` $bias^  `:` type($bias) `)`)?
        (`activation_window` `(` $activation_window^  `:` type($activation_window) `)`)?
        `->` type(results)
    }];
}

//
// DPUTaskOp
//

def VPUIPRegMapped_DPUTaskOp :
        VPUIPRegMapped_Op<
            "DPUTask",
            [
                HasParent<"vpux::VPUIPRegMapped::NCEClusterTaskOp">
            ]
        > {
    let summary = "This object represents workload for a single DPU tile";

    let arguments = (ins
        I64ArrayAttr:$start,
        I64ArrayAttr:$end,
        VPUIPRegMapped_PaddingAttr:$pad,
        VPUIPRegMapped_MPEMode:$mpe_mode
    );

    let assemblyFormat = [{
        attr-dict
    }];

    let verifier = [{
        return vpux::VPUIPRegMapped::verifyOp(*this);
    }];
}

//
// PPETaskOp
//

def VPUIPRegMapped_PPETaskOp :
        VPUIPRegMapped_Op<
            "PPETask",
            [
                HasParent<"vpux::VPUIPRegMapped::NCEClusterTaskOp">
            ]
        > {
    let summary = "PPE Type for NCE Task";

    let arguments = (ins
        VPUIPRegMapped_PPELayerType:$ppe_layer_type,

        OptionalAttr<IntAttr>:$clamp_low,
        OptionalAttr<IntAttr>:$clamp_high,
        OptionalAttr<IntAttr>:$lrelu_mult,
        OptionalAttr<IntAttr>:$lrelu_shift
    );

    let builders = [
        OpBuilder<(ins
            CArg<"vpux::VPUIPRegMapped::PPELayerType", "vpux::VPUIPRegMapped::PPELayerType::NOOP">:$ppe_layer_type,
            CArg<"int32_t", "std::numeric_limits<int32_t>::min()">:$clamp_low,
            CArg<"int32_t", "std::numeric_limits<int32_t>::max()">:$clamp_high,
            CArg<"int32_t", "1">:$lrelu_mult,
            CArg<"uint32_t", "0">:$lrelu_shift
        )>
    ];

    let assemblyFormat = [{
        $ppe_layer_type attr-dict
    }];
}

//
// NCEClusterTaskOp
//

def VPUIPRegMapped_NCEClusterTaskOp :
        VPUIPRegMapped_TaskOp<1, "NCEClusterTask",
            [
                ViewLikeOpInterface,
                NoRegionArguments
            ]
            # GraphRegionNoTerminator.traits
        > {

    let summary = "NCE Cluster Task Operation";

    let description = [{
        This operation defines NCE cluster task which describes single cluster of 5 DPUs. It is
        comprised of two argument categories:

        * Variants - describes the attributes for an individual DPU within the cluster.
        * Invariants - describes the collective attributes of the cluster.

        The variants argument takes on a region argument and up to 5 DPUTaskOps. The invariants
        take on a variety of argument types.

        The NCEClusterTaskOp also supports fixed PPE functions as well as generic PPE instruction
        lists. The generic PPE instruction list argument needs to be described as a region of PPE
        supported ops. Single fixed PPE functions and generic PPE instruciton list usage is
        mutually exclusive.
    }];

    let arguments = (ins
        MemRefOf<[F16, quant_QuantizedType]>:$input,
        Optional<MemRefOf<[F16, quant_QuantizedType]>>:$weights,
        Optional<MemRefOf<[SI32]>>:$weight_table,
        Optional<MemRefOf<[UI8]>>:$activation_window,

        AnyMemRef:$parent_input,
        AnyMemRef:$parent_output,

        MemRefOf<[F16, quant_QuantizedType]>:$output_buff,

        Variadic<VPURT_Barrier>:$waitBarriers,
        Variadic<VPURT_Barrier>:$updateBarriers,

        VPUIPRegMapped_NCETaskType:$task_type,
        OptionalAttr<I64ArrayAttr>:$kernel_size,
        OptionalAttr<I64ArrayAttr>:$kernel_strides,
        OptionalAttr<I64ArrayAttr>:$kernel_padding,
        OptionalAttr<IntAttr>:$activation_window_channel_length,
        OptionalAttr<UnitAttr>:$is_continued
    );

    let results = (outs
        MemRefOf<[F16, quant_QuantizedType]>:$output
    );

    let regions = (region
        SizedRegion<1>:$variants,
        SizedRegion<1>:$ppe
    );

    let builders = [
        OpBuilder<(ins
            "mlir::Value":$input, "mlir::Value":$weights, "mlir::Value":$weight_table, "mlir::Value":$activation_window,
            "mlir::Value":$parent_input, "mlir::Value":$parent_output,
            "mlir::Value":$output_buff,
            "vpux::VPUIPRegMapped::NCETaskType":$task_type,
            "mlir::ArrayAttr":$kernel_size, "mlir::ArrayAttr":$kernel_strides, "mlir::ArrayAttr":$kernel_padding,
            "mlir::IntegerAttr":$activation_window_channel_length,
            "mlir::UnitAttr":$is_continued
        )>
    ];

    let assemblyFormat = [{
        attr-dict
        `input` `(` $input  `:` type($input) `)`
        (`weights` `(` $weights^  `:` type($weights) `)`)?
        (`weight_table` `(` $weight_table^  `:` type($weight_table) `)`)?
        (`activation_window` `(` $activation_window^  `:` type($activation_window) `)`)?
        `parent_input` `(` $parent_input `:` type($parent_input) `)`
        `parent_output` `(` $parent_output `:` type($parent_output) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
        `variants` `:` $variants
        `PPE` `:` $ppe
    }];

    let verifier = [{
        return vpux::VPUIPRegMapped::verifyOp(*this);
    }];

    let extraClassDeclaration = [{
        vpux::VPUIPRegMapped::DPUTaskOp addDPUTask(mlir::OpBuilder& builder,
                                          mlir::ArrayAttr start, mlir::ArrayAttr end,
                                          VPUIPRegMapped::PaddingAttr pad, VPUIPRegMapped::MPEMode mpeMode);

        template <typename... Args>
        vpux::VPUIPRegMapped::PPETaskOp addPPETask(mlir::OpBuilder& builder, Args&&... args);

        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);

        static vpux::VPUIPRegMapped::TaskType getTaskType() {
            return vpux::VPUIPRegMapped::TaskType::NCE2;
        }

        void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects) {
            vpux::VPUIPRegMapped::getTaskEffects(this->getOperation(), effects);
        }
    }] # IERT_InferLayerReturnTypes<1>.impl;
}

#endif

