//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_VPUIPREGMAPPED_OPS
#define VPUX_COMPILER_DIALECT_VPUIPREGMAPPED_OPS

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/core/ops_interfaces.td"
include "vpux/compiler/dialect/const/attributes.td"
include "vpux/compiler/dialect/VPUIP/ops_interfaces.td"
include "vpux/compiler/dialect/VPUIPRegMapped/attributes.td"
include "vpux/compiler/dialect/VPUIPRegMapped/dialect.td"
include "vpux/compiler/dialect/VPUIPRegMapped/ops_interfaces.td"
include "vpux/compiler/dialect/VPUIPRegMapped/types.td"
include "vpux/compiler/dialect/VPURT/types.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPUIP/types.td"
include "vpux/compiler/dialect/VPUIP/attributes.td"
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

class VPUIPRegMapped_Op<string mnemonic, list<Trait> traits = []> :
        Op<
            VPUIPRegMapped_Dialect,
            mnemonic,
            traits
        >;

class VPUIPRegMapped_ExecutableTaskOp<string mnemonic, list<Trait> traits = []> :
        VPUIPRegMapped_Op<
            mnemonic,
            !listconcat(
                traits,
                [
                    DeclareOpInterfaceMethods<VPUIPRegMapped_ExecutableTaskOpInterface>,
                    DeclareOpInterfaceMethods<ELF_BinaryOpInterface>,
                    DeclareOpInterfaceMethods<ELF_GetOffsetOfOpInterface>,
                    VPUIPRegMapped_SingleOutputAsIndexOp
                ]
            )
        > {
}

//
// ConfigureBarrierOp
//

def VPUIPRegMapped_ConfigureBarrierOp :
        VPUIPRegMapped_Op<"ConfigureBarrier",
            [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface>,
                VPUIPRegMapped_SingleOutputAsIndexOp,
                DeclarationOp
            ]
        > {
    let summary = "A task to configure the setup for a barrier";

    let arguments = (ins
        I8Attr:$id,
        SI32Attr:$next_same_id,
        OptionalAttr<UI8Attr>:$producer_count,
        OptionalAttr<UI8Attr>:$consumer_count
    );

    let results = (outs
        VPUIPRegMapped_IndexType:$barrier
    );

    let assemblyFormat = [{
        attr-dict
        `<` $id `,` $next_same_id `>`
        `->` type(results)
    }];
}

//
// MappedInferenceOp
//
def VPUIPRegMapped_MappedInferenceOp :
        VPUIPRegMapped_Op<"MappedInference",
            [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface>,
                DeclareOpInterfaceMethods<ELF_GetOffsetOfOpInterface>,
                VPUIPRegMapped_SingleOutputAsIndexOp,
                AttrSizedOperandSegments
            ]
        > {
    let summary = "Task representing the MappedInference structure";

    let arguments = (ins
        Variadic<VPUIPRegMapped_IndexType>:$dmaTasks,
        Optional<VPUIPRegMapped_IndexType>:$invariantTasks,
        Optional<VPUIPRegMapped_IndexType>:$variantTasks,
        Optional<VPUIPRegMapped_IndexType>:$actKernelRanges,
        Optional<VPUIPRegMapped_IndexType>:$actKernelInvocations,
        Optional<VPUIPRegMapped_IndexType>:$barrierTasks,
        Optional<VPUIPRegMapped_IndexType>:$actShaveRt,
        Variadic<AnyMemRef>:$actShaveStacks,
        I64ArrayAttr:$dmaCount,
        UI32Attr:$invariantCount,
        UI32Attr:$variantCount,
        UI32Attr:$actKernelRangesCount,
        UI32Attr:$actKernelInvocationsCount,
        UI32Attr:$barrierCount
    );

    let results = (outs
        VPUIPRegMapped_IndexType:$index
    );

    let assemblyFormat = [{
    attr-dict
        (`dmas` `(` $dmaTasks^ `:` type($dmaTasks) `)`)?
        (`invariants` `(` $invariantTasks^ `:` type($invariantTasks) `)`)?
        (`variants` `(` $variantTasks^ `:` type($variantTasks) `)`)?
        (`actKernelRanges` `(` $actKernelRanges^ `:` type($actKernelRanges) `)`)?
        (`actKernelInvocations` `(` $actKernelInvocations^ `:` type($actKernelInvocations) `)`)?
        (`barriers` `(` $barrierTasks^ `:` type($barrierTasks) `)` )?
        (`actShaveRt` `(` $actShaveRt^ `:` type($actShaveRt) `)` )?
        (`actShaveStacks` `(` $actShaveStacks^ `:` type($actShaveStacks) `)`)?
        `dmaCount` `(` $dmaCount `)`
        `invariantCount` `(` $invariantCount `)`
        `variantCount` `(` $variantCount `)`
        `actKernelRangesCount` `(` $actKernelRangesCount `)`
        `actKernelInvocationsCount` `(` $actKernelInvocationsCount `)`
        `barrierCount` `(` $barrierCount `)`
        `->` type(results)
    }];
}


//
// NNDMAOp
//

def VPUIPRegMapped_NNDMAOp :
        VPUIPRegMapped_ExecutableTaskOp<"NNDMA",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "NN DMA task";

    let arguments = (ins
        AnyMemRef:$input,
        Variadic<AnyMemRef>:$output_buffs,
        Optional<VPUIPRegMapped_IndexType>:$previousDMAIdx,
        Variadic<VPUIPRegMapped_IndexType>:$waitBarriers,
        Variadic<VPUIPRegMapped_IndexType>:$updateBarriers,
        UnitAttr:$compression,
        UI64Attr:$start_after,
        UI64Attr:$clean_after,
        UnitAttr:$is_out_of_order,
        UnitAttr:$is_critical,
        DefaultValuedAttr<IntAttr, "0">:$port,
        OptionalAttr<VPUIP_DmaDescriptorAttr>:$dma_descriptor
    );

    let results = (outs
        VPUIPRegMapped_IndexType:$index
    );

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$src, "mlir::Value":$dst)
        >
    ];

    let extraClassDeclaration = [{
        static vpux::VPUIPRegMapped::TaskType getTaskType() {
            return vpux::VPUIPRegMapped::TaskType::DMA;
        }
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        (`outputs` `(` $output_buffs^ `:` type($output_buffs) `)`)?
        ( `previousDMA` `(` $previousDMAIdx^ `:` type($previousDMAIdx) `)`)?
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `start_after` `(` $start_after `)`
        `clean_after` `(` $clean_after `)`
        `->` type(results)
    }];
}

//
// Activation Kernel Ops
//

//
// ActShaveRtOp
//

def VPUIPRegMapped_ActShaveRtOp :
        VPUIPRegMapped_Op<"ActShaveRt",
            [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface>,
                VPUIPRegMapped_SingleOutputAsIndexOp,
                DeclarationOp
            ]
        > {
    let summary = "Declaration of Act Shave Management Kernel";

    let arguments = (ins
        StrAttr:$kernel_path
    );

    let results = (outs
        VPUIPRegMapped_IndexType:$index
    );

    let assemblyFormat = [{
        attr-dict `kernel` `(` $kernel_path `)` `->` type(results)
    }];

    let extraClassDeclaration = [{
        uint32_t getKernelEntry();
        uint32_t getVersion();
    }];
}

//
// DeclareKernelTextOp
//

def VPUIPRegMapped_DeclareKernelTextOp :
        VPUIPRegMapped_Op<"DeclareKernelText",
            [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface>,
                VPUIPRegMapped_SingleOutputAsIndexOp,
                DeclarationOp
            ]
        > {
    let summary = "Declaration of Software Kernel .text ";

    let arguments = (ins
        StrAttr:$kernel_path
    );

    let results = (outs
        VPUIPRegMapped_IndexType:$index
    );

    let assemblyFormat = [{
        attr-dict `kernel_path` `(` $kernel_path `)` `->` type(results)
    }];
}


//
// DeclareKernelArgsOp
//

def VPUIPRegMapped_DeclareKernelArgsOp :
        VPUIPRegMapped_Op<"DeclareKernelArgs",
            [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface>,
                VPUIPRegMapped_SingleOutputAsIndexOp,
                DeclarationOp
            ]
        > {
    let summary = "Declaration of Software Kernel .args";

    let arguments = (ins
        StrAttr:$kernel_path
    );

    let results = (outs
        VPUIPRegMapped_IndexType:$index
    );

    let assemblyFormat = [{
        attr-dict `kernel_path` `(` $kernel_path `)` `->` type(results)
    }];
}

//
// DeclareKernelEntryOp
//

def VPUIPRegMapped_DeclareKernelEntryOp :
        VPUIPRegMapped_Op<"DeclareKernelEntry",
            [
                VPUIPRegMapped_SingleOutputAsIndexOp,
                DeclarationOp
            ]
        > {
    let summary = "Declaration of Kernel Entry";

    let arguments = (ins
        StrAttr:$kernel_path
    );

    let results = (outs
        VPUIPRegMapped_IndexType:$index
    );

    let assemblyFormat = [{
        attr-dict `kernel_path` `(` $kernel_path `)` `->` type(results)
    }];

    let extraClassDeclaration = [{
        uint32_t getKernelEntry();
    }];

}

//
// KernelParamsOp
//

def VPUIPRegMapped_KernelParamsOp :
        VPUIPRegMapped_Op<"KernelParams",
            [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface>,
                DeclareOpInterfaceMethods<ELF_GetOffsetOfOpInterface>,
                VPUIPRegMapped_SingleOutputAsIndexOp,
                AttrSizedOperandSegments
            ]
        > {
    let summary = "Kernel Params";

    let arguments = (ins
        Variadic<AnyMemRef>:$inputs,
        Variadic<AnyMemRef>:$outputs,
        StrAttr:$kernel_type,
        ElementsAttr:$kernel_params
    );

    let results = (outs
        VPUIPRegMapped_IndexType:$index  // TODO: modify to VPUIndexType
    );

    let extraClassDeclaration = [{
        size_t getParamsStructSize();
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $inputs `:` type($inputs) `)`
        `outputs` `(` $outputs `:` type($outputs) `)`
        `kernel_type` `(` $kernel_type `)`
        `kernel_params` `(` $kernel_params `)`
        `->` type(results)
    }];
}

//
// ActKernelRangeOp
//

def VPUIPRegMapped_ActKernelRangeOp :
        VPUIPRegMapped_Op<"ActKernelRange",
            [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface>,
                DeclareOpInterfaceMethods<ELF_GetOffsetOfOpInterface>,
                VPUIPRegMapped_SingleOutputAsIndexOp
            ]
        > {
    let summary = "Activation Kernel Range";

    let arguments = (ins
        VPUIPRegMapped_IndexType:$kernel_text_index,
        VPUIPRegMapped_IndexType:$kernel_args_index,
        VPUIPRegMapped_IndexType:$kernel_entry_index
    );

    let results = (outs
        VPUIPRegMapped_IndexType:$index
    );

    let assemblyFormat = [{
        attr-dict
        `kernel_text_index` `(` $kernel_text_index `:` type($kernel_text_index) `)`
        `kernel_args_index` `(` $kernel_args_index `:` type($kernel_args_index) `)`
        `kernel_entry_index` `(` $kernel_entry_index `:` type($kernel_entry_index) `)`
        `->` type(results)
    }];
}

//
// ActKernelInvocationOp
//

def VPUIPRegMapped_ActKernelInvocationOp :
        VPUIPRegMapped_ExecutableTaskOp<"ActKernelInvocation",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "Activation Kernel Invocation";

    let arguments = (ins
        Variadic<VPUIPRegMapped_IndexType>:$waitBarriers,
        Variadic<VPUIPRegMapped_IndexType>:$updateBarriers,
        VPUIPRegMapped_IndexType:$range_index,
        DefaultValuedAttr<UI64Attr, "0">:$tile,
        UI64Attr:$start_after,
        UI64Attr:$clean_after
    );

    let results = (outs
        VPUIPRegMapped_IndexType:$index
    );

    let extraClassDeclaration = [{
        static vpux::VPUIPRegMapped::TaskType getTaskType() {
            return vpux::VPUIPRegMapped::TaskType::ActKernelInvocation;
        }
    }];

    let assemblyFormat = [{
        attr-dict
        `range_index` `(` $range_index `:` type($range_index) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `tile` `(` $tile `)`
        `start_after` `(` $start_after `)`
        `clean_after` `(` $clean_after `)`
        `->` type(results)
    }];
}

//
// DPU Variant Op
//

def VPUIPRegMapped_DPUVariantOp :
        VPUIPRegMapped_Op<"DPUVariant",
            [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface>,
                DeclareOpInterfaceMethods<ELF_GetOffsetOfOpInterface>,
                VPUIPRegMapped_SingleOutputAsIndexOp
            ]
        > {

    let summary = "DPU Variant Op";

    let results = (outs
        VPUIPRegMapped_IndexType:$index
    );

    let arguments = (ins
        VPUIPRegMapped_IndexType:$Invariant,
        I64ArrayAttr:$start,
        I64ArrayAttr:$end,
        VPU_PaddingAttr:$pad,
        VPU_MPEMode:$mpe_mode,
        OptionalAttr<IntAttr>:$cluster_id
    );
}

//
// DPU Invariant Op
//

def VPUIPRegMapped_DPUInvariantOp :
        VPUIPRegMapped_ExecutableTaskOp<"DPUInvariant",
            [
                AttrSizedOperandSegments
            ]
            # GraphRegionNoTerminator.traits
        > {

    let summary = "DPU Invariant Op";

    let arguments = (ins
        MemRefOf<[F16, BF16, quant_QuantizedType]>:$input,
        Optional<MemRefOf<[I1]>>:$input_sparsity_map,
        Optional<MemRefOf<[I32]>>:$input_storage_element_table,
        Optional<MemRefOf<[F16, BF16, quant_QuantizedType]>>:$weights,
        Optional<MemRefOf<[I1]>>:$weights_sparsity_map,
        Optional<MemRefOf<[SI32]>>:$weight_table,

        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$parent_input,
        Optional<MemRefOf<[I1]>>:$parent_input_sparsity_map,
        Optional<MemRefOf<[I32]>>:$parent_input_storage_element_table,

        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$parent_output,
        Optional<MemRefOf<[I1]>>:$parent_output_sparsity_map,

        Variadic<MemRefOf<[F16, F32, BF16, quant_QuantizedType]>>:$output_buffs,
        Optional<MemRefOf<[I1]>>:$output_sparsity_map_buff,
        Optional<MemRefOf<[UI64]>>:$profiling_data,

        VPUIP_NCETaskType:$task_type,
        VPU_MPEMode:$mpe_frequent_mode,
        OptionalAttr<I64ArrayAttr>:$kernel_size,
        OptionalAttr<I64ArrayAttr>:$kernel_strides,
        OptionalAttr<VPU_PaddingAttr>:$kernel_padding,
        OptionalAttr<IntAttr>:$activation_window_channel_length,
        OptionalAttr<UnitAttr>:$is_continued,
        OptionalAttr<IntAttr>:$cm_sp_pattern,
        OptionalAttr<UnitAttr>:$input_channels_compression,
        OptionalAttr<UnitAttr>:$is_segmented,
        OptionalAttr<IntAttr>:$out_channel_offset,
        OptionalAttr<UnitAttr>:$is_superdense,
        OptionalAttr<IntAttr>:$input_se_size,
        OptionalAttr<IntAttr>:$output_se_size,

        Variadic<VPUIPRegMapped_IndexType>:$waitBarriers,
        Variadic<VPUIPRegMapped_IndexType>:$updateBarriers,
        UI64Attr:$start_after,
        UI64Attr:$clean_after
    );

    let results = (outs
        VPUIPRegMapped_IndexType:$index
    );

    let regions = (region
        SizedRegion<1>:$ppe
    );

    let extraClassDeclaration = [{
        static vpux::VPUIPRegMapped::TaskType getTaskType() {
            return vpux::VPUIPRegMapped::TaskType::DPUInvariant;
        }

        std::size_t getBarrierHitsCount() {
            auto users = getResult().getUsers();
            return static_cast<std::size_t>(std::count_if(std::begin(users), std::end(users), [](auto user) { return mlir::isa<vpux::VPUIPRegMapped::DPUVariantOp>(user); }));
        }
    }];

    let assemblyFormat = [{
        attr-dict
        `input` `(` $input  `:` type($input) `)`
        (`input_sparsity_map` `(` $input_sparsity_map^  `:` type($input_sparsity_map) `)`)?
        (`input_storage_element_table` `(` $input_storage_element_table^  `:` type($input_storage_element_table) `)`)?
        (`weights` `(` $weights^  `:` type($weights) `)`)?
        (`weights_sparsity_map` `(` $weights_sparsity_map^  `:` type($weights_sparsity_map) `)`)?
        (`weight_table` `(` $weight_table^  `:` type($weight_table) `)`)?
        `parent_input` `(` $parent_input `:` type($parent_input) `)`
        (`parent_input_sparsity_map` `(` $parent_input_sparsity_map^  `:` type($parent_input_sparsity_map) `)`)?
        (`parent_input_storage_element_table` `(` $parent_input_storage_element_table^  `:` type($parent_input_storage_element_table) `)`)?
        `parent_output` `(` $parent_output `:` type($parent_output) `)`
        (`parent_output_sparsity_map` `(` $parent_output_sparsity_map^  `:` type($parent_output_sparsity_map) `)`)?
        (`outputs` `(` $output_buffs^ `:` type($output_buffs) `)`)?
        (`output_sparsity_map_buff` `(` $output_sparsity_map_buff^  `:` type($output_sparsity_map_buff) `)`)?
        (`profiling_data` `(` $profiling_data^  `:` type($profiling_data) `)`)?
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type($index)
        `PPE` `:` $ppe
    }];

}

//
// Network Metadtata Op
//

def VPUIPRegMapped_NetworkMetadataOp :
        VPUIPRegMapped_Op<"NetworkMetadata",
            [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface>
            ]
        > {
    let summary = "Network Metadata Op";

    let results = (outs
        VPUIPRegMapped_IndexType:$index
    );

    let extraClassDeclaration = [{
        void serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection, ::elf::NetworkMetadata& metadata);
    }];

    let assemblyFormat = [{
        attr-dict `->` type(results)
    }];
}

#endif
