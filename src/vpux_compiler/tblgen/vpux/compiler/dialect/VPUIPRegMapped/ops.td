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

class VPUIPRegMapped_TaskOp<int numResult, string mnemonic, list<OpTrait> traits = []> :
        VPUIPRegMapped_Op<
            mnemonic,
            !listconcat(
                traits,
                [
                    AttrSizedOperandSegments,
                    InferTypeOpInterface,
                    MemoryEffectsOpInterface,
                    DeclareOpInterfaceMethods<VPUIP_LayerOpInterface>, 
                    VPUIP_AsyncLayerOpInterface,
                    DeclareOpInterfaceMethods<VPUIPRegMapped_TaskOpInterface>,
                    DeclareOpInterfaceMethods<ELF_BinaryOpInterface>
                ]
            )
        > {
    code commonExtraClassDeclaration = VPUIP_InferLayerReturnTypes<numResult>.impl;

    let extraClassDeclaration = commonExtraClassDeclaration;
}

//
// ConfigureBarrierOp
//

def VPUIPRegMapped_ConfigureBarrierOp :
        VPUIPRegMapped_Op<"ConfigureBarrier",
            [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface>,
                DeclarationOp,
                DotInterface
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
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface>,
                DeclareOpInterfaceMethods<ELF_GetOffsetOfOpInterface>
            ]
        > {
    let summary = "Task representing the MappedInference structure";

    let arguments = (ins
        Optional<VPUIPRegMapped_IndexType>:$dmaTasks,
        Optional<VPUIPRegMapped_IndexType>:$invariantTasks,
        Optional<VPUIPRegMapped_IndexType>:$variantTasks,
        Optional<VPUIPRegMapped_IndexType>:$actKernelRanges,
        Optional<VPUIPRegMapped_IndexType>:$actKernelInvocations,
        Optional<VPUIPRegMapped_IndexType>:$barrierTasks,
        UI32Attr:$dmaCount,
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
        VPUIPRegMapped_Op<"NNDMA",
            [
                MemoryEffectsOpInterface,
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<VPUIPRegMapped_TaskOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface>,
                DeclareOpInterfaceMethods<ELF_GetOffsetOfOpInterface>,
                DotInterface
            ]
        > {
    let summary = "NN DMA task";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output_buff,
        Optional<VPUIPRegMapped_IndexType>:$previousDMAIdx,

        Variadic<VPUIPRegMapped_IndexType>:$waitBarriers,
        Variadic<VPUIPRegMapped_IndexType>:$updateBarriers,
        UnitAttr:$compression,
        DefaultValuedAttr<IntAttr, "0">:$port,
        UI64Attr:$start_after
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
            return vpux::VPUIPRegMapped::TaskType::NNDMA;
        }
        
        vpux::DotNodeColor getNodeColor() {
            const auto memKind = getType().cast<vpux::NDTypeInterface>().getMemoryKind();
            return memKind == vpux::VPU::MemoryKind::DDR ? vpux::DotNodeColor::RED : vpux::DotNodeColor::GREEN;
        }

        void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects) {
            vpux::VPUIPRegMapped::getTaskEffects(this->getOperation(), effects);
        }

    }] # VPUIP_InferLayerReturnTypes<1>.impl;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        ( `previousDMA` `(` $previousDMAIdx^ `:` type($previousDMAIdx) `)`)?
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `start_after` `(` $start_after `)`
        `->` type(results)
    }];
}

//
// Activation Kernel Ops
//

//
// DeclareKernelTextOp
//

def VPUIPRegMapped_DeclareKernelTextOp :
        VPUIPRegMapped_Op<"DeclareKernelText",
            [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface>,
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
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface>
            ]
        > {
    let summary = "Kernel Params";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output,
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
        `input` `(` $input `:` type($input) `)`
        `output` `(` $output `:` type($output) `)`
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
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface>
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
        VPUIPRegMapped_Op<"ActKernelInvocation",
            [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface>,
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
        VPUIPRegMapped_IndexType:$index  // TODO: modify to VPUIndexType
    );

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
        void serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection, vpux::ELF::NetworkMetadata& metadata);
    }];

    let assemblyFormat = [{
        attr-dict `->` type(results)
    }];
}



#endif

