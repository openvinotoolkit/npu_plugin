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

        Variadic<VPUIPRegMapped_IndexType>:$waitBarriers,
        Variadic<VPUIPRegMapped_IndexType>:$updateBarriers
    );

    let results = (outs
        VPUIPRegMapped_IndexType:$barrier
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
        Optional<VPUIPRegMapped_IndexType>:$dmaList,
        Optional<VPUIPRegMapped_IndexType>:$invariantList,
        Optional<VPUIPRegMapped_IndexType>:$variantList,
        Optional<VPUIPRegMapped_IndexType>:$actKernelRanges,
        Optional<VPUIPRegMapped_IndexType>:$actKernelInvocations,
        Optional<VPUIPRegMapped_IndexType>:$barrierList,
        UI32Attr:$dmaCount,
        UI32Attr:$invariantCount,
        UI32Attr:$variantCount,
        UI32Attr:$actKernelRangesCount,
        UI32Attr:$actKernelInvocationsCount,
        UI32Attr:$barrierCount
    );

    let results = (outs
        Index:$output);

    let assemblyFormat = [{
        attr-dict
        (`dmas` `(` $dmaList^ `:` type($dmaList) `)`)?
        (`invariants` `(` $invariantList^ `:` type($invariantList) `)`)?
        (`variants` `(` $variantList^ `:` type($variantList) `)`)?
        (`actKernelRanges` `(` $actKernelRanges^ `:` type($actKernelRanges) `)`)?
        (`actKernelInvocations` `(` $actKernelInvocations^ `:` type($actKernelInvocations) `)`)?
        (`barriers` `(` $barrierList^ `:` type($barrierList) `)` )?
        `dmaCount` `(` $dmaCount `)`
        `invariantCount` `(` $invariantCount `)`
        `variantCount` `(` $variantCount `)`
        `actKernelRangesCount` `(` $actKernelRangesCount `)`
        `actKernelInvocationsCount` `(` $actKernelInvocationsCount `)`
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
// Activation Kernel Ops
//

//
// DeclareKernelTextOp
//

def VPUIPRegMapped_DeclareKernelTextOp :
        VPUIPRegMapped_Op<"DeclareKernelText",
            [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface>
                // ViewLikeOpInterface, // ?
                // DeclarationOp // checks if op has no operands and only one result
                // DotInterface ???
            ]
        > {
    let summary = "Declaration of Software Kernel .text ";

    let arguments = (ins
        StrAttr:$kernel_path

        // IntegerAttr:$section_size
    );

    let results = (outs
        VPUIPRegMapped_IndexType:$index
    );

    let builders = [
        // OpBuilder<
        //     (ins "mlir::Value":$src, "mlir::Value":$dst)
        // >
    ];

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
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface>
                // ViewLikeOpInterface,
                // DeclarationOp
                // DotInterface
            ]
        > {
    let summary = "Declaration of Software Kernel .args";

    let arguments = (ins
        StrAttr:$kernel_path

        // IntegerAttr:$section_size
    );

    let results = (outs
        VPUIPRegMapped_IndexType:$index
    );

    let builders = [
    //     OpBuilder<
    //         (ins "mlir::Value":$src, "mlir::Value":$dst)
    //     >
    ];

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
                // DeclareOpInterfaceMethods<ELF_BinaryOpInterface>
                // ViewLikeOpInterface,
                // DeclarationOp
                // DotInterface
            ]
        > {
    let summary = "Declaration of Kernel Entry";

    let arguments = (ins
        StrAttr:$kernel_path
    );

    let results = (outs
        VPUIPRegMapped_IndexType:$index
    );

    let builders = [
    //     OpBuilder<
    //         (ins "mlir::Value":$src, "mlir::Value":$dst)
    //     >
    ];

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

    let builders = [
        // OpBuilder<
        //     (ins "mlir::Value":$src, "mlir::Value":$dst)
        // >
    ];

    // let extraClassDeclaration = [{
    //     static vpux::VPUIPRegMapped::TaskType getTaskType() {
    //         return vpux::VPUIPRegMapped::TaskType::ActShave;
    //     }
    //     vpux::DotNodeColor getNodeColor() {
    //         auto memResource = VPUIPRegMapped::getPhysicalMemory(output().getType().cast<mlir::MemRefType>());
    //         if (!mlir::failed(memResource) && memResource.getValue() == VPUIPRegMapped::PhysicalMemory::DDR)
    //             return DotNodeColor::RED;
    //         return DotNodeColor::GREEN;
    //     }
    // }] # IERT_InferLayerReturnTypes<1>.impl;

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
        VPUIPRegMapped_Op<"ActKernelRange", //TaskOp implements the needed intefaces
            [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface>
                // ViewLikeOpInterface,
                // IERT_SameShape, // has method implemented, verifySameShape -- need? delete
                // IERT_SameElementType, // same
                // IERT_SameInOutDimsOrder, //same
                // DotInterface
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

    let builders = [
        // OpBuilder<
        //     (ins "mlir::Value":$src, "mlir::Value":$dst)
        // >
    ];

    // let extraClassDeclaration = [{
    //     static vpux::VPUIPRegMapped::TaskType getTaskType() {
    //         return vpux::VPUIPRegMapped::TaskType::ActShave;
    //     }
    //     vpux::DotNodeColor getNodeColor() {
    //         auto memResource = VPUIPRegMapped::getPhysicalMemory(output().getType().cast<mlir::MemRefType>());
    //         if (!mlir::failed(memResource) && memResource.getValue() == VPUIPRegMapped::PhysicalMemory::DDR)
    //             return DotNodeColor::RED;
    //         return DotNodeColor::GREEN;
    //     }
    // }] # IERT_InferLayerReturnTypes<1>.impl;

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
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface>
                // ViewLikeOpInterface,
                // IERT_SameShape,
                // IERT_SameElementType,
                // IERT_SameInOutDimsOrder,
                // DotInterface
            ]
        > {
    let summary = "Activation Kernel Invocation";

    let arguments = (ins
        VPUIPRegMapped_IndexType:$waitBarriers,
        VPUIPRegMapped_IndexType:$updateBarriers,
        VPUIPRegMapped_IndexType:$range_index,
        DefaultValuedAttr<UI64Attr, "0">:$tile, //probably not default valued
        UI64Attr:$start_after,
        UI64Attr:$clean_after
    );

    let results = (outs
        VPUIPRegMapped_IndexType:$index  // TODO: modify to VPUIndexType
    );

    let builders = [
        // OpBuilder<
        //     (ins "mlir::Value":$src, "mlir::Value":$dst)
        // >
    ];

    // let extraClassDeclaration = [{
    //     static vpux::VPUIPRegMapped::TaskType getTaskType() {
    //         return vpux::VPUIPRegMapped::TaskType::ActShave;
    //     }
    //     vpux::DotNodeColor getNodeColor() {
    //         auto memResource = VPUIPRegMapped::getPhysicalMemory(output().getType().cast<mlir::MemRefType>());
    //         if (!mlir::failed(memResource) && memResource.getValue() == VPUIPRegMapped::PhysicalMemory::DDR)
    //             return DotNodeColor::RED;
    //         return DotNodeColor::GREEN;
    //     }
    // }] # IERT_InferLayerReturnTypes<1>.impl;

    let assemblyFormat = [{
        attr-dict
        `range_index` `(` $range_index `:` type($range_index) `)`
        `waits` `(` $waitBarriers `:` type($waitBarriers) `)`
        `updates` `(` $updateBarriers `:` type($updateBarriers) `)`
        `tile` `(` $tile `)`
        `start_after` `(` $start_after `)`
        `clean_after` `(` $clean_after `)`
        `->` type(results)
    }];
}

#endif

