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
#endif

