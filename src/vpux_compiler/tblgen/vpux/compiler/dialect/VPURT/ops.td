//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPURT_OPS
#define VPUX_COMPILER_DIALECT_VPURT_OPS

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/core/ops_interfaces.td"
include "vpux/compiler/dialect/VPUIP/types.td"
include "vpux/compiler/dialect/VPURT/attributes.td"
include "vpux/compiler/dialect/VPURT/types.td"
include "vpux/compiler/dialect/VPURT/dialect.td"
include "vpux/compiler/dialect/VPURT/ops_interfaces.td"
include "vpux/compiler/dialect/ELF/ops_interfaces.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/OpAsmInterface.td"

//
// Base classes
//

class VPURT_Op<string mnemonic, list<Trait> traits = []> :
        Op<
            VPURT_Dialect,
            mnemonic,
            traits
        >;


//
// SWRunTimeResourcesOp
//

def VPURT_SWRunTimeOp :
        VPURT_Op<"SW.Runtime"> {
    let summary = "Definition of run-time";

    let description = [{
        This operation defines various resources consumed by compiled-in Runtime Kernel:

          * stack configuration
          * Runtime kernel entry point
    }];

    let arguments = (ins
        SymbolRefAttr:$entryPoint,
        I64ArrayAttr : $stacks
    );

    let assemblyFormat = [{
        attr-dict-with-keyword
        `entryPoint` `:` $entryPoint
        `stack_configuration` `:` $stacks
    }];
}


//
// DeclareBufferOp
//

def VPURT_DeclareBufferOp :
        VPURT_Op<
            "DeclareBuffer",
            [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface>,
                NoSideEffect,
                DeclarationOp,
                DotInterface
            ]
        > {
    let summary = "Declare VPU run-time buffer";

    let arguments = (ins
        VPURT_BufferSection:$section,
        OptionalAttr<I64ArrayAttr>:$sectionIndex,
        IntAttr:$byteOffset,
        OptionalAttr<IntAttr>:$swizzlingKey
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_Buffer, VPUIP_DistributedBuffer]>:$buffer
    );

    let builders = [
        OpBuilder<
            (ins "mlir::Type":$type, "vpux::VPURT::BufferSection":$section, "int64_t":$byteOffset)
        >,
        OpBuilder<
            (ins "mlir::Type":$type, "vpux::VPURT::BufferSection":$section, "ArrayRef<int64_t>":$sectionIndex, "int64_t":$byteOffset)
        >,
        OpBuilder<
            (ins "mlir::Type":$type, "vpux::VPURT::BufferSection":$section, "int64_t":$sectionIndex, "int64_t":$byteOffset)
        >,
        OpBuilder<
            (ins "mlir::Type":$type, "vpux::VPURT::BufferSection":$section, "ArrayRef<int64_t>":$sectionIndex, "int64_t":$byteOffset, "int64_t":$swizzlingKey)
        >,
        OpBuilder<
            (ins "mlir::Type":$type, "vpux::VPURT::BufferSection":$section, "int64_t":$sectionIndex, "int64_t":$byteOffset, "int64_t":$swizzlingKey)
        >
    ];

    let hasVerifier = 1;

    let assemblyFormat = [{
        $section ($sectionIndex^)? ` ``<` $byteOffset `>` attr-dict `->` type(results)
    }];

    let extraClassDeclaration = [{
        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::ORANGE;
        }
        SmallVector<int64_t> getNonEmptySectionIndex();
    }];
}

//
// DeclareVirtualBarrierOp
//

def VPURT_DeclareVirtualBarrierOp :
        VPURT_Op<
            "DeclareVirtualBarrier",
            [
                NoSideEffect,
                DeclarationOp
            ]
        > {
    let summary = "VPURT virtual Barrier declaration";

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

def VPURT_ConfigureBarrierOp :
        VPURT_Op<"ConfigureBarrier",
            [
                DeclareOpInterfaceMethods<VPURT_SerializeInterface>,
                DeclarationOp,
                DotInterface
            ]
        > {
    let summary = "A task to configure the setup for a barrier";

    let arguments = (ins
        IntAttr:$id
    );

    let results = (outs
        VPURT_Barrier:$barrier
    );

    let extraClassDeclaration = [{
        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::AQUA;
        }

        bool printAttributes(llvm::raw_ostream& os) {
            os << "id: " << id();
            return true;
        }
    }];

    let assemblyFormat = [{
        `<` $id `>` attr-dict `->` type(results)
    }];
}

//
// TaskOp
//

def VPURT_TaskOp :
    VPURT_Op<"Task",
            [
                RecursiveSideEffects,
                MemoryEffectsOpInterface,
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<VPURT_SerializeInterface>,
                DotInterface
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Hold common scheduling logic (barriers configuration and profiling support).";

    let arguments = (ins
        Optional<MemRefOf<[UI32]>>:$profiling_data,
        Variadic<VPURT_Barrier>:$waitBarriers,
        Variadic<VPURT_Barrier>:$updateBarriers,
        DefaultValuedAttr<BoolAttr, "false">:$isTrailingSWLayer
    );

    let regions = (region
        SizedRegion<1>:$body
    );

    let builders = [
        OpBuilder<(ins
            "mlir::ValueRange":$waitBarriers, "mlir::ValueRange":$updateBarriers
        )>
    ];

    let extraClassDeclaration = [{
        mlir::Operation* getInnerTaskOp();

        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::AQUA;
        }

        void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

        vpux::VPU::ExecutorKind getExecutorKind();
    }];

    let assemblyFormat = [{
        (`profiling_data` `(` $profiling_data^  `:` type($profiling_data) `)`)?
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        attr-dict-with-keyword
        $body
    }];

    let hasVerifier = 1;
}

//
// Alloc
//

def VPURT_Alloc :
        VPURT_Op<
            "Alloc",
            [
                DeclarationOp,
                DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
            ]
        > {
    let summary = "Alloc VPUIP buffer";

    let arguments = (ins
        OptionalAttr<IntAttr>:$alignment,
        OptionalAttr<IntAttr>:$swizzlingKey
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_Buffer]>:$buffer
    );

    let assemblyFormat = [{
        attr-dict `->` type(results)
    }];
}

//
// AllocDistributed
//

def VPURT_AllocDistributed :
        VPURT_Op<
            "AllocDistributed",
            [
                DeclarationOp,
                DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
            ]
        > {
    let summary = "Alloc VPUIP distributed buffer";

    let arguments = (ins
        OptionalAttr<IntAttr>:$alignment,
        OptionalAttr<IntAttr>:$swizzlingKey
    );

    let results = (outs
        VPUIP_DistributedBuffer:$buffer
    );

    let assemblyFormat = [{
        attr-dict `->` qualified(type(results))
    }];
}

#endif
