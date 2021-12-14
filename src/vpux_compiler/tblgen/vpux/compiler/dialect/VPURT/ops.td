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

#ifndef VPUX_COMPILER_DIALECT_VPURT_OPS
#define VPUX_COMPILER_DIALECT_VPURT_OPS

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/core/ops_interfaces.td"
include "vpux/compiler/dialect/VPURT/attributes.td"
include "vpux/compiler/dialect/VPURT/types.td"
include "vpux/compiler/dialect/VPURT/dialect.td"
include "vpux/compiler/dialect/VPURT/ops_interfaces.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/OpAsmInterface.td"

//
// Base classes
//

class VPURT_Op<string mnemonic, list<OpTrait> traits = []> :
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
                NoSideEffect,
                DeclarationOp,
                DotInterface
            ]
        > {
    let summary = "Declare VPU run-time buffer";

    let arguments = (ins
        VPURT_BufferSection:$section,
        OptionalAttr<IntAttr>:$sectionIndex,
        IntAttr:$byteOffset
    );

    let results = (outs
        AnyMemRef:$buffer
    );

    let builders = [
        OpBuilder<
            (ins "mlir::MemRefType":$type, "vpux::VPURT::BufferSection":$section, "int64_t":$byteOffset)
        >,
        OpBuilder<
            (ins "mlir::MemRefType":$type, "vpux::VPURT::BufferSection":$section, "int64_t":$sectionIndex, "int64_t":$byteOffset)
        >
    ];

    let verifier = [{
        return vpux::VPURT::verifyOp(*this);
    }];

    let assemblyFormat = [{
        $section (`[` $sectionIndex^ `]`)? `<` $byteOffset `>` attr-dict `->` type(results)
    }];

    let extraClassDeclaration = [{
        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::ORANGE;
        }
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

    let arguments = (ins
        IntAttr:$id
    );

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
        IntAttr:$id,
        IntAttr:$virtualbarrierId
    );

    let results = (outs
        VPURT_Barrier:$barrier
    );

    let extraClassDeclaration = [{
        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::AQUAMARINE;
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
        Variadic<VPURT_Barrier>:$waitBarriers,
        Variadic<VPURT_Barrier>:$updateBarriers,
        DefaultValuedAttr<BoolAttr, "false">:$isTrailingSWLayer
    );

    let regions = (region
        SizedRegion<1>:$body
    );

    let verifier = [{
        return vpux::VPURT::verifyTaskOp(*this);
    }];

    let extraClassDeclaration = [{
        vpux::VPUIP::TaskOpInterface getInnerTaskOp();

        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::AQUA;
        }

        void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

        vpux::VPU::ExecutorKind getExecutorKind();
    }];

    let assemblyFormat = [{
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        attr-dict-with-keyword
        $body
    }];
}

#endif
