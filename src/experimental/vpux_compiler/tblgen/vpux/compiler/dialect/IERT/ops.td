//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#ifndef VPUX_COMPILER_DIALECT_IERT_OPS
#define VPUX_COMPILER_DIALECT_IERT_OPS

include "vpux/compiler/core/ops_interfaces.td"
include "vpux/compiler/dialect/IERT/dialect.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

//
// Base classes
//

class IERT_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<
            IERT_Dialect,
            mnemonic,
            traits
        >;

class IERT_LayerOp<string mnemonic, list<OpTrait> traits = []> :
        IERT_Op<
            mnemonic,
            !listconcat(
                traits,
                [
                    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                    DeclareOpInterfaceMethods<LayerInterface>
                ]
            )
        > {
    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands)
    }];
}

//
// RunTimeResourcesOp
//

def IERT_RunTimeResourcesOp :
        IERT_Op<
            "RunTimeResources",
            [
                IsolatedFromAbove,
                HasParent<"mlir::ModuleOp">,
                SingleBlockImplicitTerminator<"vpux::IERT::EndOp">,
                NoRegionArguments
            ]
        > {
    let summary = "Definition of run-time resources";

    let description = [{
        This operation defines various resources consumed at run-time:

          * Memory spaces for interal buffers.
          * Executors for asynchronous calls.

        The resources are divided onto 2 category:

          * Avaialble - total amount of resources, which run-time can consume.
          * Used - actual number of resources, which are needed for network execution.
    }];

    let regions = (region
        SizedRegion<1>:$availableMemory,
        SizedRegion<1>:$usedMemory,

        SizedRegion<1>:$availableExecutors,
        SizedRegion<1>:$usedExecutors
    );

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilderDAG<(ins)>
    ];

    let verifier = [{
        return vpux::IERT::verifyOp(*this);
    }];

    let extraClassDeclaration = [{
        static vpux::IERT::RunTimeResourcesOp getFromModule(mlir::ModuleOp module);

        vpux::IERT::MemoryResourceOp addAvailableMemory(mlir::Attribute kind, vpux::Byte size);
        vpux::IERT::MemoryResourceOp getAvailableMemory(mlir::Attribute kind);

        vpux::IERT::MemoryResourceOp setUsedMemory(mlir::Attribute kind, vpux::Byte size);
        vpux::IERT::MemoryResourceOp getUsedMemory(mlir::Attribute kind);

        vpux::IERT::ExecutorResourceOp addAvailableExecutor(mlir::Attribute kind, uint32_t count);
        vpux::IERT::ExecutorResourceOp getAvailableExecutor(mlir::Attribute kind);

        vpux::IERT::ExecutorResourceOp setUsedExecutor(mlir::Attribute kind, uint32_t count);
        vpux::IERT::ExecutorResourceOp getUsedExecutor(mlir::Attribute kind);
    }];

    let assemblyFormat = [{
        attr-dict
        `availableMemory` `:` $availableMemory
        `usedMemory` `:` $usedMemory
        `availableExecutors` `:` $availableExecutors
        `usedExecutors` `:` $usedExecutors
    }];
}

//
// MemoryResourceOp
//

def IERT_MemoryResourceOp :
        IERT_Op<
            "MemoryResource",
            [
                HasParent<"vpux::IERT::RunTimeResourcesOp">
            ]
        > {
    let summary = "Information about memory resource";

    let description = [{
        The memory resource is defined by the following attributes:

          * Kind - optional kind of memory space.
          * Size - size in bytes of memory space.
    }];

    let arguments = (ins
        OptionalAttr<AnyAttr>:$kind,
        I64Attr:$byteSize
    );

    let extraClassDeclaration = [{
        vpux::Byte size() {
            return vpux::Byte(byteSize());
        }
    }];

    let assemblyFormat = [{
        $byteSize `bytes` (`of` $kind^)?
        attr-dict
    }];
}

//
// ExecutorResourceOp
//

def IERT_ExecutorResourceOp :
        IERT_Op<
            "ExecutorResource",
            [
                HasParent<"vpux::IERT::RunTimeResourcesOp">
            ]
        > {
    let summary = "Information about executor resource";

    let description = [{
        The executor resource is defined by the following attributes:

          * Kind - optional kind of the executor.
          * Count - number of executor units.
    }];

    let arguments = (ins
        AnyAttr:$kind,
        I32Attr:$count
    );

    let assemblyFormat = [{
        $count `of` $kind
        attr-dict
    }];
}

//
// EndOp
//

def IERT_EndOp :
        IERT_Op<
            "End",
            [
                NoSideEffect,
                Terminator,
                NativeOpTrait<"ReturnLike">,
                HasParent<"vpux::IERT::RunTimeResourcesOp">
            ]
        > {
    let summary = "End indicator for IERT resources section";

    let assemblyFormat = "attr-dict";
}

//
// ConvertOp
//

def IERT_ConvertOp :
        IERT_LayerOp<
            "Convert",
            [
                SameOperandsShape,
                DeclareOpInterfaceMethods<ConvertLayerInterface>
            ]
        > {
    let summary = "InferenceEngine run-time Convert layer";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output
    );
}

//
// SoftMaxOp
//

def IERT_SoftMaxOp :
        IERT_LayerOp<
            "SoftMax",
            [
                SameOperandsShape,
                SameOperandsElementType,
                DeclareOpInterfaceMethods<SoftMaxLayerInterface>
            ]
        > {
    let summary = "InferenceEngine run-time SoftMax layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$output,

        I32Attr:$axisInd
    );
}

#endif
