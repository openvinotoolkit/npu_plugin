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

#ifndef VPUX_COMPILER_DIALECT_VPUIP_OPS
#define VPUX_COMPILER_DIALECT_VPUIP_OPS

include "vpux/compiler/core/ops_interfaces.td"
include "vpux/compiler/dialect/VPUIP/attributes.td"
include "vpux/compiler/dialect/VPUIP/dialect.td"
include "vpux/compiler/dialect/VPUIP/ops_interfaces.td"
include "vpux/compiler/dialect/VPUIP/types.td"

include "mlir/Interfaces/CopyOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

//
// Base classes
//

class VPUIP_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<
            VPUIP_Dialect,
            mnemonic,
            traits
        >;

class VPUIP_TaskOp<string mnemonic, list<OpTrait> traits = []> :
        VPUIP_Op<
            mnemonic,
            !listconcat(
                traits,
                [
                    AttrSizedOperandSegments,
                    MemoryEffectsOpInterface,
                    DeclareOpInterfaceMethods<VPUIP_TaskOpInterface>
                ]
            )
        > {
    let assemblyFormat = [{
        attr-dict
        (`inputs` `(` $inputTensors^ `:` type($inputTensors) `)`)?
        (`outputs` `(` $outputTensors^ `:` type($outputTensors) `)`)?
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
    }];
}

//
// GraphOp
//

def VPUIP_GraphOp :
        VPUIP_Op<
            "Graph",
            [
                HasParent<"mlir::ModuleOp">
            ]
        > {
    let summary = "The root object for the VPUIP Execution Graph";

    let arguments = (ins
        VPUIP_ExecutionFlag:$options,
        VPUIP_Version:$version
    );

    let extraClassDeclaration = [{
        static vpux::VPUIP::GraphOp getFromModule(mlir::ModuleOp module);
    }];

    let assemblyFormat = [{
        attr-dict
        `options` `:` $options
        `version` `:` $version
    }];
}

//
// DeclareTensorOp
//

def VPUIP_DeclareTensorOp :
        VPUIP_Op<
            "DeclareTensor",
            [
                NoSideEffect
            ]
        > {
    let summary = "TensorReference value declaration";

    let arguments = (ins
        VPUIP_MemoryLocation:$locale,
        OptionalAttr<I32Attr>:$localeIndex,

        I64Attr:$dataIndex,
        OptionalAttr<I64Attr>:$sparsityIndex,
        OptionalAttr<I64Attr>:$storageElementIndex,
        OptionalAttr<I32Attr>:$storageElementSize,

        OptionalAttr<I32Attr>:$leadingOffset,
        OptionalAttr<I32Attr>:$trailingOffset
    );

    let results = (outs
        AnyMemRef:$memory
    );

    let builders = [
        OpBuilderDAG<
            (ins "mlir::Type":$memory, "vpux::VPUIP::MemoryLocation":$locale, "uint64_t":$dataIndex),
            [{
                build($_builder, $_state, memory, locale,
                      nullptr, // localeIndex
                      dataIndex,
                      nullptr, // sparsityIndex
                      nullptr, // storageElementIndex
                      nullptr, // storageElementSize
                      nullptr, // leadingOffset
                      nullptr  // trailingOffset
                );
            }]
        >
    ];

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];

    let assemblyFormat = [{
        $locale (`[` $localeIndex^ `]`)? `<` $dataIndex `>` attr-dict `->` type(results)
    }];
}

//
// DeclareConstantTensorOp
//

def DenseElementsAttr :
        ElementsAttrBase<
            CPred<"$_self.isa<mlir::DenseElementsAttr>()">,
            "constant dense tensor attribute"
        > {
    let storageType = "mlir::DenseElementsAttr";
    let returnType = "mlir::DenseElementsAttr";
    let convertFromStorage = "$_self";
}

def VPUIP_DeclareConstantTensorOp :
        VPUIP_Op<
            "DeclareConstantTensorOp",
            [
                ConstantLike,
                NoSideEffect
            ]
        > {
    let summary = "Constant TensorReference value declaration";

    let arguments = (ins
        DenseElementsAttr:$content,

        UnitAttr:$csramCacheable
    );

    let results = (outs
        AnyMemRef:$memory
    );

    let builders = [
        OpBuilderDAG<
            (ins "mlir::MemRefType":$type, "mlir::DenseElementsAttr":$content),
            [{
                build($_builder, $_state, type, content, nullptr);
            }]
        >
    ];

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];

    let assemblyFormat = [{
        $content attr-dict `->` type(results)
    }];

    let hasFolder = 1;
}

//
// DeclareBarrierOp
//

def VPUIP_DeclareBarrierOp :
        VPUIP_Op<
            "DeclareBarrier",
            [
                NoSideEffect
            ]
        > {
    let summary = "VPUIP Barrier value static declaration";

    let results = (outs
        VPUIP_Barrier:$barrier
    );

    let builders = [
        OpBuilderDAG<
            (ins),
            [{
                build($_builder, $_state, vpux::VPUIP::BarrierType::get($_builder.getContext()));
            }]
        >
    ];

    let assemblyFormat = [{
        attr-dict `->` type(results)
    }];
}

//
// ConfigureBarrierOp
//

def VPUIP_ConfigureBarrierOp :
        VPUIP_TaskOp<
            "ConfigureBarrier"
        > {
    let summary = "A task to configure the setup for a barrier";

    let arguments = (ins
        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers
    );

    let results = (outs
        VPUIP_Barrier:$barrier
    );

    let builders = [
        OpBuilderDAG<
            (ins),
            [{
                build($_builder, $_state,
                      vpux::VPUIP::BarrierType::get($_builder.getContext()),
                      mlir::ValueRange {}, mlir::ValueRange {});
            }]
        >
    ];

    let extraClassDeclaration = [{
        static vpux::VPUIP::TaskType getTaskType() {
            return vpux::VPUIP::TaskType::Controller;
        }
    }];

    let assemblyFormat = [{
        attr-dict
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];
}

//
// NNDMAOp
//

def VPUIP_NNDMAOp :
        VPUIP_TaskOp<
            "NNDMA",
            VPUIP_TaskType_NNDMA,
            [
                VPUIP_UPATaskTrait
            ]
        > {
    let summary = "NN DMA task";

    let arguments = (ins
        Variadic<AnyMemRef>:$inputTensors,
        Variadic<AnyMemRef>:$outputTensors,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers
    );

    let builders = [
        OpBuilderDAG<
            (ins "mlir::Value":$src, "mlir::Value":$dst),
            [{
                build($_builder, $_state,
                      mlir::ValueRange {src}, mlir::ValueRange {dst},
                      mlir::ValueRange {}, mlir::ValueRange {});
            }]
        >
    ];
}

//
// UPADMAOp
//

def VPUIP_UPADMAOp :
        VPUIP_TaskOp<
            "UPADMA"
        > {
    let summary = "UPA DMA task";

    let arguments = (ins
        Variadic<AnyMemRef>:$inputTensors,
        Variadic<AnyMemRef>:$outputTensors,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers
    );

    let builders = [
        OpBuilderDAG<
            (ins "mlir::Value":$src, "mlir::Value":$dst),
            [{
                build($_builder, $_state,
                      mlir::ValueRange {src}, mlir::ValueRange {dst},
                      mlir::ValueRange {}, mlir::ValueRange {});
            }]
        >
    ];

    let extraClassDeclaration = [{
        static vpux::VPUIP::TaskType getTaskType() {
            return vpux::VPUIP::TaskType::UPADMA;
        }
    }];
}

//
// SoftMaxUPAOp
//

def VPUIP_SoftMaxUPAOp :
        VPUIP_TaskOp<
            "SoftMaxUPA",
            [
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<SoftMaxLayerInterface>,
                DeclareOpInterfaceMethods<VPUIP_UPATaskOpInterface>
            ]
        > {
    let summary = "SoftMax UPA SHAVE kernel";

    let arguments = (ins
        Variadic<F16MemRef>:$inputTensors,
        Variadic<F16MemRef>:$outputTensors,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        I32Attr:$axisInd,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let builders = [
        OpBuilderDAG<
            (ins
                "mlir::Value":$input, "mlir::Value":$output,
                "mlir::IntegerAttr":$axisInd
            ),
            [{
                build($_builder, $_state,
                      mlir::ValueRange {input}, mlir::ValueRange {output},
                      mlir::ValueRange {}, mlir::ValueRange {},
                      axisInd,
                      nullptr, nullptr);
            }]
        >
    ];
}

#endif
