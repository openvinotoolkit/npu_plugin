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

class VPUIP_TaskOp<string mnemonic, VPUIP_TaskTypeVal taskType, list<OpTrait> traits = []> :
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
    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];

    let extraClassDeclaration = [{
        static vpux::VPUIP::TaskType getTaskType() {
            return }] # taskType.enumVal # [{;
        }
    }];

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
                IsolatedFromAbove,
                HasParent<"mlir::ModuleOp">,
                SingleBlockImplicitTerminator<"vpux::VPUIP::EndOp">,
                NoRegionArguments,
                DeclareOpInterfaceMethods<NetInfoInterface>,
                DeclareOpInterfaceMethods<SymbolUserOpInterface>
            ]
        > {
    let summary = "The root object for the VPUIP Execution Graph";

    let arguments = (ins
        StrAttr:$identifier,
        FlatSymbolRefAttr:$entryPoint,
        VPUIP_ExecutionFlag:$options,
        VPUIP_ResourcesAttr:$resources,
        VPUIP_Version:$version
    );

    let regions = (region
        SizedRegion<1>:$inputsInfo,
        SizedRegion<1>:$outputsInfo
    );

    let assemblyFormat = [{
        attr-dict
        $identifier `at` $entryPoint
        `options` `:` $options
        `resources` `:` $resources
        `version` `:` $version
        `inputsInfo` `:` $inputsInfo
        `outputsInfo` `:` $outputsInfo
    }];
}

def VPUIP_TensorInfoOp :
        VPUIP_Op<
            "TensorInfo",
            [
                IsolatedFromAbove,
                HasParent<"vpux::VPUIP::GraphOp">,
                DeclareOpInterfaceMethods<DataInfoInterface>
            ]
        > {
    let summary = "Information about user inputs/outputs tensors";

    let arguments = (ins
        StrAttr:$name,
        TypeAttr:$precision,
        AffineMapAttr:$layout
    );

    let assemblyFormat = [{
        attr-dict
        $name `,` $precision `,` $layout
    }];
}

def VPUIP_EndOp :
        VPUIP_Op<
            "End",
            [
                NoSideEffect,
                Terminator,
                HasParent<"vpux::VPUIP::GraphOp">
            ]
        > {
    let summary = "End indicator for VPUIP Graph user inputs/outputs info lists";

    let assemblyFormat = "attr-dict";
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
    let summary = "MemRef value declaration";

    let arguments = (ins
        VPUIP_MemoryLocation:$locale,
        OptionalAttr<I32Attr>:$localeIndex,

        DefaultValuedAttr<I64Attr, "0">:$leadingOffset,
        DefaultValuedAttr<I64Attr, "0">:$trailingOffset,

        OptionalAttr<I64Attr>:$dataIndex,
        OptionalAttr<I64Attr>:$sparsityIndex,
        OptionalAttr<I64Attr>:$storageElementIndex
    );

    let results = (outs
        AnyMemRef:$memory
    );

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];

    let assemblyFormat = [{
        $locale (`:` $localeIndex^)? attr-dict `->` type(results)
    }];
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
            "ConfigureBarrier",
            VPUIP_TaskType_Controller
        > {
    let summary = "A task to configure the setup for a barrier";

    let arguments = (ins
        Variadic<AnyMemRef>:$inputTensors,
        Variadic<AnyMemRef>:$outputTensors,

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
                      mlir::ValueRange {}, mlir::ValueRange {},
                      mlir::ValueRange {}, mlir::ValueRange {});
            }]
        >
    ];

    let assemblyFormat = [{
        attr-dict
        (`inputs` `(` $inputTensors^ `:` type($inputTensors) `)`)?
        (`outputs` `(` $outputTensors^ `:` type($outputTensors) `)`)?
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];
}

//
// UPADMAOp
//

def VPUIP_UPADMAOp :
        VPUIP_TaskOp<
            "UPADMA",
            VPUIP_TaskType_UPADMA
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
}

//
// SoftMaxUPAOp
//

def VPUIP_SoftMaxUPAOp :
        VPUIP_TaskOp<
            "SoftMaxUPA",
            VPUIP_TaskType_UPA,
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

        I32Attr:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let builders = [
        OpBuilderDAG<
            (ins
                "mlir::Value":$input, "mlir::Value":$output,
                "mlir::IntegerAttr":$axisInd,
                "mlir::IntegerAttr":$maxShaves
            ),
            [{
                build($_builder, $_state,
                      mlir::ValueRange {input}, mlir::ValueRange {output},
                      mlir::ValueRange {}, mlir::ValueRange {},
                      axisInd, maxShaves,
                      nullptr);
            }]
        >
    ];
}

#endif
