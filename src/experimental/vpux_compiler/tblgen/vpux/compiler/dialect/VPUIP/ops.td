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
include "vpux/compiler/dialect/VPUIP/dialect.td"
include "vpux/compiler/dialect/VPUIP/types.td"
include "vpux/compiler/dialect/VPUIP/attributes.td"
include "vpux/compiler/dialect/VPUIP/ops_interfaces.td"

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
        >;

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
            (ins "mlir::Type":$memory, "vpux::VPUIP::MemoryLocation":$locale, "uint64_t":$dataIndex)
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

def VPUIP_DeclareConstantTensorOp :
        VPUIP_Op<
            "DeclareConstantTensorOp",
            [
                ConstantLike,
                NoSideEffect,
                DeclareOpInterfaceMethods<ConstantInterface>
            ]
        > {
    let summary = "Constant TensorReference value declaration";

    let arguments = (ins
        ElementsAttr:$value,
        UnitAttr:$csramCacheable
    );

    let results = (outs
        AnyStaticShapeMemRef:$output
    );

    let builders = [
        OpBuilderDAG<
            (ins "mlir::MemRefType":$type, "mlir::ElementsAttr":$value)
        >
    ];

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        ($csramCacheable^)? type($output) `=` $value
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
        OpBuilderDAG<(ins)>
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
        OpBuilderDAG<(ins)>
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
// UPADMAOp
//

def VPUIP_UPADMAOp :
        VPUIP_TaskOp<
            "UPADMA"
        > {
    let summary = "UPA DMA task";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers
    );

    let builders = [
        OpBuilderDAG<
            (ins "mlir::Value":$src, "mlir::Value":$dst)
        >
    ];

    let extraClassDeclaration = [{
        static vpux::VPUIP::TaskType getTaskType() {
            return vpux::VPUIP::TaskType::UPADMA;
        }
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output `:` type($output) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
    }];
}

//
// NNDMAOp
//

def VPUIP_NNDMAOp :
        VPUIP_TaskOp<
            "NNDMA"
        > {
    let summary = "NN DMA task";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        UnitAttr:$compression,
        DefaultValuedAttr<I32Attr, "0">:$port
    );

    let builders = [
        OpBuilderDAG<
            (ins "mlir::Value":$src, "mlir::Value":$dst)
        >
    ];

    let extraClassDeclaration = [{
        static vpux::VPUIP::TaskType getTaskType() {
            return vpux::VPUIP::TaskType::NNDMA;
        }
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output `:` type($output) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
    }];
}

//
// ConvertUPAOp
//

def VPUIP_ConvertUPAOp :
        VPUIP_TaskOp<
            "ConvertUPA",
            [
                SingleInputAndResultLayer,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<ConvertLayerInterface>,
                DeclareOpInterfaceMethods<VPUIP_UPATaskOpInterface>
            ]
        > {
    let summary = "Convert UPA SHAVE kernel";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        OptionalAttr<F32Attr>:$scale,
        OptionalAttr<F32Attr>:$bias,
        UnitAttr:$fromDetectionOutput,
        UnitAttr:$haveBatch,
        OptionalAttr<I32Attr>:$batchID,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let builders = [
        OpBuilderDAG<
            (ins "mlir::Value":$input, "mlir::Value":$output)
        >
    ];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output `:` type($output) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
    }];
}

//
// SoftMaxUPAOp
//

def VPUIP_SoftMaxUPAOp :
        VPUIP_TaskOp<
            "SoftMaxUPA",
            [
                SingleInputAndResultLayer,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<SoftMaxLayerInterface>,
                DeclareOpInterfaceMethods<VPUIP_UPATaskOpInterface>
            ]
        > {
    let summary = "SoftMax UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output,

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
            )
        >
    ];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output `:` type($output) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
    }];
}

//
// PoolingUPAOp
//

def VPUIP_PoolingUPAOp :
        VPUIP_TaskOp<
            "PoolingUPA",
            [
                SingleInputAndResultLayer,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<VPUIP_UPATaskOpInterface>
            ]
        > {
    let summary = "MAX and AVG Pooling UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        VPUIP_PoolLayerType:$type,
        I32ArrayAttr:$kernel,
        I32ArrayAttr:$strides,
        I32ArrayAttr:$padsBegin,
        I32ArrayAttr:$padsEnd,
        UnitAttr:$excludePad,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let builders = [
        OpBuilderDAG<
            (ins
                "mlir::Value":$input, "mlir::Value":$output,
                "vpux::VPUIP::PoolLayerTypeAttr":$type,
                "mlir::ArrayAttr":$kernel, "mlir::ArrayAttr":$strides,
                "mlir::ArrayAttr":$padsBegin, "mlir::ArrayAttr":$padsEnd,
                "mlir::UnitAttr":$excludePad
            )
        >
    ];

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output `:` type($output) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
    }];
}

//
// ConvolutionUPAOp
//

def VPUIP_ConvolutionUPAOp :
        VPUIP_TaskOp<
            "ConvolutionUPA",
            [
                DeclareOpInterfaceMethods<ConvolutionLayerInterface>,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<VPUIP_UPATaskOpInterface>
            ]
        > {
    let summary = "Convolution UPA SHAVE kernel (reference implementation)";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$filter,
        Optional<F16MemRef>:$bias,
        F16MemRef:$output,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        I32ArrayAttr:$strides,
        I32ArrayAttr:$dilations,
        I32ArrayAttr:$padsBegin,
        I32ArrayAttr:$padsEnd,
        I32Attr:$group,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let builders = [
        OpBuilderDAG<
            (ins
                "mlir::Value":$input, "mlir::Value":$filter, "mlir::Value":$bias, "mlir::Value":$output,
                "mlir::ArrayAttr":$strides, "mlir::ArrayAttr":$dilations,
                "mlir::ArrayAttr":$padsBegin, "mlir::ArrayAttr":$padsEnd
            )
        >
    ];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $filter `:` type($filter) (`,` $bias^ `:` type($bias))? `)`
        `outputs` `(` $output `:` type($output) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
    }];
}

//
// ReLUUPAOp
//

def VPUIP_ReLUUPAOp :
        VPUIP_TaskOp<
            "ReLUUPA",
            [
                SingleInputAndResultLayer,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<VPUIP_UPATaskOpInterface>
            ]
        > {
    let summary = "ReLU UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let builders = [
        OpBuilderDAG<
            (ins "mlir::Value":$input, "mlir::Value":$output)
        >
    ];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output `:` type($output) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
    }];
}

//
// SigmoidOp
//

def VPUIP_SigmoidUPAOp :
        VPUIP_TaskOp<
            "SigmoidUPA",
            [
                SingleInputAndResultLayer,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<VPUIP_UPATaskOpInterface>
            ]
        > {
    let summary = "Sigmoid UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let builders = [
        OpBuilderDAG<
            (ins "mlir::Value":$input, "mlir::Value":$output)
        >
    ];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output `:` type($output) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
    }];
}

//
// HSwishUPAOp
//

def VPUIP_HSwishUPAOp :
        VPUIP_TaskOp<
            "HSwishUPA",
            [
                SingleInputAndResultLayer,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<VPUIP_UPATaskOpInterface>
            ]
        > {
    let summary = "HSwish UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let builders = [
        OpBuilderDAG<
            (ins "mlir::Value":$input, "mlir::Value":$output)
        >
    ];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output `:` type($output) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
    }];
}

//
// ClampUPAOp
//

def VPUIP_ClampUPAOp :
        VPUIP_TaskOp<
            "ClampUPA",
            [
                SingleInputAndResultLayer,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<ClampLayerInterface>,
                DeclareOpInterfaceMethods<VPUIP_UPATaskOpInterface>
            ]
        > {
    let summary = "Clamp UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        F32Attr:$min,
        F32Attr:$max,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let builders = [
        OpBuilderDAG<
            (ins
                "mlir::Value":$input, "mlir::Value":$output,
                "mlir::FloatAttr":$min, "mlir::FloatAttr":$max
            )
        >
    ];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output `:` type($output) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
    }];
}

//
// EluUPAOp
//

def VPUIP_EluUPAOp :
        VPUIP_TaskOp<
            "EluUPA",
            [
                SingleInputAndResultLayer,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<EluLayerInterface>,
                DeclareOpInterfaceMethods<VPUIP_UPATaskOpInterface>
            ]
        > {
    let summary = "Elu UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        F32Attr:$x,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let builders = [
        OpBuilderDAG<
            (ins
                "mlir::Value":$input, "mlir::Value":$output,
                "mlir::FloatAttr":$x
            )
        >
    ];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output `:` type($output) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
    }];
}

#endif
