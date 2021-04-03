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
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/IERT/ops_interfaces.td"
include "vpux/compiler/dialect/VPUIP/dialect.td"
include "vpux/compiler/dialect/VPUIP/types.td"
include "vpux/compiler/dialect/VPUIP/attributes.td"
include "vpux/compiler/dialect/VPUIP/ops_interfaces.td"

include "mlir/Dialect/Quant/QuantOpsBase.td"
include "mlir/Interfaces/CopyOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
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
        OpBuilder<
            (ins "mlir::Type":$memory, "vpux::VPUIP::MemoryLocation":$locale, "uint64_t":$dataIndex)
        >,
        OpBuilder<
            (ins "mlir::Type":$memory, "vpux::VPUIP::MemoryLocation":$locale, "uint32_t":$localeIndex, "uint64_t":$dataIndex)
        >
    ];

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];

    let assemblyFormat = [{
        $locale (` ` `[` $localeIndex^ `]`)? `<` $dataIndex `>` attr-dict `->` type(results)
    }];
}

//
// DeclareConstantTensorOp
//

def VPUIP_DeclareConstantTensorOp :
        VPUIP_Op<
            "DeclareConstantTensor",
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
        OpBuilder<
            (ins "mlir::MemRefType":$type, "mlir::ElementsAttr":$value)
        >
    ];

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        type($output) `=` $value
    }];
}

//
// DeclareVirtualBarrierOp
//

def VPUIP_DeclareVirtualBarrierOp :
        VPUIP_Op<
            "DeclareVirtualBarrier",
            [
                NoSideEffect
            ]
        > {
    let summary = "VPUIP virtual Barrier declaration";

    let results = (outs
        VPUIP_Barrier:$barrier
    );

    let builders = [
        OpBuilder<(ins)>
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
        I32Attr:$id,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers
    );

    let results = (outs
        VPUIP_Barrier:$barrier
    );

    let builders = [
        OpBuilder<(ins "uint32_t":$id)>
    ];

    let extraClassDeclaration = [{
        static vpux::VPUIP::TaskType getTaskType() {
            return vpux::VPUIP::TaskType::Controller;
        }
    }];

    let assemblyFormat = [{
        attr-dict
        `<` $id `>`
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
            [
                SingleInputAndResultLayer,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_SameDimsOrder,
                DeclareOpInterfaceMethods<LayerInterface>
            ]
        > {
    let summary = "UPA DMA task";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers
    );

    let builders = [
        OpBuilder<
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
            "NNDMA",
            [
                SingleInputAndResultLayer,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_SameDimsOrder,
                DeclareOpInterfaceMethods<LayerInterface>
            ]
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
        OpBuilder<
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
                VPUIP_SameShape,
                VPUIP_SameDimsOrder,
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
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output)
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
// SoftMaxUPAOp
//

def VPUIP_SoftMaxUPAOp :
        VPUIP_TaskOp<
            "SoftMaxUPA",
            [
                SingleInputAndResultLayer,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_SameDimsOrder,
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
        OpBuilder<
            (ins
                "mlir::Value":$input, "mlir::Value":$output,
                "mlir::IntegerAttr":$axisInd
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
// PoolingUPAOp
//

def VPUIP_PoolingUPAOp :
        VPUIP_TaskOp<
            "PoolingUPA",
            [
                SingleInputAndResultLayer,
                VPUIP_SameElementType,
                VPUIP_SameDimsOrder,
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
        OpBuilder<
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
                VPUIP_SameElementType,
                VPUIP_SameDimsOrder,
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
        I32Attr:$groups,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let builders = [
        OpBuilder<
            (ins
                "mlir::Value":$input, "mlir::Value":$filter, "mlir::Value":$bias, "mlir::Value":$output,
                "mlir::ArrayAttr":$strides, "mlir::ArrayAttr":$dilations,
                "mlir::ArrayAttr":$padsBegin, "mlir::ArrayAttr":$padsEnd,
                "uint32_t":$groups
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
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_SameDimsOrder,
                VPUIP_Legacy4D,
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
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output)
        >
    ];

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
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
// SigmoidOp
//

def VPUIP_SigmoidUPAOp :
        VPUIP_TaskOp<
            "SigmoidUPA",
            [
                SingleInputAndResultLayer,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_SameDimsOrder,
                VPUIP_Legacy4D,
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
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output)
        >
    ];

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
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
// HSwishUPAOp
//

def VPUIP_HSwishUPAOp :
        VPUIP_TaskOp<
            "HSwishUPA",
            [
                SingleInputAndResultLayer,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_SameDimsOrder,
                VPUIP_Legacy4D,
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
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output)
        >
    ];

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
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
// ClampUPAOp
//

def VPUIP_ClampUPAOp :
        VPUIP_TaskOp<
            "ClampUPA",
            [
                SingleInputAndResultLayer,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_SameDimsOrder,
                VPUIP_Legacy4D,
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
        OpBuilder<
            (ins
                "mlir::Value":$input, "mlir::Value":$output,
                "mlir::FloatAttr":$min, "mlir::FloatAttr":$max
            )
        >
    ];

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
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
// EluUPAOp
//

def VPUIP_EluUPAOp :
        VPUIP_TaskOp<
            "EluUPA",
            [
                SingleInputAndResultLayer,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_SameDimsOrder,
                VPUIP_Legacy4D,
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
        OpBuilder<
            (ins
                "mlir::Value":$input, "mlir::Value":$output,
                "mlir::FloatAttr":$x
            )
        >
    ];

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
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
// TanhUPAOp
//

def VPUIP_TanhUPAOp :
        VPUIP_TaskOp<
            "TanhUPA",
            [
                SingleInputAndResultLayer,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_SameDimsOrder,
                VPUIP_Legacy4D,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<VPUIP_UPATaskOpInterface>
            ]
        > {
    let summary = "Tanh UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output)
        >
    ];

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
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
// FakeQuantizeUPAOp
//

def VPUIP_FakeQuantizeUPAOp :
        VPUIP_TaskOp<
            "FakeQuantizeUPA",
            [
                SingleInputAndResultLayer,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_SameDimsOrder,
                VPUIP_Legacy4D,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<VPUIP_UPATaskOpInterface>
            ]
        > {
    let summary = "FakeQuantize UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        I32Attr:$levels,
        ElementsAttr:$input_low,
        ElementsAttr:$input_high,
        ElementsAttr:$output_low,
        ElementsAttr:$output_high,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let builders = [
        OpBuilder<
            (ins
                "mlir::Value":$input, "mlir::Value":$output,
                "uint32_t":$levels,
                "mlir::ElementsAttr":$input_low, "mlir::ElementsAttr":$input_high,
                "mlir::ElementsAttr":$output_low, "mlir::ElementsAttr":$output_high
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
// QuantCastUPAOp
//

def VPUIP_QuantCastUPAOp :
        VPUIP_TaskOp<
            "QuantCastUPA",
            [
                SingleInputAndResultLayer,
                VPUIP_SameShape,
                VPUIP_SameDimsOrder,
                VPUIP_Legacy4D,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<VPUIP_UPATaskOpInterface>
            ]
        > {
    let summary = "FakeQuantize UPA SHAVE kernel";

    let arguments = (ins
        MemRefOf<[F16, quant_QuantizedType]>:$input,
        MemRefOf<[F16, quant_QuantizedType]>:$output,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let builders = [
        OpBuilder<(ins "mlir::Value":$input, "mlir::Value":$output)>
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
// PReluUPAOp
//

def VPUIP_PReluUPAOp :
        VPUIP_TaskOp<
            "PReluUPA",
            [
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<PReluLayerInterface>,
                DeclareOpInterfaceMethods<VPUIP_UPATaskOpInterface>
            ]
        > {
    let summary = "PRelu UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$negative_slope,
        F16MemRef:$output,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let builders = [
        OpBuilder<
            (ins
                "mlir::Value":$input, "mlir::Value":$negative_slope,
                "mlir::Value":$output
            )
        >
    ];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $negative_slope `:` type($negative_slope) `)`
        `outputs` `(` $output `:` type($output) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
    }];
}

//
// LeakyReluUPAOp
//

def VPUIP_LeakyReluUPAOp :
        VPUIP_TaskOp<
            "LeakyReluUPA",
            [
                SingleInputAndResultLayer,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<LeakyReluLayerInterface>,
                DeclareOpInterfaceMethods<VPUIP_UPATaskOpInterface>
            ]
        > {
    let summary = "LeakyRelu UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        F32Attr:$negative_slope,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let builders = [
        OpBuilder<
            (ins
                "mlir::Value":$input, "mlir::Value":$output,
                "mlir::FloatAttr":$negative_slope
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
// EltwiseUPAOp
//

def VPUIP_EltwiseUPAOp :
        VPUIP_TaskOp<
            "EltwiseUPA",
            [
                VPUIP_SameElementType,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<EltwiseLayerInterface>,
                DeclareOpInterfaceMethods<VPUIP_UPATaskOpInterface>
            ]
        > {
    let summary = "Eltwise UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input1,
        F16MemRef:$input2,
        F16MemRef:$output,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        VPUIP_EltwiseLayerType:$type,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let builders = [
        OpBuilder<
            (ins
                "mlir::Value":$input1, "mlir::Value":$input2,
                "mlir::Value":$output,
                "vpux::VPUIP::EltwiseLayerTypeAttr":$type
            )
        >
    ];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input1 `:` type($input1) `,` $input2 `:` type($input2) `)`
        `outputs` `(` $output `:` type($output) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
    }];
}

//
// DetectionOutput
//

def VPUIP_DetectionOutputUPAOp :
        VPUIP_TaskOp<
            "DetectionOutputUPA",
            [
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<DetectionOutputLayerInterface>,
                DeclareOpInterfaceMethods<VPUIP_UPATaskOpInterface>
            ]
        > {
    let summary = "DetectionOutput UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$in_box_logits,
        F16MemRef:$in_class_preds,
        F16MemRef:$in_proposals,
        Optional<F16MemRef>:$in_additional_preds,
        Optional<F16MemRef>:$in_additional_proposals,

        F16MemRef:$output,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        IE_DetectionOutputAttrs:$attr,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let builders = [
        OpBuilder<
            (ins
                "mlir::Value":$in_box_logits, "mlir::Value":$in_class_preds,"mlir::Value":$in_proposals,
                "mlir::Value":$in_additional_preds, "mlir::Value":$in_additional_proposals,
                "mlir::Value":$output, "vpux::IE::DetectionOutputAttr":$attr
            )
        >
    ];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $in_box_logits `:` type($in_box_logits) `,` $in_class_preds `:` type($in_class_preds) `,` $in_proposals `:` type($in_proposals) (`,` $in_additional_preds^ `:` type($in_additional_preds))? (`,` $in_additional_proposals^ `:` type($in_additional_proposals))? `)`
        `outputs` `(` $output `:` type($output) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
    }];
}

//
// SwishUPAOp
//

def VPUIP_SwishUPAOp :
        VPUIP_TaskOp<
            "SwishUPA",
            [
                SingleInputAndResultLayer,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<VPUIP_UPATaskOpInterface>
            ]
        > {
    let summary = "Swish UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        F32Attr:$beta_value,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let builders = [
        OpBuilder<
            (ins
                "mlir::Value":$input, "mlir::Value":$output,
                "mlir::FloatAttr":$beta_value
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
// GRNUPAOp
//

def VPUIP_GRNUPAOp :
        VPUIP_TaskOp<
            "GRNUPA",
            [
                SingleInputAndResultLayer,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<GRNLayerInterface>,
                DeclareOpInterfaceMethods<VPUIP_UPATaskOpInterface>
            ]
        > {
    let summary = "GRN UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        F32Attr:$bias,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let builders = [
        OpBuilder<
            (ins
                "mlir::Value":$input, "mlir::Value":$output,
                "mlir::FloatAttr":$bias
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
// ScaleShiftUPAOp
//

def VPUIP_ScaleShiftUPAOp :
        VPUIP_TaskOp<
            "ScaleShiftUPA",
            [
                VPUIP_SameElementType,
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<ScaleShiftLayerInterface>,
                DeclareOpInterfaceMethods<VPUIP_UPATaskOpInterface>
            ]
        > {
    let summary = "ScaleShift UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        Optional<F16MemRef>:$weights,
        Optional<F16MemRef>:$biases,
        F16MemRef:$output,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let builders = [
        OpBuilder<
            (ins
                "mlir::Value":$input, "mlir::Value":$weights,
                "mlir::Value":$biases,"mlir::Value":$output
            )
        >
    ];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) (`,` $weights^ `:` type($weights))? (`,` $biases^ `:` type($biases))? `)`
        `outputs` `(` $output `:` type($output) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
    }];

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
    }];
}

//
// PerAxisTileUPAOp
//

def VPUIP_PerAxisTileUPAOp :
        VPUIP_TaskOp<
            "PerAxisTileUPA",
            [
                SingleInputAndResultLayer,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<VPUIP_UPATaskOpInterface>
            ]
        > {
    let summary = "Tile for per axis case UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        I32Attr:$axis,
        I32Attr:$tiles,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let builders = [
        OpBuilder<
            (ins
                "mlir::Value":$input, "mlir::Value":$output,
                "mlir::IntegerAttr":$axis, "mlir::IntegerAttr":$tiles
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
// NegativeUPAOp
//

def VPUIP_NegativeUPAOp :
        VPUIP_TaskOp<
            "NegativeUPA",
            [
                SingleInputAndResultLayer,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<VPUIP_UPATaskOpInterface>
            ]
        > {
    let summary = "Negative UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let builders = [
        OpBuilder<
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
// ROIPoolingUPAOp
//

def VPUIP_ROIPoolingUPAOp :
        VPUIP_TaskOp<
            "ROIPoolingUPA",
            [
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<ROIPoolingLayerInterface>,
                DeclareOpInterfaceMethods<VPUIP_UPATaskOpInterface>
            ]
        > {
    let summary = "ROIPooling UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$coords,
        F16MemRef:$output,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        I32ArrayAttr:$output_size,
        F32Attr:$spatial_scale,
        IE_ROIPoolingMethod:$method,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let builders = [
        OpBuilder<
            (ins
                "mlir::Value":$input, "mlir::Value":$coords,
                "mlir::Value":$output,
                "mlir::ArrayAttr":$output_size, "mlir::FloatAttr":$spatial_scale, "IE::ROIPoolingMethodAttr":$method
            )
        >
    ];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $coords `:` type($coords) `)`
        `outputs` `(` $output `:` type($output) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
    }];
}

//
// PermuteUPAOp
//

def VPUIP_PermuteUPAOp :
        VPUIP_TaskOp<
            "PermuteUPA",
            [
                SingleInputAndResultLayer,
                DeclareOpInterfaceMethods<LayerInterface>,
                DeclareOpInterfaceMethods<VPUIP_UPATaskOpInterface>
            ]
        > {
    let summary = "Permute UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        AffineMapAttr:$order_value,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let builders = [
        OpBuilder<
            (ins
                "mlir::Value":$input, "mlir::Value":$output,
                "mlir::AffineMapAttr":$order_value
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

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];
}

//
// DPUTaskOp
//

def VPUIP_DPUTaskOp :
        VPUIP_Op<
            "DPUTask",
            [
                NoSideEffect,
                AttrSizedOperandSegments
            ]
        > {
    let summary = "This object represents workload for a single DPU tile";

    // TODO: Convert `I32ArrayAttr` into `MemRefOf` for `start` and `end` inputs.
    //       Should use result of `subview` of tensors once [CVS-49600] is done.
    let arguments = (ins
        I32ArrayAttr:$start,
        I32ArrayAttr:$end,
        I32ArrayAttr:$pads_begin,
        I32ArrayAttr:$pads_end,
        VPUIP_MPEMode:$mpe_mode,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers
    );

    let assemblyFormat = [{
        attr-dict
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
    }];

    let builders = [
        OpBuilder<(ins
            "mlir::ArrayAttr":$start, "mlir::ArrayAttr":$end, "mlir::ArrayAttr":$pads_begin,
            "mlir::ArrayAttr":$pads_end, "vpux::VPUIP::MPEMode":$mpe_mode
        )>
    ];
}

//
// PPETaskOp
//

def VPUIP_PPETaskOp :
        VPUIP_Op<
            "PPETask",
            [
                NoSideEffect
            ]
        > {
    let summary = "PPE Type for NCE Task";

    let arguments = (ins
        VPUIP_PPELayerType:$ppe_layer_type
    );

    let assemblyFormat = [{
        attr-dict $ppe_layer_type
    }];
}

//
// NCEClusterTaskOp
//

def VPUIP_NCEClusterTaskOp :
        VPUIP_TaskOp<
            "NCEClusterTask",
            [
                DeclareOpInterfaceMethods<VPUIP_NCETaskOpInterface>
            ]
            # GraphRegionNoTerminator.traits
        > {

    let summary = "NCE Cluster Task Operation";

    let description = [{
        This operation defines NCE cluster task which describes single cluster of 5 DPUs. It is
        comprised of two argument categories:

        * Variants - describes the attributes for an individual DPU within the cluster.
        * Invariants - describes the collective attributes of the cluster.

        The variants argument takes on a region argument and up to 5 DPUTaskOps. The invariants
        take on a variety of argument types.

        The NCEClusterTaskOp also supports fixed PPE functions as well as generic PPE instruction
        lists. The generic PPE instruction list argument needs to be described as a region of PPE
        supported ops. Single fixed PPE functions and generic PPE instruciton list usage is
        mutually exclusive.
    }];

    let regions = (region
        SizedRegion<1>:$variants,
        VariadicRegion<SizedRegion<1>>:$ppe_tasks
    );

    let arguments = (ins
        MemRefOf<[F16, UI8, quant_QuantizedType]>:$input,
        MemRefOf<[F16, UI8, quant_QuantizedType]>:$output,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        MemRefOf<[F16, UI8, quant_QuantizedType]>:$parent_input,
        MemRefOf<[F16, UI8, quant_QuantizedType]>:$parent_output,

        VPUIP_NCETaskType:$task_type,
        OptionalAttr<VPUIP_PPELayerType>:$fixed_ppe_task,
        Optional<MemRefOf<[F16, UI8, quant_QuantizedType]>>:$filter,
        Optional<MemRefOf<[SI32]>>:$weight_table,
        OptionalAttr<I32ArrayAttr>:$kernel_padding,
        OptionalAttr<I32ArrayAttr>:$strides
    );

    let builders = [
        OpBuilder<(ins
            "mlir::Value":$input, "mlir::Value":$output, "mlir::Value":$parent_input,
            "mlir::Value":$parent_output, "vpux::VPUIP::NCETaskType":$task_type,
            "vpux::VPUIP::PPELayerTypeAttr":$fixed_ppe_task, "mlir::Value":$filter,
            "mlir::Value":$weight_table, "mlir::ArrayAttr":$kernel_padding,
            "mlir::ArrayAttr":$strides
        )>
    ];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input  `:` type($input)
                     (`,` $filter^ `:` type($filter))?
                     (`,` $weight_table^ `:` type($weight_table))? `)`
        `outputs` `(` $output `:` type($output) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `parent_input` `(` $parent_input `:` type($parent_input) `)`
        `parent_output` `(` $parent_output `:` type($parent_output) `)`
        `variants` `:` $variants
        (`PPE` `:` $ppe_tasks^)?
    }];

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];
}

#endif
