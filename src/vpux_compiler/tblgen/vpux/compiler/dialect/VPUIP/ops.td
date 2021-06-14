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

#ifndef VPUX_COMPILER_DIALECT_VPUIP_OPS
#define VPUX_COMPILER_DIALECT_VPUIP_OPS

include "vpux/compiler/core/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/VPUIP/dialect.td"
include "vpux/compiler/dialect/VPUIP/types.td"
include "vpux/compiler/dialect/VPUIP/attributes.td"
include "vpux/compiler/dialect/VPUIP/ops_interfaces.td"

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

class VPUIP_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<
            VPUIP_Dialect,
            mnemonic,
            traits
        >;

class VPUIP_TaskOp<int numResult, string mnemonic, list<OpTrait> traits = []> :
        VPUIP_Op<
            mnemonic,
            !listconcat(
                traits,
                [
                    RTLayer,
                    LayerInterface,
                    InferTypeOpInterface,
                    MemoryEffectsOpInterface,
                    AttrSizedOperandSegments,
                    DeclareOpInterfaceMethods<VPUIP_TaskOpInterface>
                ]
            )
        > {

    code commonExtraClassDeclaration = InferReturnTypesForLayer<numResult>.impl;

    let extraClassDeclaration = commonExtraClassDeclaration;
}

class VPUIP_UPATaskOp<int numResult, string mnemonic, list<OpTrait> traits = []> :
        VPUIP_TaskOp<
            numResult,
            mnemonic,
            !listconcat(
                traits,
                [
                    DeclareOpInterfaceMethods<VPUIP_UPATaskOpInterface>
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
        I32ArrayAttr:$localeIndex,

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
        >,
        OpBuilder<
            (ins "mlir::Type":$memory, "vpux::VPUIP::MemoryLocation":$locale, "ArrayRef<int64_t>":$localeIndex, "uint64_t":$dataIndex)
        >
    ];

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];

    let assemblyFormat = [{
        $locale custom<LocaleIndex>($localeIndex) `<` $dataIndex `>` attr-dict `->` type(results)
    }];

    let extraClassDeclaration = [{
        static mlir::ParseResult parseLocaleIndex(mlir::OpAsmParser& parser, mlir::ArrayAttr& localeIndex);
        static void printLocaleIndex(mlir::OpAsmPrinter& printer, vpux::VPUIP::DeclareTensorOp& op, mlir::ArrayAttr localeIndex);
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
                DeclareOpInterfaceMethods<ConstantInterface>
            ]
        > {
    let summary = "Constant TensorReference value declaration";

    let arguments = (ins
        ElementsAttr:$value,
        I32Attr:$localeIndex,
        UnitAttr:$csramCacheable
    );

    let results = (outs
        AnyStaticShapeMemRef:$output
    );

    let builders = [
        OpBuilder<
            (ins "mlir::MemRefType":$type, "mlir::ElementsAttr":$value, "uint32_t":$localeIndex)
        >
    ];

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];

    let assemblyFormat = [{
        `[` $localeIndex `]` attr-dict type($output) `=` $value
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

    let skipDefaultBuilders = 1;
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
        VPUIP_Op<"ConfigureBarrier",
            [
                MemoryEffectsOpInterface,
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<VPUIP_TaskOpInterface>
            ]
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
// EmptyOp
//

def VPUIP_EmptyOp :
        VPUIP_Op<"Empty",
            [
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<VPUIP_TaskOpInterface>
            ]
        > {
    let summary = "Empty management task";

    let arguments = (ins
        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers
    );

    let extraClassDeclaration = [{
        static vpux::VPUIP::TaskType getTaskType() {
            return vpux::VPUIP::TaskType::Controller;
        }
    }];

    let assemblyFormat = [{
        attr-dict
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
    }];
}

//
// UPADMAOp
//

def VPUIP_UPADMAOp :
        VPUIP_TaskOp<1, "UPADMA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_SameInOutDimsOrder
            ]
        > {
    let summary = "UPA DMA task";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers
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
        static vpux::VPUIP::TaskType getTaskType() {
            return vpux::VPUIP::TaskType::UPADMA;
        }
    }] # InferReturnTypesForLayer<1>.impl;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];
}

//
// NNDMAOp
//

def VPUIP_NNDMAOp :
        VPUIP_TaskOp<1, "NNDMA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_SameInOutDimsOrder
            ]
        > {
    let summary = "NN DMA task";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        UnitAttr:$compression,
        DefaultValuedAttr<I32Attr, "0">:$port
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
        static vpux::VPUIP::TaskType getTaskType() {
            return vpux::VPUIP::TaskType::NNDMA;
        }
    }] # InferReturnTypesForLayer<1>.impl;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];
}

//
// ConvertUPAOp
//

def VPUIP_ConvertUPAOp :
        VPUIP_UPATaskOp<1, "ConvertUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameInOutDimsOrder,
                DeclareOpInterfaceMethods<ConvertLayerInterface>
            ]
        > {
    let summary = "Convert UPA SHAVE kernel";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output_buff,

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

    let results = (outs
        AnyMemRef:$output
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
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];
}

//
// SoftMaxUPAOp
//

def VPUIP_SoftMaxUPAOp :
        VPUIP_UPATaskOp<1, "SoftMaxUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_SameInOutDimsOrder,
                DeclareOpInterfaceMethods<SoftMaxLayerInterface>
            ]
        > {
    let summary = "SoftMax UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        I32Attr:$axisInd,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let results = (outs
        F16MemRef:$output
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
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];
}

//
// PoolingUPAOp
//

def VPUIP_PoolingUPAOp :
        VPUIP_UPATaskOp<1, "PoolingUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameElementType,
                VPUIP_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
            ]
        > {
    let summary = "MAX and AVG Pooling UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

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

    let results = (outs
        F16MemRef:$output
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
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];
}

//
// ConvolutionUPAOp
//

def VPUIP_ConvolutionUPAOp :
        VPUIP_UPATaskOp<1, "ConvolutionUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameElementType,
                DeclareOpInterfaceMethods<ConvolutionLayerInterface>
            ]
        > {
    let summary = "Convolution UPA SHAVE kernel (reference implementation)";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$filter,
        Optional<F16MemRef>:$bias,
        F16MemRef:$output_buff,

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

    let results = (outs
        F16MemRef:$output
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
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        static mlir::LogicalResult isSupportedLayout(mlir::Operation* op, vpux::DataOrderInfo& info);
    }] # commonExtraClassDeclaration;

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];
}

//
// ReLUUPAOp
//

def VPUIP_ReLUUPAOp :
        VPUIP_UPATaskOp<1, "ReLUUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_SameInOutDimsOrder,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "ReLU UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let results = (outs
        F16MemRef:$output
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
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];
}

//
// SigmoidOp
//

def VPUIP_SigmoidUPAOp :
        VPUIP_UPATaskOp<1, "SigmoidUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_SameInOutDimsOrder,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Sigmoid UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let results = (outs
        F16MemRef:$output
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
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];
}

//
// HSwishUPAOp
//

def VPUIP_HSwishUPAOp :
        VPUIP_UPATaskOp<1, "HSwishUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_SameInOutDimsOrder,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "HSwish UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let results = (outs
        F16MemRef:$output
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
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];
}

//
// ClampUPAOp
//

def VPUIP_ClampUPAOp :
        VPUIP_UPATaskOp<1, "ClampUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_SameInOutDimsOrder,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Clamp UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        F32Attr:$min,
        F32Attr:$max,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let results = (outs
        F16MemRef:$output
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
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];
}

//
// EluUPAOp
//

def VPUIP_EluUPAOp :
        VPUIP_UPATaskOp<1, "EluUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_SameInOutDimsOrder,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Elu UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        F32Attr:$x,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let results = (outs
        F16MemRef:$output
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
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];
}

//
// TanhUPAOp
//

def VPUIP_TanhUPAOp :
        VPUIP_UPATaskOp<1, "TanhUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_SameInOutDimsOrder,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Tanh UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let results = (outs
        F16MemRef:$output
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
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];
}

//
// ExpUPAOp
//

def VPUIP_ExpUPAOp :
        VPUIP_UPATaskOp<1, "ExpUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_SameDimsOrder,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Exp UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let results = (outs
        F16MemRef:$output
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
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];
}

//
// FakeQuantizeUPAOp
//

def VPUIP_FakeQuantizeUPAOp :
        VPUIP_UPATaskOp<1, "FakeQuantizeUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "FakeQuantize UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

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

    let results = (outs
        F16MemRef:$output
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
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];
}

//
// QuantCastUPAOp
//

def VPUIP_QuantCastUPAOp :
        VPUIP_UPATaskOp<1, "QuantCastUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "FakeQuantize UPA SHAVE kernel";

    let arguments = (ins
        MemRefOf<[F16, quant_QuantizedType]>:$input,
        MemRefOf<[F16, quant_QuantizedType]>:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let results = (outs
        MemRefOf<[F16, quant_QuantizedType]>:$output
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
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        static mlir::LogicalResult isSupportedLayout(mlir::Operation* op, vpux::DataOrderInfo& info);
    }] # commonExtraClassDeclaration;
}

//
// PReluUPAOp
//

def VPUIP_PReluUPAOp :
        VPUIP_UPATaskOp<1, "PReluUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameInOutDimsOrder
            ]
        > {
    let summary = "PRelu UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$negative_slope,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let results = (outs
        F16MemRef:$output
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
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];
}

//
// LeakyReluUPAOp
//

def VPUIP_LeakyReluUPAOp :
        VPUIP_UPATaskOp<1,  "LeakyReluUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameInOutDimsOrder
            ]
        > {
    let summary = "LeakyRelu UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        F32Attr:$negative_slope,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let results = (outs
        F16MemRef:$output
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
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];
}

//
// EltwiseUPAOp
//

def VPUIP_EltwiseUPAOp :
        VPUIP_UPATaskOp<1, "EltwiseUPA",
            [
                ViewLikeOpInterface,
                VPUIP_AnyDimsOrder
            ]
        > {
    let summary = "Eltwise UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input1,
        F16MemRef:$input2,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        VPUIP_EltwiseLayerType:$type,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let results = (outs
        F16MemRef:$output
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
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];
}

//
// DetectionOutput
//

def VPUIP_DetectionOutputUPAOp :
        VPUIP_UPATaskOp<1, "DetectionOutputUPA",
            [
                ViewLikeOpInterface,
                VPUIP_AnyDimsOrder
            ]
        > {
    let summary = "DetectionOutput UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$in_box_logits,
        F16MemRef:$in_class_preds,
        F16MemRef:$in_proposals,
        Optional<F16MemRef>:$in_additional_preds,
        Optional<F16MemRef>:$in_additional_proposals,

        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        IE_DetectionOutputAttrs:$attr,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let results = (outs
        F16MemRef:$output
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
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];
}

//
// SwishUPAOp
//

def VPUIP_SwishUPAOp :
        VPUIP_UPATaskOp<1, "SwishUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameInOutDimsOrder
            ]
        > {
    let summary = "Swish UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        F32Attr:$beta_value,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let results = (outs
        F16MemRef:$output
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
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];
}

//
// GRNUPAOp
//

def VPUIP_GRNUPAOp :
        VPUIP_UPATaskOp<1, "GRNUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameInOutDimsOrder_NCHW_NHWC
            ]
        > {
    let summary = "GRN UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        F32Attr:$bias,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let results = (outs
        F16MemRef:$output
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
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];
}

//
// ScaleShiftUPAOp
//

def VPUIP_ScaleShiftUPAOp :
        VPUIP_UPATaskOp<1, "ScaleShiftUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameElementType,
                VPUIP_SameDimsOrder,
		VPUIP_Legacy4D
            ]
        > {
    let summary = "ScaleShift UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        Optional<F16MemRef>:$weights,
        Optional<F16MemRef>:$biases,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let results = (outs
        F16MemRef:$output
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
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
    }];
}

//
// PerAxisTileUPAOp
//

def VPUIP_PerAxisTileUPAOp :
        VPUIP_UPATaskOp<1, "PerAxisTileUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameInOutDimsOrder
            ]
        > {
    let summary = "Tile for per axis case UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        I32Attr:$axis,
        I32Attr:$tiles,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let results = (outs
        F16MemRef:$output
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
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];
}

//
// NegativeUPAOp
//

def VPUIP_NegativeUPAOp :
        VPUIP_UPATaskOp<1, "NegativeUPA",
            [
                ViewLikeOpInterface,
                VPUIP_AnyDimsOrder
            ]
        > {
    let summary = "Negative UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let results = (outs
        F16MemRef:$output
    );

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output)
        >
    ];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];
}

//
// ROIPoolingUPAOp
//

def VPUIP_ROIPoolingUPAOp :
        VPUIP_UPATaskOp<1, "ROIPoolingUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
            ]
        > {
    let summary = "ROIPooling UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$coords,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        I32ArrayAttr:$output_size,
        F32Attr:$spatial_scale,
        IE_ROIPoolingMethod:$method,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let results = (outs
        F16MemRef:$output
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
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];
}

//
// PermuteUPAOp
//

def VPUIP_PermuteUPAOp :
        VPUIP_UPATaskOp<1, "PermuteUPA",
            [
                ViewLikeOpInterface,
                VPUIP_AnyDimsOrder
            ]
        > {
    let summary = "Permute UPA SHAVE kernel";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        OptionalAttr<AffineMapAttr>:$order_value,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let results = (outs
        AnyMemRef:$output
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
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
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
            "PPETask"
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
        VPUIP_TaskOp<1, "NCEClusterTask",
            [
                ViewLikeOpInterface,
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
        MemRefOf<[UI8, F16, BF16, quant_QuantizedType]>:$input,
        Optional<MemRefOf<[UI8, F16, BF16, quant_QuantizedType]>>:$filter,
        Optional<MemRefOf<[SI32]>>:$weight_table,
        Optional<MemRefOf<[UI8]>>:$activation_window,
        MemRefOf<[UI8, F16, BF16, quant_QuantizedType]>:$parent_input,
        MemRefOf<[UI8, F16, F32, BF16, quant_QuantizedType]>:$parent_output,

        MemRefOf<[UI8, F16, F32, BF16, quant_QuantizedType]>:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        VPUIP_NCETaskType:$task_type,
        OptionalAttr<VPUIP_PPELayerType>:$fixed_ppe_task,
        OptionalAttr<I32ArrayAttr>:$kernel_padding,
        OptionalAttr<I32ArrayAttr>:$strides,
        OptionalAttr<I32ArrayAttr>:$kernel_size,
        OptionalAttr<I32Attr>:$activation_window_channel_length

    );

    let results = (outs
        MemRefOf<[UI8, F16, F32, BF16, quant_QuantizedType]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "mlir::Value":$input, "mlir::Value":$filter, "mlir::Value":$weight_table,
            "mlir::Value":$activation_window, "mlir::Value":$parent_input, "mlir::Value":$parent_output,
            "mlir::Value":$output,
            "vpux::VPUIP::NCETaskType":$task_type, "vpux::VPUIP::PPELayerTypeAttr":$fixed_ppe_task,
            "mlir::ArrayAttr":$kernel_padding, "mlir::ArrayAttr":$strides, "mlir::ArrayAttr":$kernel_size,
            "mlir::IntegerAttr":$activation_window_channel_length
        )>
    ];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input  `:` type($input)
                     (`,` $filter^ `:` type($filter))?
                     (`,` $weight_table^ `:` type($weight_table))? `)`
                     (`,` $activation_window^ `:` type($activation_window))? `)`
        `parent_input` `(` $parent_input `:` type($parent_input) `)`
        `parent_output` `(` $parent_output `:` type($parent_output) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `variants` `:` $variants
        (`PPE` `:` $ppe_tasks^)?
        `->` type(results)
    }];

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];

    let extraClassDeclaration = [{
        vpux::VPUIP::DPUTaskOp addDPUTask(mlir::OpBuilder& builder, mlir::ArrayAttr start, mlir::ArrayAttr end,
                                          mlir::ArrayAttr padsBegin, mlir::ArrayAttr padsEnd,
                                          VPUIP::MPEMode mpeMode);

        static mlir::LogicalResult isSupportedLayout(mlir::Operation* op, vpux::DataOrderInfo& info);
    }]#InferReturnTypesForLayer<1>.impl;
}

//
// CTCGreedyDecoderUPAOp
//

def VPUIP_CTCGreedyDecoderUPAOp :
        VPUIP_UPATaskOp<1, "CTCGreedyDecoderUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "CTCGreedyDecoder UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$sequenceLengths,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        UnitAttr:$mergeRepeated,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let results = (outs
        F16MemRef:$output
    );

    let builders = [
        OpBuilder<
            (ins
                "mlir::Value":$input, "mlir::Value":$sequenceLengths,
                "mlir::Value":$output,
                "mlir::UnitAttr":$mergeRepeated
            )
        >
    ];

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $sequenceLengths `:` type($sequenceLengths) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        static mlir::LogicalResult isSupportedLayout(mlir::Operation* op, vpux::DataOrderInfo& info);
    }] # commonExtraClassDeclaration;
}

//
// CTCGreedyDecoderSeqLenUPAOp
//

def VPUIP_CTCGreedyDecoderSeqLenUPAOp :
        VPUIP_UPATaskOp<2, "CTCGreedyDecoderSeqLenUPA",
            [
                MultiViewOpInterface
            ]
        > {
    let summary = "CTCGreedyDecoderSeqLen UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        MemRefOf<[SI32]>:$sequenceLength,
        Optional<MemRefOf<[SI32]>>:$blankIndex,
        MemRefOf<[SI32]>:$output_buff,
        MemRefOf<[SI32]>:$outputLength_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        UnitAttr:$mergeRepeated,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let results = (outs
        MemRefOf<[SI32]>:$output,
        MemRefOf<[SI32]>:$outputLength
    );

    let builders = [
        OpBuilder<
            (ins
                "mlir::Value":$input, "mlir::Value":$sequenceLength, "mlir::Value":$blankIndex,
                "mlir::Value":$output, "mlir::Value":$outputLength,
                "mlir::UnitAttr":$mergeRepeated
            )
        >
    ];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $sequenceLength `:` type($sequenceLength) `,` ($blankIndex^ `:` type($blankIndex))? `)`
        `outputs` `(` $output_buff `:` type($output_buff) `,` $outputLength_buff `:` type($outputLength_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        static mlir::LogicalResult isSupportedLayout(mlir::Operation* op, vpux::DataOrderInfo& info);
    }] # commonExtraClassDeclaration;
}

//
// PadOp
//

def VPUIP_PadUPAOp :
        VPUIP_UPATaskOp<1,"Pad",
            [
                ViewLikeOpInterface,
                VPUIP_SameInOutDimsOrder
            ]
        > {
    let summary = "Pad UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        I32ArrayAttr:$pads_begin,
        I32ArrayAttr:$pads_end,
        OptionalAttr<F32Attr>:$pad_value,
        IE_PadMode:$mode,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let results = (outs
        F16MemRef:$output
    );

    let builders = [
        OpBuilder<
            (ins
                "mlir::Value":$input,
                "mlir::Value":$output,
                "mlir::ArrayAttr":$pads_begin, "mlir::ArrayAttr":$pads_end, "mlir::FloatAttr":$pad_value,
                "IE::PadModeAttr":$mode
            )
        >
    ];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];
}

//
// EdslUPAOp
//

def VPUIP_EdslUPAOp :
        VPUIP_Op<"EdslUPA",
            [
                RTLayer,
                LayerInterface,
                MemoryEffectsOpInterface,
                AttrSizedOperandSegments,
                ViewLikeOpInterface,
                DeclareOpInterfaceMethods<VPUIP_TaskOpInterface>,
                DeclareOpInterfaceMethods<VPUIP_UPATaskOpInterface>,
            ]
        > {
    let summary = "eDSL UPA SHAVE kernel";

    let arguments = (ins
        Variadic<AnyMemRef>:$inputs,
        Variadic<AnyMemRef>:$outputs,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        SymbolRefAttr:$kernel,
        I64ArrayAttr:$outers,
        I64ArrayAttr:$middles,
        ArrayAttr:$inits,
        VPUIP_EdslDMADescArrayAttr:$transfers,

        OptionalAttr<I32Attr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let results = (outs Variadic<AnyMemRef>:$results);

    let builders = [
        OpBuilder<
            (ins
                "mlir::ValueRange":$inputs,
                "mlir::ValueRange":$outputs,
                "::mlir::SymbolRefAttr":$kernel,
                "::mlir::ArrayAttr":$outers,
                "::mlir::ArrayAttr":$middles,
                "::mlir::ArrayAttr":$inits,
                "::mlir::ArrayAttr":$transfers
            )
        >
    ];

    let assemblyFormat = [{
        attr-dict
        (`inputs` `(` $inputs^ `:` type($inputs) `)`)?
        (`outputs` `(` $outputs^ `:` type($outputs) `)`)?
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];
}

#endif
