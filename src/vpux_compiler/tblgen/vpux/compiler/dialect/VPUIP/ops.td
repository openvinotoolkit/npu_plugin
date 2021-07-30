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

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/core/ops_interfaces.td"
include "vpux/compiler/dialect/const/attributes.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/VPUIP/attributes.td"
include "vpux/compiler/dialect/VPUIP/dialect.td"
include "vpux/compiler/dialect/VPUIP/ops_interfaces.td"
include "vpux/compiler/dialect/VPUIP/types.td"

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
                    AttrSizedOperandSegments,
                    InferTypeOpInterface,
                    MemoryEffectsOpInterface,
                    DeclareOpInterfaceMethods<RTLayerInterface>,
                    DeclareOpInterfaceMethods<VPUIP_TaskOpInterface>
                ]
            )
        > {
    code commonExtraClassDeclaration = InferReturnTypesForRTLayer<numResult>.impl;

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
        I64ArrayAttr:$localeIndex,

        IntAttr:$dataIndex,
        OptionalAttr<IntAttr>:$sparsityIndex,
        OptionalAttr<IntAttr>:$storageElementIndex,
        OptionalAttr<IntAttr>:$storageElementSize,

        OptionalAttr<IntAttr>:$leadingOffset,
        OptionalAttr<IntAttr>:$trailingOffset
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
        IntAttr:$id,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers
    );

    let results = (outs
        VPUIP_Barrier:$barrier
    );

    let builders = [
        OpBuilder<(ins "int64_t":$id)>
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
                SameShape,
                SameElementType,
                SameInOutDimsOrder
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
    }] # InferReturnTypesForRTLayer<1>.impl;

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
                SameShape,
                SameElementType,
                SameInOutDimsOrder
            ]
        > {
    let summary = "NN DMA task";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        UnitAttr:$compression,
        DefaultValuedAttr<IntAttr, "0">:$port
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
    }] # InferReturnTypesForRTLayer<1>.impl;

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
                SameShape,
                SameInOutDimsOrder,
                DeclareOpInterfaceMethods<ConvertLayerInterface>
            ]
        > {
    let summary = "Convert UPA SHAVE kernel";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        OptionalAttr<F64Attr>:$scale,
        OptionalAttr<F64Attr>:$bias,
        UnitAttr:$fromDetectionOutput,
        UnitAttr:$haveBatch,
        OptionalAttr<IntAttr>:$batchID,

        OptionalAttr<IntAttr>:$maxShaves,
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
                SameShape,
                SameElementType,
                SameInOutDimsOrder,
                DeclareOpInterfaceMethods<SoftMaxLayerInterface>
            ]
        > {
    let summary = "SoftMax UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        IntAttr:$axisInd,

        OptionalAttr<IntAttr>:$maxShaves,
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
                SameElementType,
                SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
            ]
        > {
    let summary = "MAX and AVG Pooling UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        VPUIP_PoolLayerType:$type,
        I64ArrayAttr:$kernel,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$padsBegin,
        I64ArrayAttr:$padsEnd,
        UnitAttr:$excludePad,

        OptionalAttr<IntAttr>:$maxShaves,
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
                SameElementType,
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

        I64ArrayAttr:$strides,
        I64ArrayAttr:$dilations,
        I64ArrayAttr:$padsBegin,
        I64ArrayAttr:$padsEnd,
        IntAttr:$groups,

        OptionalAttr<IntAttr>:$maxShaves,
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
        static bool isSupportedLayout(mlir::Operation* op, vpux::DataOrderInfo& info);
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
                SameShape,
                SameElementType,
                SameInOutDimsOrder,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "ReLU UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        OptionalAttr<IntAttr>:$maxShaves,
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
                SameShape,
                SameElementType,
                SameInOutDimsOrder,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Sigmoid UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        OptionalAttr<IntAttr>:$maxShaves,
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
                SameShape,
                SameElementType,
                SameInOutDimsOrder,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "HSwish UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        OptionalAttr<IntAttr>:$maxShaves,
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
// MishUPAOp
//

def VPUIP_MishUPAOp :
        VPUIP_UPATaskOp<1, "MishUPA",
            [
                ViewLikeOpInterface,
                SameShape,
                SameElementType,
                SameInOutDimsOrder,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Mish UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        OptionalAttr<IntAttr>:$maxShaves,
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
                SameShape,
                SameElementType,
                SameInOutDimsOrder,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Clamp UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        F64Attr:$min,
        F64Attr:$max,

        OptionalAttr<IntAttr>:$maxShaves,
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
                SameShape,
                SameElementType,
                SameInOutDimsOrder,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Elu UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        F64Attr:$x,

        OptionalAttr<IntAttr>:$maxShaves,
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
                SameShape,
                SameElementType,
                SameInOutDimsOrder,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Tanh UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        OptionalAttr<IntAttr>:$maxShaves,
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
                SameShape,
                SameElementType,
                SameDimsOrder,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Exp UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        OptionalAttr<IntAttr>:$maxShaves,
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
                SameShape,
                SameElementType,
                SameInOutDimsOrder_CHW_HWC_NCHW_NHWC,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "FakeQuantize UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        IntAttr:$levels,
        Const_ContentAttr:$input_low,
        Const_ContentAttr:$input_high,
        Const_ContentAttr:$output_low,
        Const_ContentAttr:$output_high,

        OptionalAttr<IntAttr>:$maxShaves,
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
                "vpux::Const::ContentAttr":$input_low, "vpux::Const::ContentAttr":$input_high,
                "vpux::Const::ContentAttr":$output_low, "vpux::Const::ContentAttr":$output_high
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
                SameShape,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "FakeQuantize UPA SHAVE kernel";

    let arguments = (ins
        MemRefOf<[F16, quant_QuantizedType]>:$input,
        MemRefOf<[F16, quant_QuantizedType]>:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        OptionalAttr<IntAttr>:$maxShaves,
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
        static bool isSupportedLayout(mlir::Operation* op, vpux::DataOrderInfo& info);
    }] # commonExtraClassDeclaration;
}

//
// PReluUPAOp
//

def VPUIP_PReluUPAOp :
        VPUIP_UPATaskOp<1, "PReluUPA",
            [
                ViewLikeOpInterface,
                SameInOutDimsOrder
            ]
        > {
    let summary = "PRelu UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$negative_slope,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        OptionalAttr<IntAttr>:$maxShaves,
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
                SameInOutDimsOrder
            ]
        > {
    let summary = "LeakyRelu UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        F64Attr:$negative_slope,

        OptionalAttr<IntAttr>:$maxShaves,
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
                AnyDimsOrder
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

        OptionalAttr<IntAttr>:$maxShaves,
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
                AnyDimsOrder
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

        OptionalAttr<IntAttr>:$maxShaves,
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
                SameInOutDimsOrder
            ]
        > {
    let summary = "Swish UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        F64Attr:$beta_value,

        OptionalAttr<IntAttr>:$maxShaves,
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
                SameInOutDimsOrder_NCHW_NHWC
            ]
        > {
    let summary = "GRN UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        F64Attr:$bias,

        OptionalAttr<IntAttr>:$maxShaves,
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
                SameElementType,
                SameDimsOrder,
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

        OptionalAttr<IntAttr>:$maxShaves,
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
                SameInOutDimsOrder
            ]
        > {
    let summary = "Tile for per axis case UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        IntAttr:$axis,
        IntAttr:$tiles,

        OptionalAttr<IntAttr>:$maxShaves,
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
                AnyDimsOrder
            ]
        > {
    let summary = "Negative UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        OptionalAttr<IntAttr>:$maxShaves,
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
// ProposalUPAOp
//

def VPUIP_ProposalUPAOp :
        VPUIP_UPATaskOp<1, "ProposalUPA",
            [
                ViewLikeOpInterface,
                AnyDimsOrder
            ]
        > {
    let summary = "Proposal UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$class_probs,
        F16MemRef:$bbox_deltas,
        F16MemRef:$image_shape,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        IE_ProposalAttrs:$proposal_attrs,

        OptionalAttr<IntAttr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let results = (outs
        F16MemRef:$output
    );

    let builders = [
        OpBuilder<
            (ins
                "mlir::Value":$class_probs, "mlir::Value":$bbox_deltas,"mlir::Value":$image_shape,
                "mlir::Value":$output, "vpux::IE::ProposalAttr":$attr
            )
        >
    ];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $class_probs `:` type($class_probs) `,` $bbox_deltas `:` type($bbox_deltas) `,` $image_shape `:` type($image_shape) `)`
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
                SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
            ]
        > {
    let summary = "ROIPooling UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$coords,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        I64ArrayAttr:$output_size,
        F64Attr:$spatial_scale,
        IE_ROIPoolingMethod:$method,

        OptionalAttr<IntAttr>:$maxShaves,
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
                AnyDimsOrder
            ]
        > {
    let summary = "Permute UPA SHAVE kernel";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        OptionalAttr<AffineMapAttr>:$order_value,

        OptionalAttr<IntAttr>:$maxShaves,
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
// WeightsTableOp
//

def VPUIP_WeightsTableOp :
        VPUIP_Op<"WeightsTableOp",
            [
                AttrSizedOperandSegments,
            ]
        > {
    let summary = "Intermediate task for creating weights table based on the addresses of CMX buffers";

    let arguments = (ins
        Optional<MemRefOf<[F16, quant_QuantizedType]>>:$op_input,
        Optional<MemRefOf<[F16, quant_QuantizedType]>>:$op_output,
        Optional<MemRefOf<[F16, quant_QuantizedType]>>:$weights,
        Optional<MemRefOf<[F16, F32]>>:$bias,
        Optional<MemRefOf<[UI8]>>:$activation_window
    );

    let results = (outs
        MemRefOf<[SI32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        (`op_input` `(` $op_input^  `:` type($op_input) `)`)?
        (`op_output` `(` $op_output^  `:` type($op_output) `)`)?
        (`weights` `(` $weights^  `:` type($weights) `)`)?
        (`bias` `(` $bias^  `:` type($bias) `)`)?
        (`activation_window` `(` $activation_window^  `:` type($activation_window) `)`)?
        `->` type(results)
    }];
}

//
// DPUTaskOp
//

def VPUIP_DPUTaskOp :
        VPUIP_Op<
            "DPUTask",
            [
                HasParent<"vpux::VPUIP::NCEClusterTaskOp">
            ]
        > {
    let summary = "This object represents workload for a single DPU tile";

    let arguments = (ins
        I64ArrayAttr:$start,
        I64ArrayAttr:$end,
        VPUIP_PaddingAttr:$pad,
        VPUIP_MPEMode:$mpe_mode
    );

    let assemblyFormat = [{
        attr-dict
    }];

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];
}

//
// PPETaskOp
//

def VPUIP_PPETaskOp :
        VPUIP_Op<
            "PPETask",
            [
                HasParent<"vpux::VPUIP::NCEClusterTaskOp">
            ]
        > {
    let summary = "PPE Type for NCE Task";

    let arguments = (ins
        VPUIP_PPELayerType:$ppe_layer_type
    );

    let assemblyFormat = [{
        $ppe_layer_type attr-dict
    }];
}

//
// NCEClusterTaskOp
//

def VPUIP_NCEClusterTaskOp :
        VPUIP_TaskOp<1, "NCEClusterTask",
            [
                ViewLikeOpInterface,
                NoRegionArguments
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

    let arguments = (ins
        MemRefOf<[F16, quant_QuantizedType]>:$input,
        Optional<MemRefOf<[F16, quant_QuantizedType]>>:$weights,
        Optional<MemRefOf<[SI32]>>:$weight_table,
        Optional<MemRefOf<[UI8]>>:$activation_window,

        AnyMemRef:$parent_input,
        AnyMemRef:$parent_output,

        MemRefOf<[F16, quant_QuantizedType]>:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        VPUIP_NCETaskType:$task_type,
        OptionalAttr<I64ArrayAttr>:$kernel_size,
        OptionalAttr<I64ArrayAttr>:$kernel_strides,
        OptionalAttr<I64ArrayAttr>:$kernel_padding,
        OptionalAttr<IntAttr>:$activation_window_channel_length
    );

    let results = (outs
        MemRefOf<[F16, quant_QuantizedType]>:$output
    );

    let regions = (region
        SizedRegion<1>:$variants,
        SizedRegion<1>:$ppe
    );

    let builders = [
        OpBuilder<(ins
            "mlir::Value":$input, "mlir::Value":$weights, "mlir::Value":$weight_table, "mlir::Value":$activation_window,
            "mlir::Value":$parent_input, "mlir::Value":$parent_output,
            "mlir::Value":$output_buff,
            "vpux::VPUIP::NCETaskType":$task_type,
            "mlir::ArrayAttr":$kernel_size, "mlir::ArrayAttr":$kernel_strides, "mlir::ArrayAttr":$kernel_padding,
            "mlir::IntegerAttr":$activation_window_channel_length
        )>
    ];

    let assemblyFormat = [{
        attr-dict
        `input` `(` $input  `:` type($input) `)`
        (`weights` `(` $weights^  `:` type($weights) `)`)?
        (`weight_table` `(` $weight_table^  `:` type($weight_table) `)`)?
        (`activation_window` `(` $activation_window^  `:` type($activation_window) `)`)?
        `parent_input` `(` $parent_input `:` type($parent_input) `)`
        `parent_output` `(` $parent_output `:` type($parent_output) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
        `variants` `:` $variants
        `PPE` `:` $ppe
    }];

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];

    let extraClassDeclaration = [{
        vpux::VPUIP::DPUTaskOp addDPUTask(mlir::OpBuilder& builder,
                                          mlir::ArrayAttr start, mlir::ArrayAttr end,
                                          VPUIP::PaddingAttr pad, VPUIP::MPEMode mpeMode);

        vpux::VPUIP::PPETaskOp addPPETask(mlir::OpBuilder& builder,
                                          vpux::VPUIP::PPELayerType ppe_layer_type);

        static bool isSupportedLayout(mlir::Operation* op, vpux::DataOrderInfo& info);

        static vpux::VPUIP::TaskType getTaskType() {
            return vpux::VPUIP::TaskType::NCE2;
        }
    }] # InferReturnTypesForRTLayer<1>.impl;
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

        OptionalAttr<IntAttr>:$maxShaves,
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
        static bool isSupportedLayout(mlir::Operation* op, vpux::DataOrderInfo& info);
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

        OptionalAttr<IntAttr>:$maxShaves,
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
        static bool isSupportedLayout(mlir::Operation* op, vpux::DataOrderInfo& info);
    }] # commonExtraClassDeclaration;
}

//
// PadOp
//

def VPUIP_PadUPAOp :
        VPUIP_UPATaskOp<1,"Pad",
            [
                ViewLikeOpInterface,
                SameInOutDimsOrder
            ]
        > {
    let summary = "Pad UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        OptionalAttr<F64Attr>:$pad_value,
        IE_PadMode:$mode,

        OptionalAttr<IntAttr>:$maxShaves,
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
// LSTMCellUPAOp
//

def VPUIP_LSTMCellUPAOp :
        VPUIP_UPATaskOp<2, "LSTMCellUPA",
            [
                MultiViewOpInterface,
                SameInOutDimsOrder
            ]
        > {
    let summary = "LSTMCell UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$inputData,
        F16MemRef:$initialHiddenState,
        F16MemRef:$initialCellState,
        F16MemRef:$weights,
        F16MemRef:$biases,
        F16MemRef:$outputHiddenState_buff,
        F16MemRef:$outputCellState_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        OptionalAttr<IntAttr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let results = (outs
        F16MemRef:$outputHiddenState,
        F16MemRef:$outputCellState
    );

    let builders = [
        OpBuilder<
            (ins
                "mlir::Value":$inputData, "mlir::Value":$initialHiddenState,
                "mlir::Value":$initialCellState, "mlir::Value":$weights, "mlir::Value":$biases,
                "mlir::Value":$outputHiddenState, "mlir::Value":$outputCellState
            )
        >
    ];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $inputData `:` type($inputData) `,` $initialHiddenState `:` type($initialHiddenState)
        `,` $initialCellState `:` type($initialCellState) `,` $weights `:` type($weights) `,` $biases `:` type($biases) `)`
        `outputs` `(` $outputHiddenState_buff `:` type($outputHiddenState_buff)
        `,` $outputCellState_buff `:` type($outputCellState_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];
}

//
// InterpolateOp
//

def VPUIP_InterpolateUPAOp :
        VPUIP_UPATaskOp<1,"Interpolate",
            [
                ViewLikeOpInterface,
                SameInOutDimsOrder_NCHW_NHWC
            ]
        > {
    let summary = "Interpolate UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,

        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        IE_InterpolateMode:$mode,
        IE_InterpolateCoordMode:$coord_mode,
        IE_InterpolateNearestMode:$nearest_mode,
        UnitAttr:$antialias,

        OptionalAttr<IntAttr>:$maxShaves,
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
                "IE::InterpolateModeAttr":$mode,
                "IE::InterpolateCoordModeAttr":$coord_mode,
                "IE::InterpolateNearestModeAttr":$nearest_mode,
                "mlir::UnitAttr":$antialias
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
// StridedSliceUPAOp
//

def VPUIP_StridedSliceUPAOp :
     VPUIP_UPATaskOp<1, "StridedSlice",
         [
             ViewLikeOpInterface,
             AnyDimsOrder
         ]
     > {
    let summary = "StridedSlice UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,

        I64ArrayAttr:$begins,
        I64ArrayAttr:$ends,
        I64ArrayAttr:$strides,

        OptionalAttr<IntAttr>:$maxShaves,
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
                "mlir::ArrayAttr":$begins, "mlir::ArrayAttr":$ends, "mlir::ArrayAttr":$strides
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
// FullyConnectedUPAOp
//

def VPUIP_FullyConnectedUPAOp :
        VPUIP_UPATaskOp<1, "FullyConnectedUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "FullyConnected UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$weights,
        Optional<F16MemRef>:$bias,
        F16MemRef:$output_buff,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,
        OptionalAttr<IntAttr>:$maxShaves,
        UnitAttr:$isTrailingSWLayer
    );

    let results = (outs
        F16MemRef:$output
    );

    let builders = [
        OpBuilder<
            (ins
                "mlir::Value":$input, "mlir::Value":$weights, "mlir::Value":$bias, "mlir::Value":$output
            )
        >
    ];

    let extraClassDeclaration = [{
        static bool isSupportedLayout(mlir::Operation* op, vpux::DataOrderInfo& info);
    }] # commonExtraClassDeclaration;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $weights `:` type($weights) (`,` $bias^ `:` type($bias))? `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];
}

//
// RegionYoloOp
//

def VPUIP_RegionYoloUPAOp :
        VPUIP_UPATaskOp<1, "RegionYoloUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "RegionYolo UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        IntAttr:$coords,
        IntAttr:$classes,
        IntAttr:$regions,
        OptionalAttr<BoolAttr>:$do_softmax,
        OptionalAttr<I64ArrayAttr>:$mask,

        Variadic<VPUIP_Barrier>:$waitBarriers,
        Variadic<VPUIP_Barrier>:$updateBarriers,
        OptionalAttr<IntAttr>:$maxShaves,
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
                "mlir::IntegerAttr":$coord,
                "mlir::IntegerAttr":$classes,
                "mlir::IntegerAttr":$regions,
                "mlir::BoolAttr":$do_softmax,
                "mlir::ArrayAttr":$mask
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

    let extraClassDeclaration = [{
        static bool isSupportedLayout(mlir::Operation* op, vpux::DataOrderInfo& info);
    }] # commonExtraClassDeclaration;
}

#endif
