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
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/IERT/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPUIP/attributes.td"
include "vpux/compiler/dialect/VPUIP/dialect.td"
include "vpux/compiler/dialect/VPUIP/ops_interfaces.td"
include "vpux/compiler/dialect/VPUIP/types.td"
include "vpux/compiler/dialect/VPURT/types.td"

include "mlir/Dialect/Quant/QuantOpsBase.td"
include "mlir/Interfaces/CopyOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"

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
                    InferTypeOpInterface,
                    DeclareOpInterfaceMethods<IERT_LayerOpInterface>,
                    IERT_AsyncLayerOpInterface,
                    DeclareOpInterfaceMethods<VPUIP_TaskOpInterface>,
                    MemoryEffectsOpInterface
                ]
            )
        > {
    code commonExtraClassDeclaration = IERT_InferLayerReturnTypes<numResult>.impl;

    let extraClassDeclaration = commonExtraClassDeclaration;
}

class VPUIP_UPATaskOp<int numResult, string mnemonic, list<OpTrait> traits = []> :
        VPUIP_TaskOp<
            numResult,
            mnemonic,
            !listconcat(
                traits,
                [
                    VPUIP_UPATask
                ]
            )
        >;

//
// NNDMAOp
//

def VPUIP_NNDMAOp :
        VPUIP_TaskOp<1, "NNDMA",
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
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$output_buff,

        DefaultValuedAttr<IntAttr, "0">:$port,
        UnitAttr:$is_out_of_order,
        UnitAttr:$is_critical
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$output
    );

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff)
        >
    ];

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DMA_NN;
        }

        vpux::DotNodeColor getNodeColor() {
            const auto memKind = getType().cast<vpux::NDTypeInterface>().getMemoryKind();
            return memKind == vpux::VPU::MemoryKind::DDR ? vpux::DotNodeColor::RED : vpux::DotNodeColor::GREEN;
        }
    }] # IERT_InferLayerReturnTypes<1>.impl;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];
}

//
// CompressedDMAOp
//

def VPUIP_CompressedDMAOp :
        VPUIP_TaskOp<1, "CompressedDMAOp",
            [
                ViewLikeOpInterface,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder,
                DotInterface
            ]
        > {
    let summary = "NN DMA task with enabled compression. Omits IERT_SameShape trait.";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output_buff,

        DefaultValuedAttr<IntAttr, "0">:$port,
        UnitAttr:$is_out_of_order,
        UnitAttr:$is_critical
    );

    let results = (outs
        AnyMemRef:$output
    );

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff)
        >
    ];

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DMA_NN;
        }

        vpux::DotNodeColor getNodeColor() {
            const auto memKind = getType().cast<vpux::NDTypeInterface>().getMemoryKind();
            return memKind == vpux::VPU::MemoryKind::DDR ? vpux::DotNodeColor::RED : vpux::DotNodeColor::GREEN;
        }
    }] # IERT_InferLayerReturnTypes<1>.impl;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
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
                IERT_SameShape,
                IERT_SameInOutDimsOrder
            ]
        > {
    let summary = "Convert UPA SHAVE kernel";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output_buff,

        OptionalAttr<F64Attr>:$scale,
        OptionalAttr<F64Attr>:$bias,
        UnitAttr:$fromDetectionOutput,
        UnitAttr:$haveBatch,
        OptionalAttr<IntAttr>:$batchID
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
                IERT_SameShape,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder
            ]
        > {
    let summary = "SoftMax UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        IntAttr:$axisInd
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
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
                IERT_SameElementType,
                IERT_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
            ]
        > {
    let summary = "MAX and AVG Pooling UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        VPUIP_PoolLayerType:$type,
        I64ArrayAttr:$kernel,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$padsBegin,
        I64ArrayAttr:$padsEnd,
        UnitAttr:$excludePad
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
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
                IERT_SameElementType
            ]
        > {
    let summary = "Convolution UPA SHAVE kernel (reference implementation)";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$filter,
        Optional<F16MemRef>:$bias,
        F16MemRef:$output_buff,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$dilations,
        I64ArrayAttr:$padsBegin,
        I64ArrayAttr:$padsEnd,
        IntAttr:$groups
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $filter `:` type($filter) (`,` $bias^ `:` type($bias))? `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
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
                IERT_SameShape,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "ReLU UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
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
                IERT_SameShape,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Sigmoid UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
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
                IERT_SameShape,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "HSwish UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// ErfOp
//

def VPUIP_ErfUPAOp :
        VPUIP_UPATaskOp<1, "ErfUPA",
            [
                ViewLikeOpInterface,
                IERT_SameShape,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Erf UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
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
                IERT_SameShape,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Mish UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// FloorUPAOp
//

def VPUIP_FloorUPAOp :
        VPUIP_UPATaskOp<1, "FloorUPA",
            [
                ViewLikeOpInterface,
                IERT_SameShape,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Floor UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// RoundUPAOp
//

def VPUIP_RoundUPAOp :
        VPUIP_UPATaskOp<1, "RoundUPA",
            [
                ViewLikeOpInterface,
                IERT_SameShape,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Round UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        IE_RoundMode:$mode
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
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
                IERT_SameShape,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Clamp UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        F64Attr:$min,
        F64Attr:$max
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
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
                IERT_SameShape,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Elu UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        F64Attr:$x
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
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
                IERT_SameShape,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Tanh UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// SqrtUPAOp
//

def VPUIP_SqrtUPAOp :
        VPUIP_UPATaskOp<1, "SqrtUPA",
            [
                ViewLikeOpInterface,
                IERT_SameShape,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Sqrt UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// SinhUPAOp
//

def VPUIP_SinhUPAOp :
        VPUIP_UPATaskOp<1, "SinhUPA",
            [
                ViewLikeOpInterface,
                IERT_SameShape,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Sinh UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// CoshUPAOp
//

def VPUIP_CoshUPAOp :
        VPUIP_UPATaskOp<1, "CoshUPA",
            [
                ViewLikeOpInterface,
                IERT_SameShape,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Cosh UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// AsinhUPAOp
//

def VPUIP_AsinhUPAOp :
        VPUIP_UPATaskOp<1, "AsinhUPA",
            [
                ViewLikeOpInterface,
                IERT_SameShape,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Asinh UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// AcoshUPAOp
//

def VPUIP_AcoshUPAOp :
        VPUIP_UPATaskOp<1, "AcoshUPA",
            [
                ViewLikeOpInterface,
                IERT_SameShape,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Acosh UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// AbsUPAOp
//

def VPUIP_AbsUPAOp :
        VPUIP_UPATaskOp<1, "AbsUPA",
            [
                ViewLikeOpInterface,
                IERT_SameShape,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Abs UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// AtanUPAOp
//

def VPUIP_AtanUPAOp :
        VPUIP_UPATaskOp<1, "AtanUPA",
            [
                ViewLikeOpInterface,
                IERT_SameShape,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Atan UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// AsinUPAOp
//

def VPUIP_AsinUPAOp :
        VPUIP_UPATaskOp<1, "AsinUPA",
            [
                ViewLikeOpInterface,
                IERT_SameShape,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Asin UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// AcosUPAOp
//

def VPUIP_AcosUPAOp :
        VPUIP_UPATaskOp<1, "AcosUPA",
            [
                ViewLikeOpInterface,
                IERT_SameShape,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Acos UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// AtanhUPAOp
//

def VPUIP_AtanhUPAOp :
        VPUIP_UPATaskOp<1, "AtanhUPA",
            [
                ViewLikeOpInterface,
                IERT_SameShape,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Atanh UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// LogUPAOp
//

def VPUIP_LogUPAOp :
        VPUIP_UPATaskOp<1, "LogUPA",
            [
                ViewLikeOpInterface,
                IERT_SameShape,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Log UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// GeluUPAOp
//

def VPUIP_GeluUPAOp :
        VPUIP_UPATaskOp<1, "GeluUPA",
            [
                ViewLikeOpInterface,
                IERT_SameShape,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Gelu UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
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
                IERT_SameShape,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Exp UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
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
                IERT_SameElementType,
                IERT_SameInOutDimsOrder_NCHW_NHWC
            ]
        > {
    let summary = "FakeQuantize UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        IntAttr:$levels,
        Const_ContentAttr:$input_low,
        Const_ContentAttr:$input_high,
        Const_ContentAttr:$output_low,
        Const_ContentAttr:$output_high
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
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
                IERT_SameShape,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "FakeQuantize UPA SHAVE kernel";

    let arguments = (ins
        MemRefOf<[F16, quant_QuantizedType]>:$input,
        MemRefOf<[F16, quant_QuantizedType]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, quant_QuantizedType]>:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # commonExtraClassDeclaration;
}

//
// PReluUPAOp
//

def VPUIP_PReluUPAOp :
        VPUIP_UPATaskOp<1, "PReluUPA",
            [
                ViewLikeOpInterface,
                IERT_SameInOutDimsOrder
            ]
        > {
    let summary = "PRelu UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$negative_slope,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $negative_slope `:` type($negative_slope) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
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
                IERT_SameInOutDimsOrder
            ]
        > {
    let summary = "LeakyRelu UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        F64Attr:$negative_slope
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
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
            ]
        > {
    let summary = "Eltwise UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input1,
        F16MemRef:$input2,
        F16MemRef:$output_buff,

        VPU_EltwiseType:$type
    );

    let results = (outs
        F16MemRef:$output
    );

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # commonExtraClassDeclaration;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input1 `:` type($input1) `,` $input2 `:` type($input2) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// LogicalNotUPAOp
//

def VPUIP_LogicalNotUPAOp :
        VPUIP_UPATaskOp<1, "LogicalNotUPA",
            [
                ViewLikeOpInterface,
            ]
        > {
    let summary = "Logical Not UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input1,
        F16MemRef:$output_buff,

        VPU_EltwiseType:$type
    );

    let results = (outs
        F16MemRef:$output
    );

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # commonExtraClassDeclaration;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input1 `:` type($input1) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// SelectUPAOp
//

def VPUIP_SelectUPAOp :
        VPUIP_UPATaskOp<1, "SelectUPA",
            [
                ViewLikeOpInterface,
                IERT_SameShape,
                IERT_SameInOutDimsOrder
            ]
        > {
    let summary = "Select UPA SHAVE kernel";

    let arguments = (ins
        MemRefOf<[SI32,F16]>:$input1,
        MemRefOf<[SI32,F16]>:$input2,
        MemRefOf<[SI32,F16]>:$input3,
        MemRefOf<[SI32,F16]>:$output_buff
    );

    let results = (outs
        MemRefOf<[SI32,F16]>:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input1 `:` type($input1) `,` $input2 `:` type($input2) `,` $input3 `:` type($input3) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// DetectionOutput
//

def VPUIP_DetectionOutputUPAOp :
        VPUIP_UPATaskOp<1, "DetectionOutputUPA",
            [
                AttrSizedOperandSegments,
                ViewLikeOpInterface,
                IERT_AnyDimsOrder
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

        IE_DetectionOutputAttrs:$attr
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $in_box_logits `:` type($in_box_logits) `,` $in_class_preds `:` type($in_class_preds) `,` $in_proposals `:` type($in_proposals) (`,` $in_additional_preds^ `:` type($in_additional_preds))? (`,` $in_additional_proposals^ `:` type($in_additional_proposals))? `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
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
                IERT_SameInOutDimsOrder
            ]
        > {
    let summary = "Swish UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        F64Attr:$beta_value
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
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
                IERT_SameInOutDimsOrder_NCHW_NHWC
            ]
        > {
    let summary = "GRN UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        F64Attr:$bias
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// ReduceUPAOp
//

def VPUIP_ReduceUPAOp :
        VPUIP_UPATaskOp<1, "ReduceUPA",
            [
                ViewLikeOpInterface,
                IERT_AnyDimsOrder
            ]
        > {
    let summary = "Reduce UPA SHAVE kernel";

    let arguments = (ins
        MemRefOf<[F16, SI32]>:$input,
        MemRefOf<[SI32]>:$axes,
        MemRefOf<[F16, SI32]>:$output_buff,

        UnitAttr:$keep_dims,
        VPUIP_ReduceLayerType:$type
    );

    let results = (outs
        MemRefOf<[F16, SI32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $axes `:` type($axes) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// NormUPAOp
//

def VPUIP_NormUPAOp :
        VPUIP_UPATaskOp<1, "NormUPA",
            [
                ViewLikeOpInterface,
                IERT_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
            ]
        > {
    let summary = "Norm UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,

        F16MemRef:$output_buff,

        F64Attr:$alpha,
        F64Attr:$beta,
        F64Attr:$bias,
        IntAttr:$local_size,
        IE_LRN_IERegion:$region
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];
}

//
// BroadcastUPAOp
//

def VPUIP_BroadcastUPAOp :
        VPUIP_UPATaskOp<1, "BroadcastUPA",
            [
                ViewLikeOpInterface,
                IERT_AnyDimsOrder
            ]
        > {
    let summary = "Broadcast UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        MemRefOf<[SI32]>:$target_shape,
        Optional<MemRefOf<[SI32]>>:$axes_mapping,

        F16MemRef:$output_buff,

        OptionalAttr<IE_BroadcastType>:$mode
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $target_shape `:` type($target_shape) (`,` $axes_mapping^ `:` type($axes_mapping))? `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// ScaleShiftUPAOp
//

def VPUIP_ScaleShiftUPAOp :
        VPUIP_UPATaskOp<1, "ScaleShiftUPA",
            [
                AttrSizedOperandSegments,
                ViewLikeOpInterface,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "ScaleShift UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        Optional<F16MemRef>:$weights,
        Optional<F16MemRef>:$biases,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) (`,` $weights^ `:` type($weights))? (`,` $biases^ `:` type($biases))? `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
    }];
}

def VPUIP_UpsamplingUPAOp :
        VPUIP_UPATaskOp<1, "UpsamplingUPA",
            [
                ViewLikeOpInterface,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
            ]
        > {
    let summary = "Upsampling UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        I64ArrayAttr:$upsampling_factor,
        I64ArrayAttr:$pad_l,
        I64ArrayAttr:$pad_r
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// PerAxisTileUPAOp
//

def VPUIP_PerAxisTileUPAOp :
        VPUIP_UPATaskOp<1, "PerAxisTileUPA",
            [
                ViewLikeOpInterface,
                IERT_SameInOutDimsOrder
            ]
        > {
    let summary = "Tile for per axis case UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        IntAttr:$axis,
        IntAttr:$tiles
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
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
                IERT_AnyDimsOrder
            ]
        > {
    let summary = "Negative UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// SignUPAOp
//

def VPUIP_SignUPAOp :
        VPUIP_UPATaskOp<1, "SignUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "Sign UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
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
                IERT_AnyDimsOrder
            ]
        > {
    let summary = "Proposal UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$class_probs,
        F16MemRef:$bbox_deltas,
        F16MemRef:$image_shape,
        F16MemRef:$output_buff,

        IE_ProposalAttrs:$proposal_attrs
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $class_probs `:` type($class_probs) `,` $bbox_deltas `:` type($bbox_deltas) `,` $image_shape `:` type($image_shape) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
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
                IERT_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
            ]
        > {
    let summary = "ROIPooling UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$coords,
        F16MemRef:$output_buff,

        I64ArrayAttr:$output_size,
        F64Attr:$spatial_scale,
        IE_ROIPoolingMethod:$method
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $coords `:` type($coords) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// ROIAlignUPAOp
//

def VPUIP_ROIAlignUPAOp :
        VPUIP_UPATaskOp<1, "ROIAlignUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "ROIAlign UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$coords,
        MemRefOf<[SI32]>:$roisIdx,
        F16MemRef:$output_buff,

        IntAttr:$pooled_h,
        IntAttr:$pooled_w,
        IntAttr:$sampling_ratio,
        F64Attr:$spatial_scale,
        IE_ROIAlignMethod:$poolingMode
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $coords `:` type($coords) `,` $roisIdx `:` type($roisIdx)`)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// PermuteUPAOp
//

def VPUIP_PermuteUPAOp :
        VPUIP_UPATaskOp<1, "PermuteUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "Permute UPA SHAVE kernel";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output_buff,

        AffineMapAttr:$order_value
    );

    let results = (outs
        AnyMemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # commonExtraClassDeclaration;
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
        VPU_PaddingAttr:$pad,
        VPU_MPEMode:$mpe_mode,
        OptionalAttr<IntAttr>:$cluster_id
    );

    let builders = [
        OpBuilder<(ins
            "mlir::ArrayAttr":$start,
            "mlir::ArrayAttr":$end,
            "VPU::PaddingAttr":$pad,
            "VPU::MPEMode":$mpeMode
        )>
    ];

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
        VPU_PPEMode:$ppe_layer_type,

        OptionalAttr<IntAttr>:$clamp_low,
        OptionalAttr<IntAttr>:$clamp_high,
        OptionalAttr<IntAttr>:$lrelu_mult,
        OptionalAttr<IntAttr>:$lrelu_shift,
        OptionalAttr<I64ArrayAttr>:$quant_mult,
        OptionalAttr<I64ArrayAttr>:$quant_shift,
        OptionalAttr<IntAttr>:$quant_post_shift
    );

    let builders = [
        OpBuilder<(ins
            CArg<"vpux::VPU::PPEMode", "vpux::VPU::PPEMode::NOOP">:$ppe_layer_type,
            CArg<"int64_t", "std::numeric_limits<int32_t>::min()">:$clamp_low,
            CArg<"int64_t", "std::numeric_limits<int32_t>::max()">:$clamp_high,
            CArg<"int64_t", "1">:$lrelu_mult,
            CArg<"int64_t", "0">:$lrelu_shift
        )>,
        OpBuilder<(ins
            "vpux::VPU::PPEMode":$ppe_layer_type,
            "int64_t":$clamp_low,
            "int64_t":$clamp_high,
            "int64_t":$lrelu_mult,
            "int64_t":$lrelu_shift,
            "int64_t":$quant_mult,
            "int64_t":$quant_shift
        )>
    ];

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
                AttrSizedOperandSegments,
                MultiViewOpInterface,
                NoRegionArguments,
                OpAsmOpInterface,
                DeclareOpInterfaceMethods<VPU_SparseOpInterface>
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
        AnyTypeOf<[MemRefOf<[F16, BF16, quant_QuantizedType]>, VPURT_SparseBuffer]>:$input,
        Optional<AnyTypeOf<[MemRefOf<[F16, BF16, quant_QuantizedType]>, VPURT_SparseBuffer]>>:$weights,
        Optional<MemRefOf<[SI32]>>:$weight_table,
        Optional<MemRefOf<[UI8]>>:$activation_window,

        AnyTypeOf<[AnyMemRef, VPURT_SparseBuffer, VPUIP_DistributedBuffer]>:$parent_input,
        AnyTypeOf<[AnyMemRef, VPURT_SparseBuffer, VPUIP_DistributedBuffer]>:$parent_output,

        AnyTypeOf<[MemRefOf<[F16, F32, BF16, quant_QuantizedType]>, VPURT_SparseBuffer, VPUIP_DistributedBuffer]>:$output_buff,
        Optional<MemRefOf<[UI64]>>:$profiling_data,

        VPUIP_NCETaskType:$task_type,
        OptionalAttr<I64ArrayAttr>:$kernel_size,
        OptionalAttr<I64ArrayAttr>:$kernel_strides,
        OptionalAttr<VPU_PaddingAttr>:$kernel_padding,
        OptionalAttr<IntAttr>:$activation_window_channel_length,
        OptionalAttr<UnitAttr>:$is_continued,
        OptionalAttr<IntAttr>:$cm_sp_pattern,
        OptionalAttr<UnitAttr>:$is_segmented,
        OptionalAttr<IntAttr>:$out_channel_offset
    );

    let results = (outs
        AnyTypeOf<[MemRefOf<[F16, F32, BF16, quant_QuantizedType]>, VPURT_SparseBuffer, VPUIP_DistributedBuffer]>:$output,
        Optional<MemRefOf<[UI64]>>:$profiling_output
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
            "mlir::ArrayAttr":$kernel_size, "mlir::ArrayAttr":$kernel_strides, "vpux::VPU::PaddingAttr":$kernel_padding,
            "mlir::IntegerAttr":$activation_window_channel_length,
            CArg<"mlir::UnitAttr", "nullptr">:$is_continued,
            CArg<"mlir::IntegerAttr", "nullptr">:$cm_sp_pattern,
            CArg<"mlir::UnitAttr", "nullptr">:$is_segmented,
            CArg<"mlir::IntegerAttr", "nullptr">:$out_channel_offset
        )>,
        OpBuilder<(ins
            "mlir::Type":$output,
            "mlir::Value":$input, "mlir::Value":$weights, "mlir::Value":$weight_table, "mlir::Value":$activation_window,
            "mlir::Value":$parent_input, "mlir::Value":$parent_output,
            "mlir::Value":$output_buff,
            "vpux::VPUIP::NCETaskType":$task_type,
            "mlir::ArrayAttr":$kernel_size, "mlir::ArrayAttr":$kernel_strides, "vpux::VPU::PaddingAttr":$kernel_padding,
            "mlir::IntegerAttr":$activation_window_channel_length,
            CArg<"mlir::UnitAttr", "nullptr">:$is_continued,
            CArg<"mlir::IntegerAttr", "nullptr">:$cm_sp_pattern,
            CArg<"mlir::UnitAttr", "nullptr">:$is_segmented,
            CArg<"mlir::IntegerAttr", "nullptr">:$out_channel_offset
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
        (`profiling_data` `(` $profiling_data^  `:` type($profiling_data) `)`)?
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
                                          vpux::VPU::PaddingAttr pad, vpux::VPU::MPEMode mpeMode, mlir::IntegerAttr clusterId = nullptr);

        template <typename... Args>
        vpux::VPUIP::PPETaskOp addPPETask(mlir::OpBuilder& builder, Args&&... args);

        int64_t getNumVariants();

        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);

        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::NCE;
        }

        static mlir::StringRef getDefaultDialect() {
            return "VPUIP";
        }

        static mlir::LogicalResult inferReturnTypes(
                mlir::MLIRContext*,
                llvm::Optional<mlir::Location>,
                mlir::ValueRange operands,
                mlir::DictionaryAttr,
                mlir::RegionRange,
                llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes);
    }];
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

        UnitAttr:$mergeRepeated
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $sequenceLengths `:` type($sequenceLengths) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
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

        UnitAttr:$mergeRepeated
    );

    let results = (outs
        MemRefOf<[SI32]>:$output,
        MemRefOf<[SI32]>:$outputLength
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $sequenceLength `:` type($sequenceLength) `,` ($blankIndex^ `:` type($blankIndex))? `)`
        `outputs` `(` $output_buff `:` type($output_buff) `,` $outputLength_buff `:` type($outputLength_buff) `)`
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # commonExtraClassDeclaration;
}

//
// PadOp
//

def VPUIP_PadUPAOp :
        VPUIP_UPATaskOp<1,"Pad",
            [
                ViewLikeOpInterface,
                IERT_SameInOutDimsOrder
            ]
        > {
    let summary = "Pad UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        OptionalAttr<F64Attr>:$pad_value,
        IE_PadMode:$mode
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
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
                IERT_SameInOutDimsOrder
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
        F16MemRef:$outputCellState_buff
    );

    let results = (outs
        F16MemRef:$outputHiddenState,
        F16MemRef:$outputCellState
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $inputData `:` type($inputData) `,` $initialHiddenState `:` type($initialHiddenState)
        `,` $initialCellState `:` type($initialCellState) `,` $weights `:` type($weights) `,` $biases `:` type($biases) `)`
        `outputs` `(` $outputHiddenState_buff `:` type($outputHiddenState_buff)
        `,` $outputCellState_buff `:` type($outputCellState_buff) `)`
        `->` type(results)
    }];
}

//
// VPUIP_SwKernelRun
//

def VPUIP_SwKernelRun
    : VPUIP_Op<"SW.Kernel.run"> {

    let arguments  = (ins
        Variadic<AnyType>:$args,
        OptionalAttr<ArrayAttr>:$attrs
    );

    let assemblyFormat = [{
        attr-dict
        `(` operands `)` `:` type(operands)
    }];
}

//
// VPUIP_SwKernelOp
//

def VPUIP_SwKernelOp :
        VPUIP_TaskOp<1, "SW.Kernel",
            [
                MultiViewOpInterface,
                IsolatedFromAbove,
                AttrSizedOperandSegments
            ]
            # GraphRegionNoTerminator.traits
        > {

    let summary = "Software Layer Task";

    let description = [{
        This operation defines Activation shave task
    }];

    let arguments = (ins
        SymbolRefAttr:$kernelFunction,
        Variadic<AnyMemRef>:$inputs,
        Variadic<AnyMemRef>:$output_buffs,
        OptionalAttr<IntAttr>:$tileIndex
    );

    let results = (outs Variadic<AnyMemRef>:$results);

    let regions = (region
        SizedRegion<1>:$body
    );

    let builders = [
        OpBuilder<(ins
            "mlir::ValueRange":$inputs,
            "mlir::ValueRange":$output_buffs,
            "mlir::SymbolRefAttr":$kernelFunction,
            "mlir::IntegerAttr":$tileIndex
        )>
    ];

    let assemblyFormat = [{
        attr-dict

        $kernelFunction

        `inputs` `(`$inputs `:` type($inputs)`)`
        `outputs` `(`$output_buffs `:` type($output_buffs)`)`
        (`on` `tile` $tileIndex^)?
        `->` type(results)

        $body

    }];

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::SHAVE_ACT;
        }

        static mlir::LogicalResult inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> loc,
                                                    mlir::ValueRange operands, mlir::DictionaryAttr attrs, mlir::RegionRange regions,
                                                    mlir::SmallVectorImpl<mlir::Type>& inferredTypes);

        static vpux::IERT::KernelInfo getKernelInfo(mlir::Operation* origOp);
    }];
}

//
// LSTMSequenceUPAOp
//

def VPUIP_LSTMSequenceUPAOp :
        VPUIP_UPATaskOp<3, "LSTMSequenceUPA",
            [
                MultiViewOpInterface
            ]
        > {
    let summary = "LSTMSequence UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$inputData,
        F16MemRef:$initialHiddenState,
        F16MemRef:$initialCellState,
        F16MemRef:$weights,
        F16MemRef:$biases,
        F16MemRef:$outputHiddenValues_buff,
        F16MemRef:$outputCellState_buff,
        F16MemRef:$outputHiddenState_buff,

        IntAttr:$sequenceLength,
        IE_RNNSequenceDirection:$direction
    );

    let results = (outs
        F16MemRef:$outputHiddenValues,
        F16MemRef:$outputCellState,
        F16MemRef:$outputHiddenState
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $inputData `:` type($inputData) `,` $initialHiddenState `:` type($initialHiddenState)
        `,` $initialCellState `:` type($initialCellState) `,` $weights `:` type($weights) `,` $biases `:` type($biases) `)`
        `outputs` `(` $outputHiddenValues_buff `:` type($outputHiddenValues_buff)
        `,` $outputCellState_buff `:` type($outputCellState_buff)
        `,` $outputHiddenState_buff `:` type($outputHiddenState_buff) `)`
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # commonExtraClassDeclaration;
}

//
// InterpolateOp
//

def VPUIP_InterpolateUPAOp :
        VPUIP_UPATaskOp<1,"Interpolate",
            [
                ViewLikeOpInterface,
            ]
        > {
    let summary = "Interpolate UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,

        F16MemRef:$output_buff,

        IE_InterpolateMode:$mode,
        IE_InterpolateCoordMode:$coord_mode,
        IE_InterpolateNearestMode:$nearest_mode,
        UnitAttr:$antialias
    );

    let results = (outs
        F16MemRef:$output
    );

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # commonExtraClassDeclaration;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
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
             IERT_AnyDimsOrder
         ]
     > {
    let summary = "StridedSlice UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        I64ArrayAttr:$begins,
        I64ArrayAttr:$ends,
        I64ArrayAttr:$strides
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
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
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # commonExtraClassDeclaration;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $weights `:` type($weights) (`,` $bias^ `:` type($bias))? `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
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
        OptionalAttr<I64ArrayAttr>:$mask
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # commonExtraClassDeclaration;
}

//
// ReorgYoloUPAOp
//

def VPUIP_ReorgYoloUPAOp :
        VPUIP_UPATaskOp<1, "ReorgYoloUPA",
            [
                ViewLikeOpInterface,
                IERT_AnyDimsOrder
            ]
        > {
    let summary = "ReorgYolo UPA SHAVE kernel";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output_buff,

        IntAttr:$stride,

        OptionalAttr<IntAttr>:$maxShaves
    );

    let results = (outs
        AnyMemRef:$output
    );

    let builders = [
        OpBuilder<
            (ins
                "mlir::Value":$input,
                "mlir::Value":$output,
                "mlir::IntegerAttr":$stride
            )
        >
    ];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// MVNOp
//

def VPUIP_MVNUPAOp :
        VPUIP_UPATaskOp<1, "MVNUPA",
            [
                ViewLikeOpInterface,
                IERT_SameShape,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder
            ]
        > {
    let summary = "MVN UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        OptionalAttr<BoolAttr>:$across_channels,
        OptionalAttr<BoolAttr>:$normalize_variance,
        F64Attr:$eps
    );

    let results = (outs
        AnyMemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// GatherUPAOp
//

def VPUIP_GatherUPAOp :
        VPUIP_UPATaskOp<1, "GatherUPA",
            [
                ViewLikeOpInterface,
                IERT_AnyDimsOrder
            ]
        > {
    let summary = "Gather UPA SHAVE kernel";

    let arguments = (ins
        AnyMemRef:$input,
        MemRefOf<[SI32]>:$indices,
        AnyMemRef:$output_buff,

        IntAttr:$axis,
        IntAttr:$batch_dims
    );

    let results = (outs
        AnyMemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];

    let extraClassDeclaration = [{
        static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
    }] # commonExtraClassDeclaration;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $indices `:` type($indices) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// SpaceToDepthOp
//

def VPUIP_SpaceToDepthUPAOp :
        VPUIP_UPATaskOp<1, "SpaceToDepthUPA",
            [
                ViewLikeOpInterface,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder,
            ]
        > {
    let summary = "SpaceToDepth UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        IntAttr:$block_size,
        IE_SpaceToDepthMode:$mode
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// ScatterNDUpdateUPAOp
//

def VPUIP_ScatterNDUpdateUPAOp :
        VPUIP_UPATaskOp<1, "ScatterNDUpdateUPA",
            [
                ViewLikeOpInterface,
                IERT_AnyDimsOrder
            ]
        > {
    let summary = "ScatterNDUpdate UPA SHAVE kernel";

    let arguments = (ins
        AnyMemRef:$input,
        MemRefOf<[SI32]>:$indices,
        AnyMemRef:$updates,
        AnyMemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $indices `:` type($indices) `,` $updates `:` type($updates) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// CeilingUPAOp
//

def VPUIP_CeilingUPAOp :
        VPUIP_UPATaskOp<1, "CeilingUPA",
            [
                ViewLikeOpInterface,
                IERT_SameShape,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Ceiling UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// GatherElementsUPAOp
//

def VPUIP_GatherElementsUPAOp :
        VPUIP_UPATaskOp<1, "GatherElementsUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "GatherElements UPA SHAVE kernel";

    let arguments = (ins
        AnyMemRef:$input,
        MemRefOf<[SI64, SI32]>:$indices,
        AnyMemRef:$output_buff,

        IntAttr:$axis
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $indices `:` type($indices) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// DepthToSpaceUPAOp
//

def VPUIP_DepthToSpaceUPAOp :
        VPUIP_UPATaskOp<1, "DepthToSpaceUPA",
            [
                ViewLikeOpInterface,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder
            ]
        > {
    let summary = "DepthToSpace UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        IntAttr:$block_size,
        IE_DepthToSpaceMode:$mode
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// NormalizeIEOp
//

def VPUIP_NormalizeIEUPAOp :
        VPUIP_UPATaskOp<1, "NormalizeIEUPA",
            [
                ViewLikeOpInterface,
                IERT_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
            ]
        > {
    let summary = "NormalizeIE UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$data,
        F16MemRef:$weights,
        F16MemRef:$output_buff,

        F64Attr:$eps,
        BoolAttr:$across_spatial,
        BoolAttr:$channel_shared
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $data `:` type($data) `,` $weights `:` type($weights) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// ReverseSequenceOp
//

def VPUIP_ReverseSequenceUPAOp :
        VPUIP_UPATaskOp<1, "ReverseSequenceUPA",
            [
                ViewLikeOpInterface,
                IERT_SameInOutDimsOrder
            ]
        > {
    let summary = "ReverseSequence UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$data,
        F16MemRef:$seq_length,
        F16MemRef:$output_buff,

        IntAttr:$seq_axis,
        IntAttr:$batch_axis
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $data `:` type($data) `,` $seq_length `:` type($seq_length) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// TopKOp
//

def VPUIP_TopKUPAOp:
        VPUIP_UPATaskOp<2,"TopK",
            [
                MultiViewOpInterface
            ]
        > {
    let summary = "TopK UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        MemRefOf<[SI32]>:$k,
        F16MemRef:$output_values_buff,
        MemRefOf<[SI32]>:$target_shape_buff,

        IntAttr:$axis,
        IE_TopKMode:$mode,
        IE_TopKSortType:$sort,
        TypeAttr:$element_type
    );

    let results = (outs
        F16MemRef:$output_values,
        MemRefOf<[SI32]>:$target_shape
    );
}

//
// SoftPlusUPAOp
//

def VPUIP_SoftPlusUPAOp :
        VPUIP_UPATaskOp<1, "SoftPlusUPA",
            [
                ViewLikeOpInterface,
                IERT_SameShape,
                IERT_SameElementType,
                IERT_SameInOutDimsOrder
            ]
        > {
    let summary = "SoftPlus UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// YuvToRgbOp
//

def VPUIP_YuvToRgbUPAOp :
        VPUIP_UPATaskOp<1,"YuvToRgbUPA",
            [
                ViewLikeOpInterface,
                AttrSizedOperandSegments,
                IERT_AnyDimsOrder
            ]
        > {
    let summary = "Yuv-to-Rgb UPA SHAVE kernel";

    let arguments = (ins
                 MemRefOf<[I8, F16, F32]> :$input1,
        Optional<MemRefOf<[I8, F16, F32]>>:$input2,
        Optional<MemRefOf<[I8, F16, F32]>>:$input3,
                 MemRefOf<[I8, F16, F32]> :$output_buff,

        IE_ColorFmt:$inFmt,
        IE_ColorFmt:$outFmt
    );

    let results = (outs
        MemRefOf<[I8, F16, F32]>:$output
    );

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input1 `:` type($input1) (`,` $input2^ `:` type($input2))? (`,` $input3^ `:` type($input3))? `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

}

//
// NCEClusterTiling
//

def VPUIP_NCEClusterTilingOp :
        VPUIP_Op<
            "NCEClusterTiling",
            [
                NoTerminator,
                IsolatedFromAbove,
                MultiViewOpInterface,
                AttrSizedOperandSegments,
                MemoryEffectsOpInterface,
                DeclareOpInterfaceMethods<IERT_LayerOpInterface>,
                DeclareOpInterfaceMethods<RegionBranchOpInterface,
                    ["getSuccessorEntryOperands", "getNumRegionInvocations"]>,
                IERT_AsyncLayerOpInterface
            ]
        > {
    let summary = "Operation that encapsulates details of tiling operation between clusters";

    let arguments = (ins
        Variadic<AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>>:$inputs,
        Variadic<AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>>:$output_buffs
    );

    let results = (outs
        Variadic<AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>>:$results
    );

    let regions = (region SizedRegion<1>:$body);

    let printer = [{ return vpux::VPUIP::print(p, *this); }];
    let parser = [{ return vpux::VPUIP::parse$cppClass(parser, result); }];
    let verifier = [{ return vpux::VPUIP::verifyOp(*this); }];

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "mlir::TypeRange":$resultTypes, "mlir::ValueRange":$operands,
            "llvm::function_ref<void(mlir::OpBuilder&, mlir::Location, mlir::ValueRange)>":$bodyBuilder)>,
    ];

    let extraClassDeclaration = [{
        using BodyBuilderFn =
            llvm::function_ref<void(mlir::OpBuilder&, mlir::Location, mlir::ValueRange)>;

        mlir::Operation* getInnerTaskOp();
        mlir::MutableArrayRef<mlir::BlockArgument> getInnerInputs();
        mlir::MutableArrayRef<mlir::BlockArgument> getInnerOutputs();

        vpux::IndexedSymbolAttr getExecutor();
    }];
}

//
// ExtractImagePatchesUPAOp
//

def VPUIP_ExtractImagePatchesUPAOp :
        VPUIP_UPATaskOp<1, "ExtractImagePatchesUPA",
            [
                ViewLikeOpInterface,
                IERT_SameInOutDimsOrder
            ]
        > {
    let summary = "ExtractImagePatches UPA SHAVE kernel";

    let arguments = (ins
        AnyMemRef:$data,
        AnyMemRef:$output_buff,

        I64ArrayAttr:$sizes,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$rates,
        IE_PadType:$auto_pad,

        OptionalAttr<IntAttr>:$maxShaves
    );

    let results = (outs
        AnyMemRef:$output
    );

    let builders = [
        OpBuilder<
            (ins
                "mlir::Value":$data, 
                "mlir::Value":$output,
                "mlir::ArrayAttr":$sizes, 
                "mlir::ArrayAttr":$strides,
                "mlir::ArrayAttr":$rates, 
                "vpux::IE::PadTypeAttr":$auto_pad
            )
        >
    ];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $data `:` type($data) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

    //let extraClassDeclaration = [{
   //     static void inferLayoutInfo(mlir::Operation* origOp, vpux::IE::LayerLayoutInfo& info);
   // }] # commonExtraClassDeclaration;

    let verifier = [{
        return vpux::VPUIP::verifyOp(*this);
    }];
}

#endif
