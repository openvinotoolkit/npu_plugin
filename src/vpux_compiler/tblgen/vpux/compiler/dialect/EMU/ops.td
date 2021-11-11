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

#ifndef VPUX_COMPILER_DIALECT_EMU_OPS
#define VPUX_COMPILER_DIALECT_EMU_OPS

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/core/ops_interfaces.td"
include "vpux/compiler/dialect/const/attributes.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/EMU/dialect.td"
include "vpux/compiler/dialect/EMU/ops_interfaces.td"
include "vpux/compiler/dialect/EMU/attributes.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Dialect/Quant/QuantOpsBase.td"

//
// Base classes
//

class EMU_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<
            EMU_Dialect,
            mnemonic,
            traits
        >;

class EMU_TaskOp<string mnemonic, list<OpTrait> traits = []> :
        EMU_Op<
            mnemonic,
            [
                NoSideEffect,
                DeclareOpInterfaceMethods<EMU_TaskOpInterface>
            ] # traits
        > {

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
}

class EMU_UPATaskOp<string mnemonic, list<OpTrait> traits = []> :
        EMU_TaskOp<
            mnemonic,
            !listconcat(
                traits,
                [
                    DeclareOpInterfaceMethods<EMU_UPATaskOpInterface>
                ]
            )
        >;

//
// ConvertUPAOp
//

def EMU_ConvertUPAOp :
        EMU_UPATaskOp<"ConvertUPA",
            [
                SameOperandsAndResultShape
            ]
        > {
    let summary = "Convert UPA SHAVE kernel";

    let arguments = (ins
        AnyRankedTensor:$input,

        OptionalAttr<F64Attr>:$scale,
        OptionalAttr<F64Attr>:$bias,
        UnitAttr:$fromDetectionOutput,
        UnitAttr:$haveBatch,
        OptionalAttr<IntAttr>:$batchID
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let verifier = [{
        return vpux::EMU::verifyOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];
}

//
// SoftMaxUPAOp
//

def EMU_SoftMaxUPAOp :
        EMU_UPATaskOp<"SoftMaxUPA",
            [
                SameOperandsAndResultShape,
                SameOperandsAndResultElementType
            ]
        > {
    let summary = "SoftMax UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input,

        IntAttr:$axisInd
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let verifier = [{
        return vpux::EMU::verifyOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];
}

//
// PoolingUPAOp
//

def EMU_PoolingUPAOp :
        EMU_UPATaskOp<"PoolingUPA",
            [
                SameOperandsAndResultElementType
            ]
        > {
    let summary = "MAX and AVG Pooling UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input,

        EMU_PoolLayerType:$type,
        I64ArrayAttr:$kernel,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$padsBegin,
        I64ArrayAttr:$padsEnd,
        UnitAttr:$excludePad
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let verifier = [{
        return vpux::EMU::verifyOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];
}

//
// ConvolutionUPAOp
//

def EMU_ConvolutionUPAOp :
        EMU_UPATaskOp<"ConvolutionUPA",
            [
                SameOperandsAndResultElementType
            ]
        > {
    let summary = "Convolution UPA SHAVE kernel (reference implementation)";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input,
        RankedTensorOf<[F16]>:$filter,
        Optional<RankedTensorOf<[F16]>>:$bias,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$dilations,
        I64ArrayAttr:$padsBegin,
        I64ArrayAttr:$padsEnd,
        IntAttr:$groups
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $filter `:` type($filter) (`,` $bias^ `:` type($bias))? `)`
        `->` type(results)
    }];

    let verifier = [{
        return vpux::EMU::verifyOp(*this);
    }];
}

//

// ReLUUPAOp
//

def EMU_ReLUUPAOp :
        EMU_UPATaskOp<"ReLUUPA",
            [
                SameOperandsAndResultShape,
                SameOperandsAndResultElementType
            ]
        > {
    let summary = "ReLU UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let verifier = [{
        return vpux::EMU::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];
}

//
// SigmoidOp
//

def EMU_SigmoidUPAOp :
        EMU_UPATaskOp<"SigmoidUPA",
            [
                SameOperandsAndResultShape,
                SameOperandsAndResultElementType
            ]
        > {
    let summary = "Sigmoid UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let verifier = [{
        return vpux::EMU::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];
}

//
// HSwishUPAOp
//

def EMU_HSwishUPAOp :
        EMU_UPATaskOp<"HSwishUPA",
            [
                SameOperandsAndResultShape,
                SameOperandsAndResultElementType
            ]
        > {
    let summary = "HSwish UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let verifier = [{
        return vpux::EMU::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];
}

//
// ErfOp
//

def EMU_ErfUPAOp :
        EMU_UPATaskOp<"ErfUPA",
            [
                SameOperandsAndResultShape,
                SameOperandsAndResultElementType
            ]
        > {
    let summary = "Erf UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let verifier = [{
        return vpux::EMU::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];
}

//
// MishUPAOp
//

def EMU_MishUPAOp :
        EMU_UPATaskOp<"MishUPA",
            [
                SameOperandsAndResultShape,
                SameOperandsAndResultElementType
            ]
        > {
    let summary = "Mish UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let verifier = [{
        return vpux::EMU::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];
}

//
// FloorUPAOp
//

def EMU_FloorUPAOp :
        EMU_UPATaskOp<"FloorUPA",
            [
                SameOperandsAndResultShape,
                SameOperandsAndResultElementType
            ]
        > {
    let summary = "Floor UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let verifier = [{
        return vpux::EMU::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];
}

//
// RoundUPAOp
//

def EMU_RoundUPAOp :
        EMU_UPATaskOp<"RoundUPA",
            [
                SameOperandsAndResultShape,
                SameOperandsAndResultElementType
            ]
        > {
    let summary = "Round UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input,

        IE_RoundMode:$mode
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let verifier = [{
        return vpux::EMU::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
       }];
} 

//
// ClampUPAOp
//

def EMU_ClampUPAOp :
        EMU_UPATaskOp<"ClampUPA",
            [
                SameOperandsAndResultShape,
                SameOperandsAndResultElementType
            ]
        > {
    let summary = "Clamp UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input,

        F64Attr:$min,
        F64Attr:$max
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let verifier = [{
        return vpux::EMU::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];
}

//
// EluUPAOp
//

def EMU_EluUPAOp :
        EMU_UPATaskOp<"EluUPA",
            [
                SameOperandsAndResultShape,
                SameOperandsAndResultElementType
            ]
        > {
    let summary = "Elu UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input,

        F64Attr:$x
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let verifier = [{
        return vpux::EMU::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];
}

//
// TanhUPAOp
//

def EMU_TanhUPAOp :
        EMU_UPATaskOp<"TanhUPA",
            [
                SameOperandsAndResultShape,
                SameOperandsAndResultElementType
            ]
        > {
    let summary = "Tanh UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let verifier = [{
        return vpux::EMU::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];
}

//
// SqrtUPAOp
//

def EMU_SqrtUPAOp :
        EMU_UPATaskOp<"SqrtUPA",
            [
                SameOperandsAndResultShape,
                SameOperandsAndResultElementType
            ]
        > {
    let summary = "Sqrt UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let verifier = [{
        return vpux::EMU::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];
}

//
// LogUPAOp
//

def EMU_LogUPAOp :
        EMU_UPATaskOp<"LogUPA",
            [
                SameOperandsAndResultShape,
                SameOperandsAndResultElementType
            ]
        > {
    let summary = "Log UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let verifier = [{
        return vpux::EMU::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];
}

//
// ExpUPAOp
//

def EMU_ExpUPAOp :
        EMU_UPATaskOp<"ExpUPA",
            [
                SameOperandsAndResultShape,
                SameOperandsAndResultElementType
            ]
        > {
    let summary = "Exp UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let verifier = [{
        return vpux::EMU::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];
}

//
// FakeQuantizeUPAOp
//

def EMU_FakeQuantizeUPAOp :
        EMU_UPATaskOp<"FakeQuantizeUPA",
            [
                SameOperandsAndResultElementType
            ]
        > {
    let summary = "FakeQuantize UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input,

        IntAttr:$levels,
        Const_ContentAttr:$input_low,
        Const_ContentAttr:$input_high,
        Const_ContentAttr:$output_low,
        Const_ContentAttr:$output_high
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let verifier = [{
        return vpux::EMU::verifyOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];
}

//
// QuantCastUPAOp
//

def EMU_QuantCastUPAOp :
        EMU_UPATaskOp<"QuantCastUPA",
            [
                SameOperandsAndResultShape
            ]
        > {
    let summary = "FakeQuantize UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16, quant_QuantizedType]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, quant_QuantizedType]>:$output
    );

    let verifier = [{
        return vpux::EMU::verifyOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];
}

//
// PReluUPAOp
//

def EMU_PReluUPAOp :
        EMU_UPATaskOp<"PReluUPA"> {
    let summary = "PRelu UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input,
        RankedTensorOf<[F16]>:$negative_slope
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $negative_slope `:` type($negative_slope) `)`
        `->` type(results)
    }];
}

//
// LeakyReluUPAOp
//

def EMU_LeakyReluUPAOp :
        EMU_UPATaskOp< "LeakyReluUPA"> {
    let summary = "LeakyRelu UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input,

        F64Attr:$negative_slope
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];
}

//
// EltwiseUPAOp
//

def EMU_EltwiseUPAOp :
        EMU_UPATaskOp<"EltwiseUPA"> {
    let summary = "Eltwise UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input1,
        RankedTensorOf<[F16]>:$input2,

        EMU_EltwiseLayerType:$type
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input1 `:` type($input1) `,` $input2 `:` type($input2) `)`
        `->` type(results)
    }];
}

//
// DetectionOutput
//

def EMU_DetectionOutputUPAOp :
        EMU_UPATaskOp<"DetectionOutputUPA",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "DetectionOutput UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$in_box_logits,
        RankedTensorOf<[F16]>:$in_class_preds,
        RankedTensorOf<[F16]>:$in_proposals,
        Optional<RankedTensorOf<[F16]>>:$in_additional_preds,
        Optional<RankedTensorOf<[F16]>>:$in_additional_proposals,

        IE_DetectionOutputAttrs:$attr
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $in_box_logits `:` type($in_box_logits) `,` $in_class_preds `:` type($in_class_preds) `,` $in_proposals `:` type($in_proposals) (`,` $in_additional_preds^ `:` type($in_additional_preds))? (`,` $in_additional_proposals^ `:` type($in_additional_proposals))? `)`
        `->` type(results)
    }];
}

//
// SwishUPAOp
//

def EMU_SwishUPAOp :
        EMU_UPATaskOp<"SwishUPA"> {
    let summary = "Swish UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input,

        F64Attr:$beta_value
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];
}

//
// GRNUPAOp
//

def EMU_GRNUPAOp :
        EMU_UPATaskOp<"GRNUPA"> {
    let summary = "GRN UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input,

        F64Attr:$bias
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];
}

//
// NormUPAOp
//

def EMU_NormUPAOp :
        EMU_UPATaskOp<"NormUPA"> {
    let summary = "Norm UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input,

        F64Attr:$alpha,
        F64Attr:$beta,
        F64Attr:$bias,
        IntAttr:$local_size,
        IE_LRN_IERegion:$region
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];

    let verifier = [{
        return vpux::EMU::verifyOp(*this);
    }];
}

//
// BroadcastUPAOp
//

def EMU_BroadcastUPAOp :
        EMU_UPATaskOp<"BroadcastUPA"> {
    let summary = "Broadcast UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input,

        RankedTensorOf<[SI32]>:$target_shape,
        Optional<RankedTensorOf<[SI32]>>:$axes_mapping,
        OptionalAttr<IE_BroadcastType>:$mode
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $target_shape `:` type($target_shape) (`,` $axes_mapping^ `:` type($axes_mapping))? `)`
        `->` type(results)
    }];
}

//
// ScaleShiftUPAOp
//

def EMU_ScaleShiftUPAOp :
        EMU_UPATaskOp<"ScaleShiftUPA",
            [
                SameOperandsAndResultElementType,
                AttrSizedOperandSegments
            ]
        > {
    let summary = "ScaleShift UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input,

        Optional<RankedTensorOf<[F16]>>:$weights,
        Optional<RankedTensorOf<[F16]>>:$biases
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) (`,` $weights^ `:` type($weights))? (`,` $biases^ `:` type($biases))? `)`
        `->` type(results)
    }];

    let verifier = [{
        return vpux::EMU::verifyPostOp(*this);
    }];
}

//
// PerAxisTileUPAOp
//

def EMU_PerAxisTileUPAOp :
        EMU_UPATaskOp<"PerAxisTileUPA"> {
    let summary = "Tile for per axis case UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input,

        IntAttr:$axis,
        IntAttr:$tiles
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];
}

//
// NegativeUPAOp
//

def EMU_NegativeUPAOp :
        EMU_UPATaskOp<"NegativeUPA"> {
    let summary = "Negative UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];
}

//
// ProposalUPAOp
//

def EMU_ProposalUPAOp :
        EMU_UPATaskOp<"ProposalUPA"> {
    let summary = "Proposal UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$class_probs,
        RankedTensorOf<[F16]>:$bbox_deltas,
        RankedTensorOf<[F16]>:$image_shape,

        IE_ProposalAttrs:$proposal_attrs
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $class_probs `:` type($class_probs) `,` $bbox_deltas `:` type($bbox_deltas) `,` $image_shape `:` type($image_shape) `)`
        `->` type(results)
    }];
}

//
// ROIPoolingUPAOp
//

def EMU_ROIPoolingUPAOp :
        EMU_UPATaskOp<"ROIPoolingUPA"> {
    let summary = "ROIPooling UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input,
        RankedTensorOf<[F16]>:$coords,

        I64ArrayAttr:$output_size,
        F64Attr:$spatial_scale,
        IE_ROIPoolingMethod:$method
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $coords `:` type($coords) `)`
        `->` type(results)
    }];
}

//
// ROIAlignUPAOp
//

def EMU_ROIAlignUPAOp :
        EMU_UPATaskOp<"ROIAlignUPA"> {
    let summary = "ROIAlign UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input,
        RankedTensorOf<[F16]>:$coords,
        RankedTensorOf<[SI32]>:$roisIdx,

        IntAttr:$pooled_h,
        IntAttr:$pooled_w,
        IntAttr:$sampling_ratio,
        F64Attr:$spatial_scale,
        IE_ROIAlignMethod:$poolingMode
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $coords `:` type($coords) `,` $roisIdx `:` type($roisIdx)`)`
        `->` type(results)
    }];
}

//
// PermuteUPAOp
//

def EMU_PermuteUPAOp :
        EMU_UPATaskOp<"PermuteUPA"> {
    let summary = "Permute UPA SHAVE kernel";

    let arguments = (ins
        AnyRankedTensor:$input,

        OptionalAttr<AffineMapAttr>:$order_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];

    let verifier = [{
        return vpux::EMU::verifyOp(*this);
    }];
}

//
// WeightsTableOp
//

def EMU_WeightsTableOp :
        EMU_Op<"WeightsTableOp",
            [
                NoSideEffect,
                AttrSizedOperandSegments
            ]
        > {
    let summary = "Intermediate task for creating weights table based on the addresses of CMX buffers";

    let arguments = (ins
        RankedTensorOf<[F16, quant_QuantizedType]>:$op_input,
        RankedTensorOf<[F16, quant_QuantizedType]>:$op_output,
        Optional<RankedTensorOf<[F16, quant_QuantizedType]>>:$weights,
        Optional<RankedTensorOf<[F16, F32]>>:$bias
    );

    let results = (outs
        RankedTensorOf<[SI32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `op_input` `(` $op_input  `:` type($op_input) `)`
        `op_output` `(` $op_output  `:` type($op_output) `)`
        (`weights` `(` $weights^  `:` type($weights) `)`)?
        (`bias` `(` $bias^  `:` type($bias) `)`)?
        `->` type(results)
    }];
}

//
// PPETaskOp
//

def EMU_PPETaskOp :
        EMU_Op<
            "PPETask",
            [
                HasParent<"vpux::EMU::NCEClusterTaskOp">
            ]
        > {
    let summary = "PPE Type for NCE Task";

    let arguments = (ins
        EMU_PPELayerType:$ppe_layer_type,

        OptionalAttr<IntAttr>:$clamp_low,
        OptionalAttr<IntAttr>:$clamp_high,
        OptionalAttr<IntAttr>:$lrelu_mult,
        OptionalAttr<IntAttr>:$lrelu_shift,
        OptionalAttr<I32ArrayAttr>:$quant_mult,
        OptionalAttr<I32ArrayAttr>:$quant_shift,
        OptionalAttr<IntAttr>:$quant_post_shift
    );

    let builders = [
        OpBuilder<(ins
            CArg<"vpux::EMU::PPELayerType", "vpux::EMU::PPELayerType::NOOP">:$ppe_layer_type,
            CArg<"int32_t", "std::numeric_limits<int32_t>::min()">:$clamp_low,
            CArg<"int32_t", "std::numeric_limits<int32_t>::max()">:$clamp_high,
            CArg<"int32_t", "1">:$lrelu_mult,
            CArg<"uint32_t", "0">:$lrelu_shift
        )>,
        OpBuilder<(ins
            "vpux::EMU::PPELayerType":$ppe_layer_type,
            "int32_t":$clamp_low,
            "int32_t":$clamp_high,
            "int32_t":$lrelu_mult,
            "uint32_t":$lrelu_shift,
            "ArrayRef<int32_t>":$quant_mult,
            "ArrayRef<int32_t>":$quant_shift,
            "int32_t":$quant_post_shift
        )>
    ];

    let assemblyFormat = [{
        $ppe_layer_type attr-dict
    }];
}

//
// NCEClusterTaskOp
//

def EMU_NCEClusterTaskOp :
        EMU_TaskOp<"NCEClusterTask",
            [
                NoRegionArguments,
                OpAsmOpInterface,
                AttrSizedOperandSegments
            ]
            # GraphRegionNoTerminator.traits
        > {

    let summary = "NCE Cluster Task Operation";

    let description = [{
        This operation defines an NCE task.

        It supports fixed PPE functions as well as generic PPE instruction
        lists. The generic PPE instruction list argument needs to be described as a region of PPE
        supported ops. Single fixed PPE functions and generic PPE instruciton list usage is
        mutually exclusive.
    }];

    let arguments = (ins
        RankedTensorOf<[F16, quant_QuantizedType]>:$input,
        Optional<RankedTensorOf<[F16, quant_QuantizedType]>>:$weights,
        Optional<RankedTensorOf<[SI32]>>:$weight_table,

        EMU_NCETaskType:$task_type,
        OptionalAttr<I64ArrayAttr>:$kernel_size,
        OptionalAttr<I64ArrayAttr>:$kernel_strides,
        OptionalAttr<I64ArrayAttr>:$kernel_padding
    );

    let results = (outs
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$output
    );

    let regions = (region
        SizedRegion<1>:$ppe
    );

    let assemblyFormat = [{
        attr-dict
        `input` `(` $input  `:` type($input) `)`
        (`weights` `(` $weights^  `:` type($weights) `)`)?
        (`weight_table` `(` $weight_table^  `:` type($weight_table) `)`)?
        `->` type(results)
        `PPE` `:` $ppe
    }];

    let verifier = [{
        return vpux::EMU::verifyOp(*this);
    }];

    let extraClassDeclaration = [{
        template <typename... Args>
        vpux::EMU::PPETaskOp addPPETask(mlir::OpBuilder& builder, Args&&... args);

        static vpux::EMU::TaskType getTaskType() {
            return vpux::EMU::TaskType::NCE2;
        }

        static mlir::StringRef getDefaultDialect() {
            return "EMU";
        }
    }];
}

//
// CTCGreedyDecoderUPAOp
//

def EMU_CTCGreedyDecoderUPAOp :
        EMU_UPATaskOp<"CTCGreedyDecoderUPA"> {
    let summary = "CTCGreedyDecoder UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input,
        RankedTensorOf<[F16]>:$sequenceLengths,

        UnitAttr:$mergeRepeated
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let verifier = [{
        return vpux::EMU::verifyOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $sequenceLengths `:` type($sequenceLengths) `)`
        `->` type(results)
    }];
}

//
// CTCGreedyDecoderSeqLenUPAOp
//

def EMU_CTCGreedyDecoderSeqLenUPAOp :
        EMU_UPATaskOp<"CTCGreedyDecoderSeqLenUPA"> {
    let summary = "CTCGreedyDecoderSeqLen UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input,
        RankedTensorOf<[SI32]>:$sequenceLength,
        Optional<RankedTensorOf<[SI32]>>:$blankIndex,

        UnitAttr:$mergeRepeated
    );

    let results = (outs
        RankedTensorOf<[SI32]>:$output,
        RankedTensorOf<[SI32]>:$outputLength
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $sequenceLength `:` type($sequenceLength) `,` ($blankIndex^ `:` type($blankIndex))? `)`
        `->` type(results)
    }];
}

//
// PadOp
//

def EMU_PadUPAOp :
        EMU_UPATaskOp<"Pad"> {
    let summary = "Pad UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input,

        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        OptionalAttr<F64Attr>:$pad_value,
        IE_PadMode:$mode
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];
}

//
// LSTMCellUPAOp
//

def EMU_LSTMCellUPAOp :
        EMU_UPATaskOp<"LSTMCellUPA"> {
    let summary = "LSTMCell UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$inputData,
        RankedTensorOf<[F16]>:$initialHiddenState,
        RankedTensorOf<[F16]>:$initialCellState,
        RankedTensorOf<[F16]>:$weights,
        RankedTensorOf<[F16]>:$biases
    );

    let results = (outs
        RankedTensorOf<[F16]>:$outputHiddenState,
        RankedTensorOf<[F16]>:$outputCellState
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $inputData `:` type($inputData) `,` $initialHiddenState `:` type($initialHiddenState)
        `,` $initialCellState `:` type($initialCellState) `,` $weights `:` type($weights) `,` $biases `:` type($biases) `)`
        `->` type(results)
    }];
}

//
// LSTMSequenceUPAOp
//

def EMU_LSTMSequenceUPAOp :
        EMU_UPATaskOp<"LSTMSequenceUPA"> {
    let summary = "LSTMSequence UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$inputData,
        RankedTensorOf<[F16]>:$initialHiddenState,
        RankedTensorOf<[F16]>:$initialCellState,
        RankedTensorOf<[F16]>:$weights,
        RankedTensorOf<[F16]>:$biases,

        IntAttr:$sequenceLength,
        IE_RNNSequenceDirection:$direction
    );

    let results = (outs
        RankedTensorOf<[F16]>:$outputHiddenValues,
        RankedTensorOf<[F16]>:$outputCellState,
        RankedTensorOf<[F16]>:$outputHiddenState
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $inputData `:` type($inputData) `,` $initialHiddenState `:` type($initialHiddenState)
        `,` $initialCellState `:` type($initialCellState) `,` $weights `:` type($weights) `,` $biases `:` type($biases) `)`
        `->` type(results)
    }];
}

//
// InterpolateOp
//

def EMU_InterpolateUPAOp :
        EMU_UPATaskOp<"Interpolate"> {
    let summary = "Interpolate UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input,

        IE_InterpolateMode:$mode,
        IE_InterpolateCoordMode:$coord_mode,
        IE_InterpolateNearestMode:$nearest_mode,
        UnitAttr:$antialias
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];

}


//
// StridedSliceUPAOp
//

def EMU_StridedSliceUPAOp :
     EMU_UPATaskOp<"StridedSlice"> {
    let summary = "StridedSlice UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input,

        I64ArrayAttr:$begins,
        I64ArrayAttr:$ends,
        I64ArrayAttr:$strides
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];
}

//
// FullyConnectedUPAOp
//

def EMU_FullyConnectedUPAOp :
        EMU_UPATaskOp<"FullyConnectedUPA"> {
    let summary = "FullyConnected UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input,
        RankedTensorOf<[F16]>:$weights,
        Optional<RankedTensorOf<[F16]>>:$bias
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $weights `:` type($weights) (`,` $bias^ `:` type($bias))? `)`
        `->` type(results)
    }];
}

//
// RegionYoloOp
//

def EMU_RegionYoloUPAOp :
        EMU_UPATaskOp<"RegionYoloUPA"> {
    let summary = "RegionYolo UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input,

        IntAttr:$coords,
        IntAttr:$classes,
        IntAttr:$regions,
        OptionalAttr<BoolAttr>:$do_softmax,
        OptionalAttr<I64ArrayAttr>:$mask
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];
}

//
// MVNOp
//

def EMU_MVNUPAOp :
        EMU_UPATaskOp<"MVNUPA",
            [
                SameOperandsAndResultShape,
                SameOperandsAndResultElementType
            ]
        > {
    let summary = "MVN UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input,

        OptionalAttr<BoolAttr>:$across_channels,
        OptionalAttr<BoolAttr>:$normalize_variance,
        F64Attr:$eps
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let verifier = [{
        return vpux::EMU::verifyOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];
}

//
// GatherUPAOp
//

def EMU_GatherUPAOp :
        EMU_UPATaskOp<"GatherUPA"> {
    let summary = "Gather UPA SHAVE kernel";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[SI32]>:$indices,

        IntAttr:$axis
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let verifier = [{
        return vpux::EMU::verifyOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $indices `:` type($indices) `)`
        `->` type(results)
    }];
}

//
// CeilingUPAOp
//

def EMU_CeilingUPAOp :
        EMU_UPATaskOp<"CeilingUPA",
            [
                SameOperandsAndResultShape,
                SameOperandsAndResultElementType
            ]
        > {
    let summary = "Ceiling UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let verifier = [{
        return vpux::EMU::verifyPostOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];
}

//
// NormalizeIEOp
//

def EMU_NormalizeIEUPAOp :
        EMU_UPATaskOp<"NormalizeIEUPA"> {
    let summary = "NormalizeIE UPA SHAVE kernel";

    let arguments = (ins
        RankedTensorOf<[F16]>:$data,
        RankedTensorOf<[F16]>:$weights,

        F64Attr:$eps,
        BoolAttr:$across_spatial,
        BoolAttr:$channel_shared
    );

    let results = (outs
        RankedTensorOf<[F16]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $data `:` type($data) `,` $weights `:` type($weights) `)`
        `->` type(results)
    }];
}

//
// SplitOp
//

def EMU_SplitUPAOp :
        EMU_UPATaskOp<"Split"> {
    let summary = "Split UPA SHAVE kernel";

    let description = [{
        This layer although being marked as UPA will be run
        as native PC code in Emulator.
    }];

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<AnyRankedTensor>:$axis,

        IntAttr:$num_splits,
        OptionalAttr<IntAttr>:$axis_value
    );

    let results = (outs
        Variadic<AnyRankedTensor>:$outputs
    );

    let verifier = [{
        return vpux::EMU::verifyOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `input` `(` $input `:` type($input) `,` $axis `:` type($axis) `)`
        `->` type(results)
    }];
}

//
// ConcatOp
//

def EMU_ConcatUPAOp :
        EMU_UPATaskOp<"Concat"> {
    let summary = "Concat UPA SHAVE kernel";

    let description = [{
        This layer although being marked as UPA will be run
        as native PC code in Emulator.
    }];

    let arguments = (ins
        Variadic<AnyRankedTensor>:$inputs,

        OptionalAttr<IE_ConcatAttrs>:$per_axis,
        OptionalAttr<I64ArrayOfArraysAttr>:$static_offsets
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let verifier = [{
        return vpux::EMU::verifyOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $inputs `:` type($inputs) `)`
        `->` type(results)
    }];
}

//
// SliceOp
//

def EMU_SliceUPAOp :
        EMU_UPATaskOp<"Slice"> {
    let summary = "Slice UPA SHAVE kernel";

    let description = [{
        This layer although being marked as UPA will be run
        as native PC code in Emulator.
    }];

    let arguments = (ins
        AnyRankedTensor:$source,

        I64ArrayAttr:$static_offsets,
        I64ArrayAttr:$static_sizes
    );

    let results = (outs
        AnyRankedTensor:$result
    );

    let verifier = [{
        return vpux::EMU::verifyOp(*this);
    }];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $source `:` type($source) `)`
        `->` type(results)
    }];
}

#endif
