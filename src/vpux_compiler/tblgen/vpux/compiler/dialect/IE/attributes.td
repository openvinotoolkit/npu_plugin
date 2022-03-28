//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_IE_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_IE_ATTRIBUTES

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/dialect/IE/dialect.td"

//
// TypeComparisonMode
//

defvar IE_STRICT_EQUAL = "STRICT_EQUAL";
defvar IE_ALLOW_QUANT_MIXED_PRECISION = "ALLOW_QUANT_MIXED_PRECISION";
defvar IE_ALLOW_DIFFERENT_QUANT = "ALLOW_DIFFERENT_QUANT";
defvar IE_TypeComparisonMode_STRICT_EQUAL = !strconcat("vpux::IE::TypeComparisonMode::", IE_STRICT_EQUAL);
defvar IE_TypeComparisonMode_ALLOW_QUANT_MIXED_PRECISION = !strconcat("vpux::IE::TypeComparisonMode::", IE_ALLOW_QUANT_MIXED_PRECISION);
defvar IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT = !strconcat("vpux::IE::TypeComparisonMode::", IE_ALLOW_DIFFERENT_QUANT);

def IE_TypeComparisonMode :
        BitEnumAttr<
            "TypeComparisonMode",
            "Tensor element type comparison modes",
            [
                BitEnumAttrCase<IE_STRICT_EQUAL, 0x1>,
                BitEnumAttrCase<IE_ALLOW_QUANT_MIXED_PRECISION, 0x2>,
                BitEnumAttrCase<IE_ALLOW_DIFFERENT_QUANT, 0x4>
            ]
        > {
    let cppNamespace = "vpux::IE";
    let genSpecializedAttr = 1;
}

//
// TensorAttr
//

def IE_TensorAttr :
        StructAttr<
            "TensorAttr",
            IE_Dialect,
            [
                StructFieldAttr<"order", OptionalAttr<AffineMapAttr>>,
                StructFieldAttr<"mem_space", OptionalAttr<IndexedSymbolAttr>>,
                StructFieldAttr<"sparse", OptionalAttr<UnitAttr>>,
            ]
        >;

//
// TopKMode
//

def IE_TopKMode :
        StrEnumAttr<
            "TopKMode",
            "TopKMode that the InferenceEngine supports",
            [
                StrEnumAttrCase<"MAX">,
                StrEnumAttrCase<"MIN">,
            ]
        > {
    let cppNamespace = "vpux::IE";
    let genSpecializedAttr = 1;
}

//
// TopKSortType
//

def IE_TopKSortType :
        StrEnumAttr<
            "TopKSortType",
            "TopKSortType that the InferenceEngine supports",
            [
                StrEnumAttrCase<"NONE">,
                StrEnumAttrCase<"SORT_INDICES">,
                StrEnumAttrCase<"SORT_VALUES">,
            ]
        > {
    let cppNamespace = "vpux::IE";
    let genSpecializedAttr = 1;
}

//
// ROIPoolingMethod
//

def IE_ROIPoolingMethod :
        StrEnumAttr<
            "ROIPoolingMethod",
            "ROIPoolingMethod that the InferenceEngine supports",
            [
                StrEnumAttrCase<"max">,
                StrEnumAttrCase<"bilinear">,
            ]
        > {
    let cppNamespace = "vpux::IE";
    let genSpecializedAttr = 1;
}

//
// ROIAlignMethod
//

def IE_ROIAlignMethod :
        StrEnumAttr<
            "ROIAlignMethod",
            "ROIAlignMethod that the InferenceEngine supports",
            [
                StrEnumAttrCase<"avg">,
                StrEnumAttrCase<"max">,
            ]
        > {
    let cppNamespace = "vpux::IE";
    let genSpecializedAttr = 1;
}

//
// AutoBroadcastType
//

def IE_AutoBroadcastType :
        StrEnumAttr<
            "AutoBroadcastType",
            "Specifies rules used for auto-broadcasting of input tensors",
            [
                StrEnumAttrCase<"NONE_OR_EXPLICIT">,
                StrEnumAttrCase<"NUMPY">,
                StrEnumAttrCase<"PDPD">,
            ]
        > {
    let cppNamespace = "vpux::IE";
    let genSpecializedAttr = 1;
}

//
// BroadcastType
//

def IE_BroadcastType :
        StrEnumAttr<
            "BroadcastType",
            "Broadcast type that operations support",
            [
                StrEnumAttrCase<"NUMPY">,
                StrEnumAttrCase<"EXPLICIT">,
                StrEnumAttrCase<"BIDIRECTIONAL">,
            ]
        > {
    let cppNamespace = "vpux::IE";
    let genSpecializedAttr = 1;
}

//
// RoundingType
//

def IE_RoundingType :
        StrEnumAttr<
            "RoundingType",
            "Rounding type that operations support",
            [
                StrEnumAttrCase<"FLOOR">,
                StrEnumAttrCase<"CEIL">,
            ]
        > {
    let cppNamespace = "vpux::IE";
    let genSpecializedAttr = 1;
}

//
// LRN_IERegion
//

def IE_LRN_IERegion :
        StrEnumAttr<
            "LRN_IERegion",
            "LRN_IE region that operations support",
            [
                StrEnumAttrCase<"same">,
                StrEnumAttrCase<"across">
            ]
        > {
    let cppNamespace = "vpux::IE";
    let genSpecializedAttr = 1;
}

//
// Proposal
//

def IE_ProposalAttrs :
        StructAttr<
            "ProposalAttr",
            IE_Dialect,
            [
                StructFieldAttr<"baseSize",           IntAttr>,
                StructFieldAttr<"preNmsTopN",         IntAttr>,
                StructFieldAttr<"postNmsTopN",        IntAttr>,
                StructFieldAttr<"nmsThresh",          F64Attr>,
                StructFieldAttr<"featStride",         IntAttr>,
                StructFieldAttr<"minSize",            IntAttr>,
                StructFieldAttr<"ratio",              F64ArrayAttr>,
                StructFieldAttr<"scale",              F64ArrayAttr>,
                StructFieldAttr<"clipBeforeNms",      BoolAttr>,
                StructFieldAttr<"clipAfterNms",       BoolAttr>,
                StructFieldAttr<"normalize",          BoolAttr>,
                StructFieldAttr<"boxSizeScale",       F64Attr>,
                StructFieldAttr<"boxCoordinateScale", F64Attr>,
                StructFieldAttr<"framework",          StrAttr>,
                StructFieldAttr<"inferProbs",         BoolAttr>,
            ]
        >;

//
// Interpolate
//

def IE_InterpolateMode :
        StrEnumAttr<
            "InterpolateMode",
            "Specifies type of interpolation",
            [
                StrEnumAttrCase<"nearest">,
                StrEnumAttrCase<"linear">,
                StrEnumAttrCase<"linear_onnx">,
                StrEnumAttrCase<"cubic">,
            ]
        > {
    let cppNamespace = "vpux::IE";
    let genSpecializedAttr = 1;
}

def IE_InterpolateCalcMode :
        StrEnumAttr<
            "InterpolateCalcMode",
            "Specifies which input, sizes or scales, is used to calculate an output shape.",
            [
                StrEnumAttrCase<"sizes">,
                StrEnumAttrCase<"scales">,
            ]
        > {
    let cppNamespace = "vpux::IE";
    let genSpecializedAttr = 1;
}

def IE_InterpolateCoordMode :
        StrEnumAttr<
            "InterpolateCoordMode",
            "coordinate_transformation_mode specifies how to transform the coordinate.",
            [
                StrEnumAttrCase<"half_pixel">,
                StrEnumAttrCase<"pytorch_half_pixel">,
                StrEnumAttrCase<"asymmetric">,
                StrEnumAttrCase<"tf_half_pixel_for_nn">,
                StrEnumAttrCase<"align_corners">,
            ]
        > {
    let cppNamespace = "vpux::IE";
    let genSpecializedAttr = 1;
}

def IE_InterpolateNearestMode :
        StrEnumAttr<
            "InterpolateNearestMode",
            "specifies round mode when mode == nearest",
            [
                StrEnumAttrCase<"round_prefer_floor">,
                StrEnumAttrCase<"round_prefer_ceil">,
                StrEnumAttrCase<"floor">,
                StrEnumAttrCase<"ceil">,
                StrEnumAttrCase<"simple">,
            ]
        > {
    let cppNamespace = "vpux::IE";
    let genSpecializedAttr = 1;
}

def IE_InterpolateAttr :
        StructAttr<
            "InterpolateAttr",
            IE_Dialect,
            [
                StructFieldAttr<"mode",             IE_InterpolateMode>,
                StructFieldAttr<"shape_calc_mode",  IE_InterpolateCalcMode>,
                StructFieldAttr<"coord_mode",       IE_InterpolateCoordMode>,
                StructFieldAttr<"nearest_mode",     IE_InterpolateNearestMode>,
                StructFieldAttr<"antialias",        BoolAttr>,
                StructFieldAttr<"pads_begin",       I64ArrayAttr>,
                StructFieldAttr<"pads_end",         I64ArrayAttr>,
                StructFieldAttr<"cube_coeff",       F64Attr>,
            ]
        >;

//
// DetectionOutputAttrs
//

def IE_DetectionOutputAttrs :
        StructAttr<
            "DetectionOutputAttr",
            IE_Dialect,
            [
                StructFieldAttr<"num_classes",                IntAttr>,
                StructFieldAttr<"background_label_id",        IntAttr>,
                StructFieldAttr<"top_k",                      IntAttr>,
                StructFieldAttr<"variance_encoded_in_target", BoolAttr>,
                StructFieldAttr<"keep_top_k",                 I64ArrayAttr>,
                StructFieldAttr<"code_type",                  StrAttr>,
                StructFieldAttr<"share_location",             BoolAttr>,
                StructFieldAttr<"nms_threshold",              F64Attr>,
                StructFieldAttr<"confidence_threshold",       F64Attr>,
                StructFieldAttr<"clip_after_nms",             BoolAttr>,
                StructFieldAttr<"clip_before_nms",            BoolAttr>,
                StructFieldAttr<"decrease_label_id",          BoolAttr>,
                StructFieldAttr<"normalized",                 BoolAttr>,
                StructFieldAttr<"input_height",               IntAttr>,
                StructFieldAttr<"input_width",                IntAttr>,
                StructFieldAttr<"objectness_score",           F64Attr>,
            ]
        >;

//
// PadMode
//

def IE_PadMode :
        StrEnumAttr<
            "PadMode",
            "TPadMode that the InferenceEngine supports",
            [
                StrEnumAttrCase<"CONSTANT">,
                StrEnumAttrCase<"EDGE">,
                StrEnumAttrCase<"REFLECT">,
                StrEnumAttrCase<"SYMMETRIC">,
            ]
        > {
    let cppNamespace = "vpux::IE";
    let genSpecializedAttr = 1;
}

//
// RoundMode
//

def IE_RoundMode :
        StrEnumAttr<
            "RoundMode",
            "RoundMode that the InferenceEngine supports",
            [
                StrEnumAttrCase<"HALF_TO_EVEN">,
                StrEnumAttrCase<"HALF_AWAY_FROM_ZERO">,
            ]
        > {
    let cppNamespace = "vpux::IE";
    let genSpecializedAttr = 1;
}

//
// RNNSequenceDirection
//

def IE_RNNSequenceDirection :
        StrEnumAttr<
            "RNNSequenceDirection",
            "RNNSequenceDirection that the InferenceEngine supports",
            [
                StrEnumAttrCase<"FORWARD">,
                StrEnumAttrCase<"REVERSE">,
            ]
        > {
    let cppNamespace = "vpux::IE";
    let genSpecializedAttr = 1;
}

//
// PostOp
//

def IE_PostOp :
        StructAttr<
            "PostOp",
            IE_Dialect,
            [
                StructFieldAttr<"name",     StrAttr>,
                StructFieldAttr<"attrs",    DictionaryAttr>,
            ]
        > {
    let cppNamespace = "vpux::IE";
}

//
// ConcatAttrs
//

def IE_ConcatAttrs :
        StructAttr<
            "ConcatAttrs",
            IE_Dialect,
            [
                StructFieldAttr<"axis",     IntAttr>,
                StructFieldAttr<"offset",   OptionalAttr<IntAttr>>,
                StructFieldAttr<"stride",   OptionalAttr<IntAttr>>
            ]
        >;

//
// IE_DepthToSpaceMode
//

def IE_DepthToSpaceMode :
        StrEnumAttr<
            "DepthToSpaceMode",
            "DepthToSpaceMode that the InferenceEngine supports",
            [
                StrEnumAttrCase<"BLOCKS_FIRST">,
                StrEnumAttrCase<"DEPTH_FIRST">,
            ]
        > {
    let cppNamespace = "vpux::IE";
    let genSpecializedAttr = 1;
}

//
// IE_SpaceToDepthMode
//

def IE_SpaceToDepthMode :
        StrEnumAttr<
            "SpaceToDepthMode",
            "SpaceToDepthMode that the InferenceEngine supports",
            [
                StrEnumAttrCase<"BLOCKS_FIRST">,
                StrEnumAttrCase<"DEPTH_FIRST">,
            ]
        > {
    let cppNamespace = "vpux::IE";
    let genSpecializedAttr = 1;
    }

// Color Formats
//

def IE_ColorFmt :
        StrEnumAttr<
            "ColorFmt",
            "YUV, RGB color formats",
            [
                StrEnumAttrCase<"NV12">,
                StrEnumAttrCase<"I420">,
                StrEnumAttrCase<"RGB">,
                StrEnumAttrCase<"BGR">
            ]
        > {
    let cppNamespace = "vpux::IE";
    let genSpecializedAttr = 1;
    }

#endif
