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

#ifndef VPUX_COMPILER_DIALECT_IE_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_IE_ATTRIBUTES

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/dialect/IE/dialect.td"

//
// TensorAttr
//

def IE_TensorAttr :
        StructAttr<
            "TensorAttr",
            IE_Dialect,
            [
                StructFieldAttr<"order", OptionalAttr<AffineMapAttr>>,
            ]
        >;

//
// EpsMode
//

def IE_EpsMode :
        StrEnumAttr<
            "EpsMode",
            "EpsMode that the InferenceEngine supports",
            [
                StrEnumAttrCase<"ADD">,
                StrEnumAttrCase<"MAX">,
            ]
        > {
    let cppNamespace = "vpux::IE";
    let genSpecializedAttr = 1;
}

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
// PostOpKind
//

def IE_PostOpKind :
        StrEnumAttr<
            "PostOpKind",
            "Operations that can be performed during post-processing",
            [
                // Activations
                StrEnumAttrCase<"RELU">,
                StrEnumAttrCase<"CLAMP">,
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
                StructFieldAttr<"kind",   IE_PostOpKind>,
                StructFieldAttr<"params", DictionaryAttr>,
            ]
        > {
    let cppNamespace = "vpux::IE";
}

#endif
