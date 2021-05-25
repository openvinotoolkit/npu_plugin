//
// Copyright 2020 Intel Corporation.
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

include "vpux/compiler/dialect/IE/dialect.td"

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
                StructFieldAttr<"baseSize",           I32Attr>,
                StructFieldAttr<"preNmsTopN",         I32Attr>,
                StructFieldAttr<"postNmsTopN",        I32Attr>,
                StructFieldAttr<"nmsThresh",          DefaultValuedAttr<F32Attr, "0.0f">>,
                StructFieldAttr<"featStride",         DefaultValuedAttr<I32Attr, "1">>,
                StructFieldAttr<"minSize",            DefaultValuedAttr<I32Attr, "1">>,
                StructFieldAttr<"ratio",              F32ArrayAttr>,
                StructFieldAttr<"scale",              F32ArrayAttr>,
                StructFieldAttr<"clipBeforeNms",      DefaultValuedAttr<BoolAttr, "true">>,
                StructFieldAttr<"clipAfterNms",       DefaultValuedAttr<BoolAttr, "false">>,
                StructFieldAttr<"normalize",          DefaultValuedAttr<BoolAttr, "false">>,
                StructFieldAttr<"boxSizeScale",       DefaultValuedAttr<F32Attr,  "1.0f">>,
                StructFieldAttr<"boxCoordinateScale", DefaultValuedAttr<F32Attr,  "1.0f">>,
                StructFieldAttr<"framework",          DefaultValuedAttr<StrAttr,  "">>,
                StructFieldAttr<"inferProbs",         DefaultValuedAttr<BoolAttr, "false">>,
            ]
        >;

//
// Interpolate
//

def IE_InterpolateMode :
        StrEnumAttr<
            "InterpolateMode",
            "Interpolation mode",
            [
                StrEnumAttrCase<"nearest">,
                StrEnumAttrCase<"linear">,
                StrEnumAttrCase<"cubic">,
                StrEnumAttrCase<"area">,
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
                StructFieldAttr<"axes",           I64ArrayAttr>,
                StructFieldAttr<"mode",           IE_InterpolateMode>,
                StructFieldAttr<"align_corners",  DefaultValuedAttr<BoolAttr, "true">>,
                StructFieldAttr<"antialias",      DefaultValuedAttr<BoolAttr, "false">>,
                StructFieldAttr<"pads_begin",     I32ArrayAttr>,
                StructFieldAttr<"pads_end",       I32ArrayAttr>,
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
                StructFieldAttr<"num_classes",                I32Attr>,
                StructFieldAttr<"background_label_id",        DefaultValuedAttr<I32Attr, "0">>,
                StructFieldAttr<"top_k",                      DefaultValuedAttr<I32Attr, "-1">>,
                StructFieldAttr<"variance_encoded_in_target", DefaultValuedAttr<BoolAttr, "false">>,
                StructFieldAttr<"keep_top_k",                 I32ArrayAttr>,
                StructFieldAttr<"code_type",                  DefaultValuedAttr<StrAttr,  "">>,
                StructFieldAttr<"share_location",             DefaultValuedAttr<BoolAttr, "false">>,
                StructFieldAttr<"nms_threshold",              DefaultValuedAttr<F32Attr,  "0.0f">>,
                StructFieldAttr<"confidence_threshold",       DefaultValuedAttr<F32Attr,  "0.0f">>,
                StructFieldAttr<"clip_after_nms",             DefaultValuedAttr<BoolAttr, "false">>,
                StructFieldAttr<"clip_before_nms",            DefaultValuedAttr<BoolAttr, "false">>,
                StructFieldAttr<"decrease_label_id",          DefaultValuedAttr<BoolAttr, "false">>,
                StructFieldAttr<"normalized",                 DefaultValuedAttr<BoolAttr, "false">>,
                StructFieldAttr<"input_height",               DefaultValuedAttr<UI32Attr, "1">>,
                StructFieldAttr<"input_width",                DefaultValuedAttr<UI32Attr, "1">>,
                StructFieldAttr<"objectness_score",           DefaultValuedAttr<F32Attr,  "0.0f">>,
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
#endif
