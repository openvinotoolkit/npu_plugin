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

#endif
