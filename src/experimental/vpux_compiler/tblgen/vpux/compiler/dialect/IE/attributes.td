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
// Layout
//

def IE_Layout_ANY       : I32EnumAttrCase<"ANY",    0>;
def IE_Layout_SCALAR    : I32EnumAttrCase<"SCALAR", 1>;
def IE_Layout_C         : I32EnumAttrCase<"C",      2>;
def IE_Layout_NC        : I32EnumAttrCase<"NC",     3>;
def IE_Layout_CHW       : I32EnumAttrCase<"CHW",    4>;
def IE_Layout_NCHW      : I32EnumAttrCase<"NCHW",   5>;
def IE_Layout_NHWC      : I32EnumAttrCase<"NHWC",   6>;
def IE_Layout_NCDHW     : I32EnumAttrCase<"NCDHW",  7>;
def IE_Layout_NDHWC     : I32EnumAttrCase<"NDHWC",  8>;

def IE_Layout :
        I32EnumAttr<
            "Layout",
            "Layouts that the InferenceEngine supports",
            [
                IE_Layout_ANY,
                IE_Layout_SCALAR,
                IE_Layout_C,
                IE_Layout_NC,
                IE_Layout_CHW,
                IE_Layout_NCHW,
                IE_Layout_NHWC,
                IE_Layout_NCDHW,
                IE_Layout_NDHWC,
            ]
        > {
    let cppNamespace = "vpux::IE";
}


//
// AutoBroadcastType
//

def IE_AutoBroadcastType_NONE_OR_EXPLICIT : I32EnumAttrCase<"NONE_OR_EXPLICIT", 0>;
def IE_AutoBroadcastType_NUMPY            : I32EnumAttrCase<"NUMPY", 1>;
def IE_AutoBroadcastType_PDPD             : I32EnumAttrCase<"PDPD", 2>;

def IE_AutoBroadcastType :
        I32EnumAttr<
            "AutoBroadcastType",
            "Specifies rules used for auto-broadcasting of input tensors",
            [
                IE_AutoBroadcastType_NONE_OR_EXPLICIT,
                IE_AutoBroadcastType_NUMPY,
                IE_AutoBroadcastType_PDPD,
            ]
        > {
    let cppNamespace = "vpux::IE";
}

//
// Rounding type
//

def IE_RoundingType_FLOOR   : I32EnumAttrCase<"FLOOR",  0>;
def IE_RoundingType_CEIL    : I32EnumAttrCase<"CEIL",   1>;

def IE_RoundingType :
        I32EnumAttr<
            "RoundingType",
            "Rounding type that operations support",
            [
                IE_RoundingType_FLOOR,
                IE_RoundingType_CEIL
            ]
        > {
    let cppNamespace = "vpux::IE";
}

#endif
