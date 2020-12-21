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
// AutoBroadcastType
//

def IE_AutoBroadcastType_NONE_OR_EXPLICIT : StrEnumAttrCase<"NONE_OR_EXPLICIT">;
def IE_AutoBroadcastType_NUMPY            : StrEnumAttrCase<"NUMPY">;
def IE_AutoBroadcastType_PDPD             : StrEnumAttrCase<"PDPD">;

def IE_AutoBroadcastType :
        StrEnumAttr<
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

def IE_RoundingType_FLOOR   : StrEnumAttrCase<"FLOOR">;
def IE_RoundingType_CEIL    : StrEnumAttrCase<"CEIL">;

def IE_RoundingType :
        StrEnumAttr<
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
