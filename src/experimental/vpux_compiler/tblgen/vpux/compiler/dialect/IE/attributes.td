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

#endif
