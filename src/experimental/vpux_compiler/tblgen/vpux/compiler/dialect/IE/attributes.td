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
// Base class
//

class IE_Attr<Pred pred, string descr> :
        DialectAttr<
            IE_Dialect,
            pred,
            descr
        >;

//
// Layout
//

def IE_Layout_ANY       : StrEnumAttrCase<"ANY">;
def IE_Layout_SCALAR    : StrEnumAttrCase<"SCALAR">;
def IE_Layout_C         : StrEnumAttrCase<"C">;
def IE_Layout_NC        : StrEnumAttrCase<"NC">;
def IE_Layout_CHW       : StrEnumAttrCase<"CHW">;
def IE_Layout_NCHW      : StrEnumAttrCase<"NCHW">;
def IE_Layout_NHWC      : StrEnumAttrCase<"NHWC">;
def IE_Layout_NCDHW     : StrEnumAttrCase<"NCDHW">;
def IE_Layout_NDHWC     : StrEnumAttrCase<"NDHWC">;

def IE_LayoutEnum :
        StrEnumAttr<
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

def IE_LayoutAttr :
        IE_Attr<
            CPred<"$_self.isa<vpux::IE::LayoutAttr>()">,
            "Layouts that the InferenceEngine supports"
        > {
    let storageType = "vpux::IE::LayoutAttr";
    let returnType = "vpux::IE::Layout";

    let constBuilderCall = "vpux::IE::LayoutAttr::get($_builder.getContext(), $0)";
}

#endif
