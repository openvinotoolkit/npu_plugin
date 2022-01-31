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

#ifndef VPUX_COMPILER_DIALECT_VPUIP_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_VPUIP_ATTRIBUTES

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/dialect/VPUIP/dialect.td"

//
// PoolLayerType
//

def VPUIP_PoolLayerType :
        StrEnumAttr<
            "PoolLayerType",
            "Type of Pooling layer",
            [
                StrEnumAttrCase<"MAX">,
                StrEnumAttrCase<"AVG">,
            ]
        > {
    let cppNamespace = "vpux::VPUIP";
    let genSpecializedAttr = 1;
}

//
// ReduceLayerType
//

def VPUIP_ReduceLayerType :
        StrEnumAttr<
            "ReduceLayerType",
            "Type of Reduce layer",
            [
                StrEnumAttrCase<"MAX">,
                StrEnumAttrCase<"MEAN">,
                StrEnumAttrCase<"SUM">,
            ]
        > {
    let cppNamespace = "vpux::VPUIP";
    let genSpecializedAttr = 1;
}

//
// NCETaskType
//

def VPUIP_NCETaskType :
        StrEnumAttr<
            "NCETaskType",
            "NCE task type",
            [
                StrEnumAttrCase<"CONV">,
                StrEnumAttrCase<"DWCONV">,
                StrEnumAttrCase<"MAXPOOL">,
                StrEnumAttrCase<"ADA_AVG_POOL">,
                StrEnumAttrCase<"AVEPOOL">,
                StrEnumAttrCase<"FCL">,
                StrEnumAttrCase<"ELTWISE">,
                StrEnumAttrCase<"IDENTITY">,
                StrEnumAttrCase<"CMCONV">
            ]
        > {
    let cppNamespace = "vpux::VPUIP";
    let genSpecializedAttr = 1;
}

#endif
