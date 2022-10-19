//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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
                StrEnumAttrCase<"LOGICAL_OR">,
                StrEnumAttrCase<"LOGICAL_AND">,
                StrEnumAttrCase<"PROD">,
                StrEnumAttrCase<"SUM">,
                StrEnumAttrCase<"MIN">,
                StrEnumAttrCase<"L1">,
                StrEnumAttrCase<"L2">,
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

//
// MemRefAttr
//

def VPUIP_MemRefAttr :
        StructAttr<
            "MemRefAttr",
            VPUIP_Dialect,
            [
                StructFieldAttr<"order", AffineMapAttr>,
                StructFieldAttr<"strides", I64ArrayAttr>,
            ]
        >;

#endif
