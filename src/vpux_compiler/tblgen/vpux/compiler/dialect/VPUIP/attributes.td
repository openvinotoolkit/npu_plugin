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
                StrEnumAttrCase<"SUM">,
            ]
        > {
    let cppNamespace = "vpux::VPUIP";
    let genSpecializedAttr = 1;
}

//
// EltwiseLayerType
//

def VPUIP_EltwiseLayerType :
        StrEnumAttr<
            "EltwiseLayerType",
            "Type of Eltwise layer",
            [
                StrEnumAttrCase<"ADD">,
                StrEnumAttrCase<"MULTIPLY">,
                StrEnumAttrCase<"DIVIDE">,
                StrEnumAttrCase<"SQUARED_DIFF">,
                StrEnumAttrCase<"POWER">,
                StrEnumAttrCase<"FLOOR_MOD">,
                StrEnumAttrCase<"MIN">,
                StrEnumAttrCase<"MAX">,
                StrEnumAttrCase<"AND">,
                StrEnumAttrCase<"EQUAL">,
                StrEnumAttrCase<"LESS">,
                StrEnumAttrCase<"LESS_EQUAL">,
                StrEnumAttrCase<"NOT_EQUAL">,
                StrEnumAttrCase<"GREATER">,
                StrEnumAttrCase<"GREATER_EQUAL">,
                StrEnumAttrCase<"LOGICAL_OR">,
                StrEnumAttrCase<"LOGICAL_XOR">,
            ]
        > {
    let cppNamespace = "vpux::VPUIP";
    let genSpecializedAttr = 1;
}

//
// PPEType
//

def VPUIP_PPELayerType :
        StrEnumAttr<
            "PPELayerType",
            "Post Processing Element Type",
            [
                // Low-level instructions
                StrEnumAttrCase<"STORE">,
                StrEnumAttrCase<"LOAD">,
                StrEnumAttrCase<"CLEAR">,
                StrEnumAttrCase<"NOOP">,
                StrEnumAttrCase<"HALT">,

                // Element-Wise Operations
                StrEnumAttrCase<"ADD">,
                StrEnumAttrCase<"SUB">,
                StrEnumAttrCase<"MULT">,
                StrEnumAttrCase<"MAXIMUM">,
                StrEnumAttrCase<"MINIMUM">,
                StrEnumAttrCase<"AND">,
                StrEnumAttrCase<"OR">,
                StrEnumAttrCase<"XOR">,

                // Activations
                StrEnumAttrCase<"LRELU">,
                StrEnumAttrCase<"LRELUX">,
                StrEnumAttrCase<"LPRELU">,
                StrEnumAttrCase<"CEIL">,
                StrEnumAttrCase<"FLOOR">,
                StrEnumAttrCase<"POW">,
                StrEnumAttrCase<"EXP">,
                StrEnumAttrCase<"SIGMOID">,
                StrEnumAttrCase<"TANH">,
                StrEnumAttrCase<"SQRT">,
                StrEnumAttrCase<"RSQRT">,
                StrEnumAttrCase<"FLEXARB">,
                StrEnumAttrCase<"NOT">,
                StrEnumAttrCase<"ABS">,
                StrEnumAttrCase<"NEG">
            ]
        > {
    let cppNamespace = "vpux::VPUIP";
    let genSpecializedAttr = 1;
}

//
// MPEMode
//

def VPUIP_MPEMode :
        StrEnumAttr<
            "MPEMode",
            "MPE Mode",
            [
                StrEnumAttrCase<"VECTOR">,
                StrEnumAttrCase<"MATRIX">,
                StrEnumAttrCase<"VECTOR_FP16">,
                StrEnumAttrCase<"CUBOID_16x16">,
                StrEnumAttrCase<"CUBOID_8x16">,
                StrEnumAttrCase<"CUBOID_4x16">,
                StrEnumAttrCase<"NOP">
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
// PaddingAttr
//

def VPUIP_PaddingAttr :
        StructAttr<
            "PaddingAttr",
            VPUIP_Dialect,
            [
                StructFieldAttr<"left",   IntAttr>,
                StructFieldAttr<"right",  IntAttr>,
                StructFieldAttr<"top",    IntAttr>,
                StructFieldAttr<"bottom", IntAttr>,
            ]
        >;

#endif
