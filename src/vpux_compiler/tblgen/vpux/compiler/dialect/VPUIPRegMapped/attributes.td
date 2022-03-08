//
// Copyright (C) 2022 Intel Corporation.
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

#ifndef VPUX_COMPILER_DIALECT_VPUIPREGMAPPED_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_VPUIPREGMAPPED_ATTRIBUTES

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/dialect/VPUIPRegMapped/dialect.td"


//
// TaskType
//

def VPUIPRegMapped_TaskType :
        StrEnumAttr<
            "TaskType",
            "The type of the relevant processing code of the VPU for the task",
            [
                StrEnumAttrCase<"UPADMA">,
                StrEnumAttrCase<"NNDMA">,
                StrEnumAttrCase<"NCE2">,
                StrEnumAttrCase<"Controller">,
                StrEnumAttrCase<"UPA">,
                StrEnumAttrCase<"SNN">,
                StrEnumAttrCase<"Generic">,
            ]
        > {
    let cppNamespace = "vpux::VPUIPRegMapped";
    let genSpecializedAttr = 1;
}

//
// PPEType
//

def VPUIPRegMapped_PPELayerType :
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
    let cppNamespace = "vpux::VPUIPRegMapped";
    let genSpecializedAttr = 1;
}

//
// MPEMode
//

def VPUIPRegMapped_MPEMode :
        StrEnumAttr<
            "MPEMode",
            "MPE Mode",
            [
                StrEnumAttrCase<"VECTOR">,
                StrEnumAttrCase<"MATRIX">,
                StrEnumAttrCase<"VECTOR_FP16">,
                StrEnumAttrCase<"CUBOID_16x16">,
                StrEnumAttrCase<"CUBOID_8x16">,
                StrEnumAttrCase<"NOP">
            ]
        > {
    let cppNamespace = "vpux::VPUIPRegMapped";
    let genSpecializedAttr = 1;
}
#endif
