//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_VPUIPREGMAPPED_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_VPUIPREGMAPPED_ATTRIBUTES

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/dialect/VPUIPRegMapped/dialect.td"

class VPUIPRegMapped_Attr<string name, list<Trait> traits = []> : AttrDef<VPUIPRegMapped_Dialect, name, traits> {
    let mnemonic = name;
}

//
// TaskType
//

def VPUIPRegMapped_TaskType :
        StrEnumAttr<
            "TaskType",
            "The type of the relevant processing code of the VPU for the task",
            [
                StrEnumAttrCase<"DMA">,
                StrEnumAttrCase<"ActKernelInvocation">,
                StrEnumAttrCase<"DPUInvariant">,
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

//
// RegisterFieldAttr
//

def VPUIPRegMapped_RegisterFieldAttr : VPUIPRegMapped_Attr<"RegisterField"> {
    let description = "This object represents closely a RegisterField Attr";
    let cppNamespace = "vpux::VPUIPRegMapped";

    let parameters = (ins "vpux::VPUIPRegMapped::RegFieldType":$regField);

    let genVerifyDecl = 1;
}

//
// RegisterFieldArrayAttr
//

def VPUIPRegMapped_RegisterFieldArrayAttr : TypedArrayAttrBase<VPUIPRegMapped_RegisterFieldAttr,
    "array of RegisterFields">{
        string cppType = "::mlir::ArrayAttr";
    }

//
// RegisterAttr
//

def VPUIPRegMapped_RegisterAttr : VPUIPRegMapped_Attr<"Register"> {
    let description = "This object represents closely a Register Attr";
    let cppNamespace = "vpux::VPUIPRegMapped";

    let parameters = (ins "vpux::VPUIPRegMapped::RegisterType":$reg);

    let genVerifyDecl = 1;
}

//
// RegisterArrayAttr
//

def VPUIPRegMapped_RegisterArrayAttr : TypedArrayAttrBase<VPUIPRegMapped_RegisterAttr,
    "array of Registers">{
        string cppType = "::mlir::ArrayAttr";
    }

//
// RegisterMappedAttr
//

def VPUIPRegMapped_RegisterMapped : VPUIPRegMapped_Attr<"RegisterMapped"> {
    let description = "This object represents closely a RegisterMapped Attr";
    let cppNamespace = "vpux::VPUIPRegMapped";

    let parameters = (ins "vpux::VPUIPRegMapped::RegMappedType":$regMapped);

    let genVerifyDecl = 1;
}

#endif
