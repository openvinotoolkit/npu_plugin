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

#ifndef VPUX_COMPILER_DIALECT_EMU_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_EMU_ATTRIBUTES

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/dialect/EMU/dialect.td"

//
// TaskType
//

def EMU_TaskType :
        StrEnumAttr<
            "TaskType",
            "The type of the relevant processing code of the VPU for the task",
            [
                StrEnumAttrCase<"NCE2">,
                StrEnumAttrCase<"UPA">,
                StrEnumAttrCase<"Generic">,
            ]
        > {
    let cppNamespace = "vpux::EMU";
    let genSpecializedAttr = 1;
}

//
// ArchKind
//

//
// MemoryLocation
//

def EMU_MemoryLocation :
        StrEnumAttr<
            "MemoryLocation",
            "Values indicating which type of memory a tensor resides in",
            [
                StrEnumAttrCase<"ProgrammableInput">,
                StrEnumAttrCase<"ProgrammableOutput">,
                StrEnumAttrCase<"VPU_DDR_Heap">,
                StrEnumAttrCase<"GraphFile">
            ]
        > {
    let cppNamespace = "vpux::EMU";
    let genSpecializedAttr = 1;
}

def EMU_ArchKind :
        StrEnumAttr<
            "ArchKind",
            "Represents VPU architecture generation",
            [
                StrEnumAttrCase<"UNKNOWN">,
                StrEnumAttrCase<"KMB">,
                StrEnumAttrCase<"TBH">,
                StrEnumAttrCase<"MTL">,
                StrEnumAttrCase<"LNL">,
            ]
        > {
    let cppNamespace = "vpux::EMU";
    let genSpecializedAttr = 1;
}

//
// PoolLayerType
//

def EMU_PoolLayerType :
        StrEnumAttr<
            "PoolLayerType",
            "Type of Pooling layer",
            [
                StrEnumAttrCase<"MAX">,
                StrEnumAttrCase<"AVG">,
            ]
        > {
    let cppNamespace = "vpux::EMU";
    let genSpecializedAttr = 1;
}

//
// EltwiseLayerType
//

def EMU_EltwiseLayerType :
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
            ]
        > {
    let cppNamespace = "vpux::EMU";
    let genSpecializedAttr = 1;
}

//
// PPEType
//

def EMU_PPELayerType :
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
    let cppNamespace = "vpux::EMU";
    let genSpecializedAttr = 1;
}

//
// NCETaskType
//

def EMU_NCETaskType :
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
    let cppNamespace = "vpux::EMU";
    let genSpecializedAttr = 1;
}

//
// CompilationMode
//

def EMU_CompilationMode  :
        StrEnumAttr<
            "CompilationMode",
            "Compilation Mode",
            [
                StrEnumAttrCase<"ReferenceSW">,
                StrEnumAttrCase<"ReferenceHW">
            ]
        > {
    let cppNamespace = "vpux::EMU";
    let genSpecializedAttr = 1;
}

//
// PaddingAttr
//

def EMU_PaddingAttr :
        StructAttr<
            "PaddingAttr",
            EMU_Dialect,
            [
                StructFieldAttr<"left",   IntAttr>,
                StructFieldAttr<"right",  IntAttr>,
                StructFieldAttr<"top",    IntAttr>,
                StructFieldAttr<"bottom", IntAttr>,
            ]
        >;

#endif
