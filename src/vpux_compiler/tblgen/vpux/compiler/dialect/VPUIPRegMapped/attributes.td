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
// PhysicalProcessor
//

def VPUIPRegMapped_PhysicalProcessor :
        StrEnumAttr<
            "PhysicalProcessor",
            "Enum for describing processor resources",
            [
                StrEnumAttrCase<"ARM">,
                StrEnumAttrCase<"Leon_RT">,
                StrEnumAttrCase<"Leon_NN">,
                StrEnumAttrCase<"SHAVE_UPA">,
                StrEnumAttrCase<"SHAVE_NN">,
                StrEnumAttrCase<"NCE_Cluster">,
                StrEnumAttrCase<"NCE_PerClusterDPU">,
            ]
        > {
    let cppNamespace = "vpux::VPUIPRegMapped";
    let genSpecializedAttr = 1;
}

//
// DMAEngine
//

def VPUIPRegMapped_DMAEngine :
        StrEnumAttr<
            "DMAEngine",
            "DMA engine kind",
            [
                StrEnumAttrCase<"DMA_UPA">,
                StrEnumAttrCase<"DMA_NN">,
            ]
        > {
    let cppNamespace = "vpux::VPUIPRegMapped";
    let genSpecializedAttr = 1;
}

//
// PhysicalMemory
//

def VPUIPRegMapped_PhysicalMemory :
        StrEnumAttr<
            "PhysicalMemory",
            "Represents the actual hardware memory hierarchy",
            [
                StrEnumAttrCase<"DDR">,
                StrEnumAttrCase<"CSRAM">,
                StrEnumAttrCase<"CMX_UPA">,
                StrEnumAttrCase<"CMX_NN">,
                StrEnumAttrCase<"Register">,
            ]
        > {
    let cppNamespace = "vpux::VPUIPRegMapped";
    let genSpecializedAttr = 1;
}

//
// Version
//

def VPUIPRegMapped_Version :
        StructAttr<
            "VersionAttr",
            VPUIPRegMapped_Dialect,
            [
                StructFieldAttr<"majorV",     IntAttr>,
                StructFieldAttr<"minorV",     IntAttr>,
                StructFieldAttr<"patchV",     IntAttr>,
                StructFieldAttr<"hash",       StrAttr>,
                StructFieldAttr<"contextStr", StrAttr>,
            ]
        >;

//
// ExecutionFlag
//

def VPUIPRegMapped_ExecutionFlag:
        BitEnumAttr<
            "ExecutionFlag",
            "Each of these enums' presence informs how the current schedule is configured",
            [
                BitEnumAttrCase<"NONE",             0x0000>,
                BitEnumAttrCase<"DynamicBarriers",  0x0001>,
            ]
        > {
    let cppNamespace = "vpux::VPUIPRegMapped";
}

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
// MemoryLocation
//

def VPUIPRegMapped_MemoryLocation :
        StrEnumAttr<
            "MemoryLocation",
            "Values indicating which type of memory a tensor resides in",
            [
                StrEnumAttrCase<"ProgrammableInput">,
                StrEnumAttrCase<"ProgrammableOutput">,
                StrEnumAttrCase<"VPU_DDR_Heap">,
                StrEnumAttrCase<"GraphFile">,
                StrEnumAttrCase<"VPU_CMX_NN">,
                StrEnumAttrCase<"VPU_CMX_UPA">,
                StrEnumAttrCase<"VPU_DDR_BSS">,
                StrEnumAttrCase<"VPU_CSRAM">,
                StrEnumAttrCase<"AbsoluteAddr">,
                StrEnumAttrCase<"MAC_Accumulators">,
            ]
        > {
    let cppNamespace = "vpux::VPUIPRegMapped";
    let genSpecializedAttr = 1;
}

//
// ArchKind
//

def VPUIPRegMapped_ArchKind :
        StrEnumAttr<
            "ArchKind",
            "Represents VPU architecture generation",
            [
                StrEnumAttrCase<"KMB">,
                StrEnumAttrCase<"TBH">,
                StrEnumAttrCase<"MTL">,
            ]
        > {
    let cppNamespace = "vpux::VPUIPRegMapped";
    let genSpecializedAttr = 1;
}

//
// PoolLayerType
//

def VPUIPRegMapped_PoolLayerType :
        StrEnumAttr<
            "PoolLayerType",
            "Type of Pooling layer",
            [
                StrEnumAttrCase<"MAX">,
                StrEnumAttrCase<"AVG">,
            ]
        > {
    let cppNamespace = "vpux::VPUIPRegMapped";
    let genSpecializedAttr = 1;
}

//
// EltwiseLayerType
//

def VPUIPRegMapped_EltwiseLayerType :
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
// NCETaskType
//

def VPUIPRegMapped_NCETaskType :
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
    let cppNamespace = "vpux::VPUIPRegMapped";
    let genSpecializedAttr = 1;
}

//
// CompilationMode
//

def VPUIPRegMapped_CompilationMode  :
        StrEnumAttr<
            "CompilationMode",
            "Compilation Mode",
            [
                StrEnumAttrCase<"ReferenceSW">,
                StrEnumAttrCase<"ReferenceHW">
            ]
        > {
    let cppNamespace = "vpux::VPUIPRegMapped";
    let genSpecializedAttr = 1;
}

//
// PaddingAttr
//

def VPUIPRegMapped_PaddingAttr :
        StructAttr<
            "PaddingAttr",
            VPUIPRegMapped_Dialect,
            [
                StructFieldAttr<"left",   IntAttr>,
                StructFieldAttr<"right",  IntAttr>,
                StructFieldAttr<"top",    IntAttr>,
                StructFieldAttr<"bottom", IntAttr>,
            ]
        >;

//
// eDSL DMA
//

def VPUIPRegMapped_EdslDMAStage :
        StrEnumAttr<
            "EdslDMAStage",
            "eDSL DMA stage",
            [
                StrEnumAttrCase<"NONE">,
                StrEnumAttrCase<"OUTER">,
                StrEnumAttrCase<"MIDDLE">,
                StrEnumAttrCase<"ALL">,
            ]
        > {
    let cppNamespace = "vpux::VPUIPRegMapped";
    let genSpecializedAttr = 1;
}

def VPUIPRegMapped_EdslDMADirection :
        StrEnumAttr<
            "EdslDMADirection",
            "eDSL DMA direction",
            [
                StrEnumAttrCase<"NONE">,
                StrEnumAttrCase<"IN">,
                StrEnumAttrCase<"OUT">,
                StrEnumAttrCase<"INOUT">,
            ]
        > {
    let cppNamespace = "vpux::VPUIPRegMapped";
    let genSpecializedAttr = 1;
}

def VPUIPRegMapped_EdslDMADesc :
        StructAttr<
            "VPUIPRegMapped_EdslDMADesc",
            VPUIPRegMapped_Dialect,
            [
                StructFieldAttr<"baseMap", AffineMapAttr>,
                StructFieldAttr<"stage", VPUIPRegMapped_EdslDMAStage>,
                StructFieldAttr<"dir", VPUIPRegMapped_EdslDMADirection>
            ]
        > {
    let cppNamespace = "vpux::VPUIPRegMapped";
}

def VPUIPRegMapped_EdslDMADescArrayAttr : TypedArrayAttrBase<VPUIPRegMapped_EdslDMADesc,
                                      "eDSL DMA descriptor array attribute">;

#endif
