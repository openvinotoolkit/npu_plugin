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
// PhysicalProcessor
//

def VPUIP_PhysicalProcessor :
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
    let cppNamespace = "vpux::VPUIP";
    let genSpecializedAttr = 1;
}

//
// DMAEngine
//

def VPUIP_DMAEngine :
        StrEnumAttr<
            "DMAEngine",
            "DMA engine kind",
            [
                StrEnumAttrCase<"DMA_UPA">,
                StrEnumAttrCase<"DMA_NN">,
            ]
        > {
    let cppNamespace = "vpux::VPUIP";
    let genSpecializedAttr = 1;
}

//
// PhysicalMemory
//

def VPUIP_PhysicalMemory :
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
    let cppNamespace = "vpux::VPUIP";
    let genSpecializedAttr = 1;
}

//
// Version
//

def VPUIP_Version :
        StructAttr<
            "VersionAttr",
            VPUIP_Dialect,
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

def VPUIP_ExecutionFlag:
        BitEnumAttr<
            "ExecutionFlag",
            "Each of these enums' presence informs how the current schedule is configured",
            [
                BitEnumAttrCase<"NONE",             0x0000>,
                BitEnumAttrCase<"DynamicBarriers",  0x0001>,
            ]
        > {
    let cppNamespace = "vpux::VPUIP";
}

//
// TaskType
//

def VPUIP_TaskType :
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
    let cppNamespace = "vpux::VPUIP";
    let genSpecializedAttr = 1;
}

//
// MemoryLocation
//

def VPUIP_MemoryLocation :
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
            ]
        > {
    let cppNamespace = "vpux::VPUIP";
    let genSpecializedAttr = 1;
}

//
// ArchKind
//

def VPUIP_ArchKind :
        StrEnumAttr<
            "ArchKind",
            "Represents VPU architecture generation",
            [
                StrEnumAttrCase<"KMB">,
                StrEnumAttrCase<"TBH">,
                StrEnumAttrCase<"MTL">,
            ]
        > {
    let cppNamespace = "vpux::VPUIP";
    let genSpecializedAttr = 1;
}

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
// CompilationMode
//

def VPUIP_CompilationMode  :
        StrEnumAttr<
            "CompilationMode",
            "Compilation Mode",
            [
                StrEnumAttrCase<"ReferenceSW">,
                StrEnumAttrCase<"ReferenceHW">
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

//
// eDSL DMA
//

def VPUIP_EdslDMAStage :
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
    let cppNamespace = "vpux::VPUIP";
    let genSpecializedAttr = 1;
}

def VPUIP_EdslDMADirection :
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
    let cppNamespace = "vpux::VPUIP";
    let genSpecializedAttr = 1;
}

def VPUIP_EdslDMADesc :
        StructAttr<
            "VPUIP_EdslDMADesc",
            VPUIP_Dialect,
            [
                StructFieldAttr<"baseMap", AffineMapAttr>,
                StructFieldAttr<"stage", VPUIP_EdslDMAStage>,
                StructFieldAttr<"dir", VPUIP_EdslDMADirection>
            ]
        > {
    let cppNamespace = "vpux::VPUIP";
}

def VPUIP_EdslDMADescArrayAttr : TypedArrayAttrBase<VPUIP_EdslDMADesc,
                                      "eDSL DMA descriptor array attribute">;

#endif
