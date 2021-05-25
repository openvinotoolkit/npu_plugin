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

#ifndef VPUX_COMPILER_DIALECT_VPUIP_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_VPUIP_ATTRIBUTES

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
                StructFieldAttr<"majorV",     I32Attr>,
                StructFieldAttr<"minorV",     I32Attr>,
                StructFieldAttr<"patchV",     I32Attr>,
                StructFieldAttr<"hash",       DefaultValuedAttr<StrAttr, "">>,
                StructFieldAttr<"contextStr", DefaultValuedAttr<StrAttr, "">>,
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
                StrEnumAttrCase<"VPU3400_A0">, // Keem bay A0
                StrEnumAttrCase<"VPU3400">,    // Keem bay B0
                StrEnumAttrCase<"VPU3700">,    // Keem bay B0
                StrEnumAttrCase<"VPU3900">,    // Thunder bay harbor A0
                StrEnumAttrCase<"VPU3720">,    // Meteor lake
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
