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

def VPUIP_PhysicalProcessor_ARM                  : StrEnumAttrCase<"ARM">;
def VPUIP_PhysicalProcessor_Leon_RT              : StrEnumAttrCase<"Leon_RT">;
def VPUIP_PhysicalProcessor_Leon_NN              : StrEnumAttrCase<"Leon_NN">;
def VPUIP_PhysicalProcessor_SHAVE_UPA            : StrEnumAttrCase<"SHAVE_UPA">;
def VPUIP_PhysicalProcessor_SHAVE_NN             : StrEnumAttrCase<"SHAVE_NN">;
def VPUIP_PhysicalProcessor_NCE_Cluster          : StrEnumAttrCase<"NCE_Cluster">;
def VPUIP_PhysicalProcessor_NCE_PerClusterDPU    : StrEnumAttrCase<"NCE_PerClusterDPU">;

def VPUIP_PhysicalProcessor :
        StrEnumAttr<
            "PhysicalProcessor",
            "Enum for describing processor resources",
            [
                VPUIP_PhysicalProcessor_ARM,
                VPUIP_PhysicalProcessor_Leon_RT,
                VPUIP_PhysicalProcessor_Leon_NN,
                VPUIP_PhysicalProcessor_SHAVE_UPA,
                VPUIP_PhysicalProcessor_SHAVE_NN,
                VPUIP_PhysicalProcessor_NCE_Cluster,
                VPUIP_PhysicalProcessor_NCE_PerClusterDPU,
            ]
        > {
    let cppNamespace = "vpux::VPUIP";
}

//
// DMAEngine
//

def VPUIP_DMAEngine_UPA  : StrEnumAttrCase<"DMA_UPA">;
def VPUIP_DMAEngine_NN   : StrEnumAttrCase<"DMA_NN">;

def VPUIP_DMAEngine :
        StrEnumAttr<
            "DMAEngine",
            "DMA engine kind",
            [
                VPUIP_DMAEngine_UPA,
                VPUIP_DMAEngine_NN,
            ]
        > {
    let cppNamespace = "vpux::VPUIP";
}

//
// PhysicalMemory
//

def VPUIP_PhysicalMemory_DDR     : StrEnumAttrCase<"DDR">;
def VPUIP_PhysicalMemory_CSRAM   : StrEnumAttrCase<"CSRAM">;
def VPUIP_PhysicalMemory_CMX_UPA : StrEnumAttrCase<"CMX_UPA">;
def VPUIP_PhysicalMemory_CMX_NN  : StrEnumAttrCase<"CMX_NN">;

def VPUIP_PhysicalMemory :
        StrEnumAttr<
            "PhysicalMemory",
            "Represents the actual hardware memory hierarchy",
            [
                VPUIP_PhysicalMemory_DDR,
                VPUIP_PhysicalMemory_CSRAM,
                VPUIP_PhysicalMemory_CMX_UPA,
                VPUIP_PhysicalMemory_CMX_NN,
            ]
        > {
    let cppNamespace = "vpux::VPUIP";
}

//
// Version
//

def VPUIP_Version_majorV        : StructFieldAttr<"majorV",     I32Attr>;
def VPUIP_Version_minorV        : StructFieldAttr<"minorV",     I32Attr>;
def VPUIP_Version_patchV        : StructFieldAttr<"patchV",     I32Attr>;
def VPUIP_Version_hash          : StructFieldAttr<"hash",       DefaultValuedAttr<StrAttr, "">>;
def VPUIP_Version_contextStr    : StructFieldAttr<"contextStr", DefaultValuedAttr<StrAttr, "">>;

def VPUIP_Version :
        StructAttr<
            "VersionAttr",
            VPUIP_Dialect,
            [
                VPUIP_Version_majorV,
                VPUIP_Version_minorV,
                VPUIP_Version_patchV,
                VPUIP_Version_hash,
                VPUIP_Version_contextStr,
            ]
        >;

//
// ExecutionFlag
//

def VPUIP_ExecutionFlag_NONE :              BitEnumAttrCase<"NONE",             0x0000>;
def VPUIP_ExecutionFlag_DynamicBarriers :   BitEnumAttrCase<"DynamicBarriers",  0x0001>;

def VPUIP_ExecutionFlag:
        BitEnumAttr<
            "ExecutionFlag",
            "Each of these enums' presence informs how the current schedule is configured",
            [
                VPUIP_ExecutionFlag_NONE,
                VPUIP_ExecutionFlag_DynamicBarriers,
            ]
        > {
    let cppNamespace = "vpux::VPUIP";
}

//
// TaskType
//

class VPUIP_TaskTypeVal<string sym> : StrEnumAttrCase<sym> {
    string enumVal = "vpux::VPUIP::TaskType::" # sym;
}

def VPUIP_TaskType_UPADMA       : VPUIP_TaskTypeVal<"UPADMA">;
def VPUIP_TaskType_NNDMA        : VPUIP_TaskTypeVal<"NNDMA">;
def VPUIP_TaskType_NCE2         : VPUIP_TaskTypeVal<"NCE2">;
def VPUIP_TaskType_Controller   : VPUIP_TaskTypeVal<"Controller">;
def VPUIP_TaskType_UPA          : VPUIP_TaskTypeVal<"UPA">;
def VPUIP_TaskType_SNN          : VPUIP_TaskTypeVal<"SNN">;
def VPUIP_TaskType_Generic      : VPUIP_TaskTypeVal<"Generic">;

def VPUIP_TaskType :
        StrEnumAttr<
            "TaskType",
            "The type of the relevant processing code of the VPU for the task",
            [
                VPUIP_TaskType_UPADMA,
                VPUIP_TaskType_NNDMA,
                VPUIP_TaskType_NCE2,
                VPUIP_TaskType_Controller,
                VPUIP_TaskType_UPA,
                VPUIP_TaskType_SNN,
                VPUIP_TaskType_Generic,
            ]
        > {
    let cppNamespace = "vpux::VPUIP";
}

//
// MemoryLocation
//

def VPUIP_MemoryLocation_ProgrammableInput  : StrEnumAttrCase<"ProgrammableInput">;
def VPUIP_MemoryLocation_ProgrammableOutput : StrEnumAttrCase<"ProgrammableOutput">;
def VPUIP_MemoryLocation_VPU_DDR_Heap       : StrEnumAttrCase<"VPU_DDR_Heap">;
def VPUIP_MemoryLocation_GraphFile          : StrEnumAttrCase<"GraphFile">;
def VPUIP_MemoryLocation_VPU_CMX_NN         : StrEnumAttrCase<"VPU_CMX_NN">;
def VPUIP_MemoryLocation_VPU_CMX_UPA        : StrEnumAttrCase<"VPU_CMX_UPA">;
def VPUIP_MemoryLocation_VPU_DDR_BSS        : StrEnumAttrCase<"VPU_DDR_BSS">;
def VPUIP_MemoryLocation_VPU_CSRAM          : StrEnumAttrCase<"VPU_CSRAM">;

def VPUIP_MemoryLocation :
        StrEnumAttr<
            "MemoryLocation",
            "Values indicating which type of memory a tensor resides in",
            [
                VPUIP_MemoryLocation_ProgrammableInput,
                VPUIP_MemoryLocation_ProgrammableOutput,
                VPUIP_MemoryLocation_VPU_DDR_Heap,
                VPUIP_MemoryLocation_GraphFile,
                VPUIP_MemoryLocation_VPU_CMX_NN,
                VPUIP_MemoryLocation_VPU_CMX_UPA,
                VPUIP_MemoryLocation_VPU_DDR_BSS,
                VPUIP_MemoryLocation_VPU_CSRAM,
            ]
        > {
    let cppNamespace = "vpux::VPUIP";
}

//
// ArchKind
//

def VPUIP_ArchKind_MA2490       : StrEnumAttrCase<"MA2490">;    // Keem bay A0
def VPUIP_ArchKind_MA2490_B0    : StrEnumAttrCase<"MA2490_B0">; // Keem bay B0
def VPUIP_ArchKind_MA3100       : StrEnumAttrCase<"MA3100">;    // Thunder bay harbor A0
def VPUIP_ArchKind_MA3720       : StrEnumAttrCase<"MA3720">;    // Meteor lake

def VPUIP_ArchKind :
        StrEnumAttr<
            "ArchKind",
            "Represents VPU architecture generation",
            [
                VPUIP_ArchKind_MA2490,
                VPUIP_ArchKind_MA2490_B0,
                VPUIP_ArchKind_MA3100,
                VPUIP_ArchKind_MA3720,
            ]
        > {
    let cppNamespace = "vpux::VPUIP";
}

//
// PoolLayerType
//

def VPUIP_PoolLayerType_MAX : StrEnumAttrCase<"MAX">;
def VPUIP_PoolLayerType_AVG : StrEnumAttrCase<"AVG">;

def VPUIP_PoolLayerType :
        StrEnumAttr<
            "PoolLayerType",
            "Type of Pooling layer",
            [
                VPUIP_PoolLayerType_MAX,
                VPUIP_PoolLayerType_AVG,
            ]
        > {
    let cppNamespace = "vpux::VPUIP";
}

#endif
