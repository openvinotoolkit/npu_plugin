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
            ]
        > {
    let cppNamespace = "vpux::VPUIP";
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
            ]
        > {
    let cppNamespace = "vpux::VPUIP";
}

//
// ArchKind
//

def VPUIP_ArchKind :
        StrEnumAttr<
            "ArchKind",
            "Represents VPU architecture generation",
            [
                StrEnumAttrCase<"MA2490">,    // Keem bay A0
                StrEnumAttrCase<"MA2490_B0">, // Keem bay B0
                StrEnumAttrCase<"MA3100">,    // Thunder bay harbor A0
                StrEnumAttrCase<"MA3720">,    // Meteor lake
            ]
        > {
    let cppNamespace = "vpux::VPUIP";
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
}

#endif
