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
// Base class
//

class VPUIP_Attr<Pred pred, string descr> :
        DialectAttr<
            VPUIP_Dialect,
            pred,
            descr
        >;

//
// PhysicalProcessor
//

def VPUIP_PhysicalProcessor_ARM                  : I32EnumAttrCase<"ARM",               0>;
def VPUIP_PhysicalProcessor_Leon_RT              : I32EnumAttrCase<"Leon_RT",           1>;
def VPUIP_PhysicalProcessor_Leon_NN              : I32EnumAttrCase<"Leon_NN",           2>;
def VPUIP_PhysicalProcessor_SHAVE_UPA            : I32EnumAttrCase<"SHAVE_UPA",         3>;
def VPUIP_PhysicalProcessor_SHAVE_NN             : I32EnumAttrCase<"SHAVE_NN",          4>;
def VPUIP_PhysicalProcessor_NCE_Cluster          : I32EnumAttrCase<"NCE_Cluster",       5>;
def VPUIP_PhysicalProcessor_NCE_PerClusterDPU    : I32EnumAttrCase<"NCE_PerClusterDPU", 6>;

def VPUIP_PhysicalProcessor :
        I32EnumAttr<
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

def VPUIP_DMAEngine_UPA  : I32EnumAttrCase<"UPA",   0>;
def VPUIP_DMAEngine_NN   : I32EnumAttrCase<"NN",    1>;

def VPUIP_DMAEngine :
        I32EnumAttr<
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

def VPUIP_PhysicalMemory_DDR     : I32EnumAttrCase<"DDR",       0>;
def VPUIP_PhysicalMemory_CSRAM   : I32EnumAttrCase<"CSRAM",     1>;
def VPUIP_PhysicalMemory_CMX_UPA : I32EnumAttrCase<"CMX_UPA",   2>;
def VPUIP_PhysicalMemory_CMX_NN  : I32EnumAttrCase<"CMX_NN",    3>;

def VPUIP_PhysicalMemory :
        I32EnumAttr<
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
// ArchKind
//

def VPUIP_ArchKind_KMB : I32EnumAttrCase<"KMB", 0>;

def VPUIP_ArchKind :
        I32EnumAttr<
            "ArchKind",
            "Represents VPU architecture generation",
            [
                VPUIP_ArchKind_KMB,
            ]
        > {
    let cppNamespace = "vpux::VPUIP";
}

//
// MemoryLocation
//

def VPUIP_MemoryLocation_ProgrammableInput  : I32EnumAttrCase<"ProgrammableInput",  0>;
def VPUIP_MemoryLocation_ProgrammableOutput : I32EnumAttrCase<"ProgrammableOutput", 1>;
def VPUIP_MemoryLocation_VPU_DDR_Heap       : I32EnumAttrCase<"VPU_DDR_Heap",       2>;
def VPUIP_MemoryLocation_GraphFile          : I32EnumAttrCase<"GraphFile",          3>;
def VPUIP_MemoryLocation_VPU_CMX_NN         : I32EnumAttrCase<"VPU_CMX_NN",         4>;
def VPUIP_MemoryLocation_VPU_CMX_UPA        : I32EnumAttrCase<"VPU_CMX_UPA",        5>;
def VPUIP_MemoryLocation_VPU_DDR_BSS        : I32EnumAttrCase<"VPU_DDR_BSS",        6>;
def VPUIP_MemoryLocation_VPU_CSRAM          : I32EnumAttrCase<"VPU_CSRAM",          7>;

def VPUIP_MemoryLocation :
        I32EnumAttr<
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
// TaskType
//

class VPUIP_TaskTypeVal<string sym, int val> : I32EnumAttrCase<sym, val> {
    string enumVal = "vpux::VPUIP::TaskType::" # sym;
}

def VPUIP_TaskType_UPADMA       : VPUIP_TaskTypeVal<"UPADMA",       0>;
def VPUIP_TaskType_NNDMA        : VPUIP_TaskTypeVal<"NNDMA",        1>;
def VPUIP_TaskType_NCE2         : VPUIP_TaskTypeVal<"NCE2",         2>;
def VPUIP_TaskType_Controller   : VPUIP_TaskTypeVal<"Controller",   3>;
def VPUIP_TaskType_UPA          : VPUIP_TaskTypeVal<"UPA",          4>;
def VPUIP_TaskType_SNN          : VPUIP_TaskTypeVal<"SNN",          5>;

def VPUIP_TaskType :
        I32EnumAttr<
            "TaskType",
            "The type of the relevant processing code of the VPU for the task",
            [
                VPUIP_TaskType_UPADMA,
                VPUIP_TaskType_NNDMA,
                VPUIP_TaskType_NCE2,
                VPUIP_TaskType_Controller,
                VPUIP_TaskType_UPA,
                VPUIP_TaskType_SNN,
            ]
        > {
    let cppNamespace = "vpux::VPUIP";
}

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
// Resources
//

def VPUIP_Resources_upa_shaves          : StructFieldAttr<"upa_shaves",             DefaultValuedAttr<I32Attr, "0">>;
def VPUIP_Resources_nce2_blocks         : StructFieldAttr<"nce2_blocks",            DefaultValuedAttr<I32Attr, "0">>;
def VPUIP_Resources_upa_shared_cmx      : StructFieldAttr<"upa_shared_cmx",         DefaultValuedAttr<I32Attr, "0">>;
def VPUIP_Resources_nn_cmx_per_slice    : StructFieldAttr<"nn_cmx_per_slice",       DefaultValuedAttr<I32Attr, "0">>;
def VPUIP_Resources_nn_cmx_slice_amount : StructFieldAttr<"nn_cmx_slice_amount",    DefaultValuedAttr<I32Attr, "0">>;
def VPUIP_Resources_ddr_scratch         : StructFieldAttr<"ddr_scratch",            DefaultValuedAttr<I64Attr, "0">>;
def VPUIP_Resources_csram_storage       : StructFieldAttr<"csram_storage",          DefaultValuedAttr<I64Attr, "0">>;

def VPUIP_ResourcesAttr :
        StructAttr<
            "ResourcesAttr",
            VPUIP_Dialect,
            [
                VPUIP_Resources_upa_shaves,
                VPUIP_Resources_nce2_blocks,
                VPUIP_Resources_upa_shared_cmx,
                VPUIP_Resources_nn_cmx_per_slice,
                VPUIP_Resources_nn_cmx_slice_amount,
                VPUIP_Resources_ddr_scratch,
                VPUIP_Resources_csram_storage,
            ]
        >;

#endif
