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

def VPUIP_PhysicalProcessor_ARM                  : StrEnumAttrCase<"ARM">;
def VPUIP_PhysicalProcessor_Leon_RT              : StrEnumAttrCase<"Leon_RT">;
def VPUIP_PhysicalProcessor_Leon_NN              : StrEnumAttrCase<"Leon_NN">;
def VPUIP_PhysicalProcessor_SHAVE_UPA            : StrEnumAttrCase<"SHAVE_UPA">;
def VPUIP_PhysicalProcessor_SHAVE_NN             : StrEnumAttrCase<"SHAVE_NN">;
def VPUIP_PhysicalProcessor_NCE_Cluster          : StrEnumAttrCase<"NCE_Cluster">;
def VPUIP_PhysicalProcessor_NCE_PerClusterDPU    : StrEnumAttrCase<"NCE_PerClusterDPU">;

def VPUIP_PhysicalProcessorEnum :
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

def VPUIP_PhysicalProcessorAttr :
        VPUIP_Attr<
            CPred<"$_self.isa<vpux::VPUIP::PhysicalProcessorAttr>()">,
            "Attribute for describing processor resources"
        > {
    let storageType = "vpux::VPUIP::PhysicalProcessorAttr";
    let returnType = "vpux::VPUIP::PhysicalProcessor";

    let constBuilderCall = "vpux::VPUIP::PhysicalProcessorAttr::get($_builder.getContext(), $0)";
}

//
// DMAEngine
//

def VPUIP_DMAEngine_UPA  : StrEnumAttrCase<"UPA">;
def VPUIP_DMAEngine_NN   : StrEnumAttrCase<"NN">;

def VPUIP_DMAEngineEnum :
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

def VPUIP_DMAEngineAttr :
        VPUIP_Attr<
            CPred<"$_self.isa<vpux::VPUIP::DMAEngineAttr>()">,
            "DMA engine kind"
        > {
    let storageType = "vpux::VPUIP::DMAEngineAttr";
    let returnType = "vpux::VPUIP::DMAEngine";

    let constBuilderCall = "vpux::VPUIP::DMAEngineAttr::get($_builder.getContext(), $0)";
}

//
// PhysicalMemory
//

def VPUIP_PhysicalMemory_DDR     : StrEnumAttrCase<"DDR">;
def VPUIP_PhysicalMemory_CSRAM   : StrEnumAttrCase<"CSRAM">;
def VPUIP_PhysicalMemory_CMX_UPA : StrEnumAttrCase<"CMX_UPA">;
def VPUIP_PhysicalMemory_CMX_NN  : StrEnumAttrCase<"CMX_NN">;

def VPUIP_PhysicalMemoryEnum :
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

def VPUIP_PhysicalMemoryAttr :
        VPUIP_Attr<
            CPred<"$_self.isa<vpux::VPUIP::PhysicalMemoryAttr>()">,
            "Represents the actual hardware memory hierarchy"
        > {
    let storageType = "vpux::VPUIP::PhysicalMemoryAttr";
    let returnType = "vpux::VPUIP::PhysicalMemory";

    let constBuilderCall = "vpux::VPUIP::PhysicalMemoryAttr::get($_builder.getContext(), $0)";
}

//
// ArchKind
//

def VPUIP_ArchKind_KMB : StrEnumAttrCase<"KMB">;

def VPUIP_ArchKindEnum :
        StrEnumAttr<
            "ArchKind",
            "Represents VPU architecture generation",
            [
                VPUIP_ArchKind_KMB,
            ]
        > {
    let cppNamespace = "vpux::VPUIP";
}

def VPUIP_ArchKindAttr :
        VPUIP_Attr<
            CPred<"$_self.isa<vpux::VPUIP::ArchKindAttr>()">,
            "Represents VPU architecture generation"
        > {
    let storageType = "vpux::VPUIP::ArchKindAttr";
    let returnType = "vpux::VPUIP::ArchKind";

    let constBuilderCall = "vpux::VPUIP::ArchKindAttr::get($_builder.getContext(), $0)";
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

def VPUIP_MemoryLocationEnum :
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

def VPUIP_MemoryLocationAttr :
        VPUIP_Attr<
            CPred<"$_self.isa<vpux::VPUIP::MemoryLocationAttr>()">,
            "Values indicating which type of memory a tensor resides in"
        > {
    let storageType = "vpux::VPUIP::MemoryLocationAttr";
    let returnType = "vpux::VPUIP::MemoryLocation";

    let constBuilderCall = "vpux::VPUIP::MemoryLocationAttr::get($_builder.getContext(), $0)";
}

//
// ExecutionFlag
//

def VPUIP_ExecutionFlag_DynamicBarriers     : StrEnumAttrCase<"DynamicBarriers">;
def VPUIP_ExecutionFlag_Compiled_For_VPU3   : StrEnumAttrCase<"Compiled_For_VPU3">;

def VPUIP_ExecutionFlagEnum :
        StrEnumAttr<
            "ExecutionFlag",
            "Each of these enums' presence informs how the current schedule is configured",
            [
                VPUIP_ExecutionFlag_DynamicBarriers,
                VPUIP_ExecutionFlag_Compiled_For_VPU3,
            ]
        > {
    let cppNamespace = "vpux::VPUIP";
}

def VPUIP_ExecutionFlagAttr :
        VPUIP_Attr<
            CPred<"$_self.isa<vpux::VPUIP::ExecutionFlagAttr>()">,
            "Each of these enums' presence informs how the current schedule is configured"
        > {
    let storageType = "vpux::VPUIP::ExecutionFlagAttr";
    let returnType = "vpux::VPUIP::ExecutionFlag";

    let constBuilderCall = "vpux::VPUIP::ExecutionFlagAttr::get($_builder.getContext(), $0)";
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

def VPUIP_TaskTypeEnum :
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
            ]
        > {
    let cppNamespace = "vpux::VPUIP";
}

def VPUIP_TaskTypeAttr :
        VPUIP_Attr<
            CPred<"$_self.isa<vpux::VPUIP::TaskTypeAttr>()">,
            "Each of these enums' presence informs how the current schedule is configured"
        > {
    let storageType = "vpux::VPUIP::TaskTypeAttr";
    let returnType = "vpux::VPUIP::TaskType";

    let constBuilderCall = "vpux::VPUIP::TaskTypeAttr::get($_builder.getContext(), $0)";
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
