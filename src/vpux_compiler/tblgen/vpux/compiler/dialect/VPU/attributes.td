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

#ifndef VPUX_COMPILER_DIALECT_VPU_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_VPU_ATTRIBUTES

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/dialect/VPU/dialect.td"

//
// ArchKind
//

def VPU_ArchKind :
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
    let cppNamespace = "vpux::VPU";
    let genSpecializedAttr = 1;
}

//
// MemoryKind
//

def VPU_MemoryKind :
        StrEnumAttr<
            "MemoryKind",
            "Represents the actual hardware memory hierarchy",
            [
                StrEnumAttrCase<"DDR">,
                StrEnumAttrCase<"CSRAM">,
                StrEnumAttrCase<"CMX_UPA">,
                StrEnumAttrCase<"CMX_NN">,
                StrEnumAttrCase<"Register">,
            ]
        > {
    let cppNamespace = "vpux::VPU";
    let genSpecializedAttr = 1;
}

//
// ExecutorKind
//

def VPU_ExecutorKind :
        StrEnumAttr<
            "ExecutorKind",
            "Representd hardware executror resources",
            [
                StrEnumAttrCase<"DMA_UPA">,
                StrEnumAttrCase<"DMA_NN">,
                StrEnumAttrCase<"NCE">,
                StrEnumAttrCase<"DPU">,
                StrEnumAttrCase<"SHAVE_UPA">,
                StrEnumAttrCase<"SHAVE_NN">,
                StrEnumAttrCase<"SHAVE_ACT">,
            ]
        > {
    let cppNamespace = "vpux::VPU";
    let genSpecializedAttr = 1;
}

//
// CompilationMode
//

def VPU_CompilationMode  :
        StrEnumAttr<
            "CompilationMode",
            "Compilation Mode",
            [
                StrEnumAttrCase<"ReferenceSW">,
                StrEnumAttrCase<"ReferenceHW">,
                StrEnumAttrCase<"DefaultHW">
            ]
        > {
    let cppNamespace = "vpux::VPU";
    let genSpecializedAttr = 1;
}

#endif
