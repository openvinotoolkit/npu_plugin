//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUREGMAPPED_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_VPUREGMAPPED_ATTRIBUTES

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/dialect/VPURegMapped/dialect.td"

class VPURegMapped_Attr<string name, list<Trait> traits = []> : AttrDef<VPURegMapped_Dialect, name, traits> {
    let mnemonic = name;
}

//
// RegFieldDataType
//

def VPURegMapped_RegFieldDataType :
        StrEnumAttr<
            "RegFieldDataType",
            "RegFieldDataType that the regField supports",
            [
                StrEnumAttrCase<"UINT">,
                StrEnumAttrCase<"SINT">,
                StrEnumAttrCase<"FP">,
            ]
        > {
    let cppNamespace = "vpux::VPURegMapped";
    let genSpecializedAttr = 1;
}

//
// RegisterFieldAttr
//

def VPURegMapped_RegisterFieldAttr : VPURegMapped_Attr<"RegisterField"> {
    let description = "This object represents closely a RegisterField Attr";
    let cppNamespace = "vpux::VPURegMapped";

    let parameters = (ins "vpux::VPURegMapped::RegFieldType":$regField);

    let genVerifyDecl = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// RegisterFieldArrayAttr
//

def VPURegMapped_RegisterFieldArrayAttr : TypedArrayAttrBase<VPURegMapped_RegisterFieldAttr,
    "array of RegisterFields">{
        string cppType = "::mlir::ArrayAttr";
    }

//
// RegisterAttr
//

def VPURegMapped_RegisterAttr : VPURegMapped_Attr<"Register"> {
    let description = "This object represents closely a Register Attr";
    let cppNamespace = "vpux::VPURegMapped";

    let parameters = (ins "vpux::VPURegMapped::RegisterType":$reg);

    let genVerifyDecl = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// RegisterArrayAttr
//

def VPURegMapped_RegisterArrayAttr : TypedArrayAttrBase<VPURegMapped_RegisterAttr,
    "array of Registers">{
        string cppType = "::mlir::ArrayAttr";
    }

//
// RegisterMappedAttr
//

def VPURegMapped_RegisterMappedAttr : VPURegMapped_Attr<"RegisterMapped"> {
    let description = "This object represents closely a RegisterMapped Attr";
    let cppNamespace = "vpux::VPURegMapped";

    let parameters = (ins "vpux::VPURegMapped::RegMappedType":$regMapped);

    let genVerifyDecl = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// TaskType
//

def VPURegMapped_TaskType :
        StrEnumAttr<
            "TaskType",
            "The type of the relevant processing code of the VPU for the task",
            [
                StrEnumAttrCase<"DMA">,
                StrEnumAttrCase<"ActKernelInvocation">,
                StrEnumAttrCase<"ActKernelRange">,
                StrEnumAttrCase<"DPUInvariant">,
                StrEnumAttrCase<"DPUVariant">
            ]
        > {
    let cppNamespace = "vpux::VPURegMapped";
    let genSpecializedAttr = 1;
}

#endif
