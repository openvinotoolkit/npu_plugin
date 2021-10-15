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

#ifndef VPUX_COMPILER_DIALECT_ELF_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_ELF_ATTRIBUTES

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/dialect/ELF/dialect.td"


// Inspired from VPUIP's dialect "def VPUIP_PhysicalProcessor"
def ELF_SectionTypeAttr1 :
        StrEnumAttr<
            "SectionTypeAttr1",
            "Enum for describing ELF section types",
            [
                // From include/mlir/IR/OpBase.td: class StrEnumAttrCase<string sym, int val = -1>
                StrEnumAttrCase<"SHT_NULL", 0>, // 0
                StrEnumAttrCase<"SHT_PROGBITS", 1>, // 1
                StrEnumAttrCase<"SHT_SYMTAB", 2>, // 2
                StrEnumAttrCase<"SHT_STRTAB", 3>, // 3
                StrEnumAttrCase<"SHT_RELA", 4>, // 4
                StrEnumAttrCase<"SHT_HASH", 5>, // 5
                StrEnumAttrCase<"SHT_DYNAMIC", 6>, // 6
                StrEnumAttrCase<"SHT_NOTE", 7>, // 7
                StrEnumAttrCase<"SHT_NOBITS", 8>, // 8
                StrEnumAttrCase<"SHT_REL", 9>, // 9
                StrEnumAttrCase<"SHT_SHLIB", 10>, // 10
                StrEnumAttrCase<"SHT_DYNSYM", 11>, // 11
                StrEnumAttrCase<"SHT_SYMTAB_SHNDX", 18>, // 18
            ]
        > {
    let cppNamespace = "vpux::ELF";
    let genSpecializedAttr = 1;
}


// Inspired from VPUIP's dialect "def VPUIP_ExecutionFlag"
def ELF_SectionTypeAttr2:
        BitEnumAttr<
            "SectionTypeAttr2",
            "Each of these enums' presence informs how ...",
            [
                BitEnumAttrCase<"SHT_NULL",         0x0000>,
                BitEnumAttrCase<"SHT_PROGBITS",     0x0001>,
                BitEnumAttrCase<"SHT_SYMTAB",       0x0002>,
                BitEnumAttrCase<"SHT_STRTAB",       0x0003>,
                BitEnumAttrCase<"SHT_RELA",         0x0004>,
                BitEnumAttrCase<"SHT_HASH",         0x0005>,
                BitEnumAttrCase<"SHT_DYNAMIC",      0x0006>,
                BitEnumAttrCase<"SHT_NOTE",         0x0007>,
                BitEnumAttrCase<"SHT_NOBITS",       0x0008>,
                BitEnumAttrCase<"SHT_REL",          0x0009>,
                BitEnumAttrCase<"SHT_SHLIB",        0x000A>,
                BitEnumAttrCase<"SHT_DYNSYM",       0x000B>,
                BitEnumAttrCase<"SHT_SYMTAB_SHNDX", 0x0012>,
            ]
        > {
    let cppNamespace = "vpux::ELF";
}


#endif
