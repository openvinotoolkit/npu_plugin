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


//
// SectionTypeAttr
//

def ELF_SectionTypeAttr :
        StrEnumAttr<
            "SectionTypeAttr",
            "Enum for describing ELF section types",
            [
                // From include/mlir/IR/OpBase.td: class StrEnumAttrCase<string sym, int val = -1>. So we can add an integer, which is the value of the respectiv enum case.
                // Cases are taken from elfio/elf_types.hpp
                StrEnumAttrCase<"SHT_NULL",           0>,
                StrEnumAttrCase<"SHT_PROGBITS",       1>,
                StrEnumAttrCase<"SHT_SYMTAB",         2>,
                StrEnumAttrCase<"SHT_STRTAB",         3>,
                StrEnumAttrCase<"SHT_RELA",           4>,
                StrEnumAttrCase<"SHT_HASH",           5>,
                StrEnumAttrCase<"SHT_DYNAMIC",        6>,
                StrEnumAttrCase<"SHT_NOTE",           7>,
                StrEnumAttrCase<"SHT_NOBITS",         8>,
                StrEnumAttrCase<"SHT_REL",            9>,
                StrEnumAttrCase<"SHT_SHLIB",          10>,
                StrEnumAttrCase<"SHT_DYNSYM",         11>,
                StrEnumAttrCase<"SHT_INIT_ARRAY",     14>,
                StrEnumAttrCase<"SHT_FINI_ARRAY",     15>,
                StrEnumAttrCase<"SHT_PREINIT_ARRAY",  16>,
                StrEnumAttrCase<"SHT_GROUP",          17>,
                StrEnumAttrCase<"SHT_SYMTAB_SHNDX",   18>,
                StrEnumAttrCase<"SHT_GNU_ATTRIBUTES", 0x6ffffff5>,
                StrEnumAttrCase<"SHT_GNU_HASH",       0x6ffffff6>,
                StrEnumAttrCase<"SHT_GNU_LIBLIST",    0x6ffffff7>,
                StrEnumAttrCase<"SHT_GNU_CHECKSUM",   0x6ffffff8>,
                StrEnumAttrCase<"SHT_GNU_LOSUNW",     0x6ffffffa>,
                // <<error: duplicate case value>>: StrEnumAttrCase<"SHT_SUNW_move",      0x6ffffffa>,
                StrEnumAttrCase<"SHT_SUNW_COMDAT",    0x6ffffffb>,
                StrEnumAttrCase<"SHT_SUNW_syminfo",   0x6ffffffc>,
                StrEnumAttrCase<"SHT_GNU_verdef",     0x6ffffffd>,
                StrEnumAttrCase<"SHT_GNU_verneed",    0x6ffffffe>,
                StrEnumAttrCase<"SHT_GNU_versym",     0x6fffffff>,
                //StrEnumAttrCase<"SHT_LOOS",           0x60000000>,
                // <<error: duplicate case value>>: StrEnumAttrCase<"SHT_HIOS",           0x6fffffff>,
                //StrEnumAttrCase<"SHT_LOPROC",         0x70000000>,
                //StrEnumAttrCase<"SHT_HIPROC",         0x7FFFFFFF>,
                // <<error: enumerator value ‘2147483648’ is outside the range of underlying type ‘int’>>: StrEnumAttrCase<"SHT_LOUSER",         0x80000000>,
                // <<error: enumerator value ‘4294967295’ is outside the range of underlying type ‘int’>>: StrEnumAttrCase<"SHT_HIUSER",         0xFFFFFFFF>,
            ]
        > {
    let cppNamespace = "vpux::ELF";
    let genSpecializedAttr = 1;
}


//
// SectionFlagsAttr
//

def ELF_SectionFlagsAttr :
        BitEnumAttr<
            "SectionFlagsAttr",
            "Enum for describing ELF section flags (we have also cases encountered with OR)",
            [
                // Cases are taken from elfio/elf_types.hpp
                BitEnumAttrCase<"SHF_NONE",                0x0>, // Added by Alex
                BitEnumAttrCase<"SHF_WRITE",               0x1>,
                BitEnumAttrCase<"SHF_ALLOC",               0x2>,
                BitEnumAttrCase<"SHF_EXECINSTR",           0x4>,
                BitEnumAttrCase<"SHF_ALLOC_OR_EXECINSTR",  0x6>, // NEW: Ored
                BitEnumAttrCase<"SHF_MERGE",               0x10>,
                BitEnumAttrCase<"SHF_STRINGS",             0x20>,
                BitEnumAttrCase<"SHF_INFO_LINK",           0x40>,
                BitEnumAttrCase<"SHF_LINK_ORDER",          0x80>,
                BitEnumAttrCase<"SHF_OS_NONCONFORMING",    0x100>,
                BitEnumAttrCase<"SHF_GROUP",               0x200>,
                BitEnumAttrCase<"SHF_TLS",                 0x400>,
                //BitEnumAttrCase<"SHF_MASKOS",              0x0ff00000>,
                // error: enumerator value ‘4026531840’ is outside the range of underlying type ‘int’: BitEnumAttrCase<"SHF_MASKPROC",         0xF0000000>,
            ]
        > {
    let cppNamespace = "vpux::ELF";
}



#endif
