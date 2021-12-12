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

//TODO: include vpux lef type definitions

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
                // See https://docs.oracle.com/cd/E19683-01/816-1386/6m7qcoblj/index.html for meaning of these Section Header Types.
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
// SymbolTypeAttr
//


def ELF_SymbolTypeAttr :
    StrEnumAttr<
        "SymbolTypeAttr",
        "Enum to represent symbol types",
        [
            StrEnumAttrCase<"STT_NOTYPE"     , 0>,
            StrEnumAttrCase<"STT_OBJECT"     , 1>,
            StrEnumAttrCase<"STT_FUNC"       , 2>,
            StrEnumAttrCase<"STT_SECTION"    , 3>,
            StrEnumAttrCase<"STT_FILE"       , 4>,
            StrEnumAttrCase<"STT_COMMON"     , 5>,
            StrEnumAttrCase<"STT_TLS"        , 6>,
            StrEnumAttrCase<"VPU_STT_ENTRY"  , 10>,
            StrEnumAttrCase<"VPU_STT_INPUT"  , 11>,
            StrEnumAttrCase<"VPU_STT_OUTPUT" , 12>,
        ]
    > {
    let cppNamespace = "vpux::ELF";
    let genSpecializedAttr = 1;
}
//
// SectionFlagsAttr
//

//TODO::BitEnumAttr has underlying type as uint32_t. ELF section flags can span uint64_t range.
//Need to write "own version" of BitEnumAttr with underlying type as uint64_t
//Until then restrict only to flags that fit in uint32_t!
def ELF_SectionFlagsAttr :
        BitEnumAttr<
            "SectionFlagsAttr",
            "Enum for describing ELF section flags (we have also cases encountered with OR)",
            [
                BitEnumAttrCase<"SHF_NONE",                0x0>,
                BitEnumAttrCase<"SHF_WRITE",               0x1>,
                BitEnumAttrCase<"SHF_ALLOC",               0x2>,
                BitEnumAttrCase<"SHF_EXECINSTR",           0x4>,
                BitEnumAttrCase<"SHF_INFO_LINK",           0x40>,
                BitEnumAttrCase<"VPU_SHF_JIT",             0x10000000>,
                BitEnumAttrCase<"VPU_SHF_USERINPUT",       0x20000000>,
                BitEnumAttrCase<"VPU_SHF_USEROUTPUT",      0x40000000>,
            ]
        > {
    let cppNamespace = "vpux::ELF";
}




//
// RelocationTypeAttr
//

def ELF_RelocationTypeAttr :
        StrEnumAttr<
            "RelocationTypeAttr",
            "Enum for describing ELF relocation types",
            [
                // From include/mlir/IR/OpBase.td: class StrEnumAttrCase<string sym, int val = -1>. So we can add an integer, which is the value of the respectiv enum case.
                // Cases are taken from [kmb-plugin]/src/vpux_elf/include/elf/types/vpu_extensions.hpp
                StrEnumAttrCase<"R_VPU_64",                                  0>,
                StrEnumAttrCase<"R_VPU_64_OR",                               1>,
                StrEnumAttrCase<"R_VPU_64_OR_RTM",                           2>,
                StrEnumAttrCase<"R_VPU_64_LSHIFT",                           3>,
                StrEnumAttrCase<"R_VPU_32",                                  4>,
                StrEnumAttrCase<"R_VPU_32_OR_RTM",                           5>,
                StrEnumAttrCase<"R_VPU_32_SUM",                              6>,
                StrEnumAttrCase<"R_VPU_32_MULTICAST_BASE",                   7>,
                StrEnumAttrCase<"R_VPU_32_MULTICAST_BASE_SUB",               8>,
                StrEnumAttrCase<"R_VPU_32_MULTICAST_OFFSET_4_BIT_SHIFT_OR",  9>,
                StrEnumAttrCase<"R_VPU_32_MULTICAST_OFFSET_CMP_OR",         10>,
            ]
        > {
    let cppNamespace = "vpux::ELF";
    let genSpecializedAttr = 1;
}



#endif
