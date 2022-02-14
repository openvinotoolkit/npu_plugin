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


// For further development, please refer to the ticket #29279.


//
// SectionTypeAttr
//
// Refer to https://docs.oracle.com/cd/E19455-01/806-3773/elf-2/index.html
//   for the detailed description of the case values.

// I64EnumAttr has uint64_t as the underlying type, which is OK for all cases.
def ELF_SectionTypeAttr :
        I64EnumAttr<
            "SectionTypeAttr",
            "Enum for describing ELF section header types",
            [
                I64EnumAttrCase<"SHT_NULL",     0>,
                I64EnumAttrCase<"SHT_PROGBITS", 1>,
                I64EnumAttrCase<"SHT_SYMTAB",   2>,
                I64EnumAttrCase<"SHT_STRTAB",   3>,
                I64EnumAttrCase<"SHT_RELA",     4>,
                I64EnumAttrCase<"SHT_HASH",     5>,
                I64EnumAttrCase<"SHT_DYNAMIC",  6>,
                I64EnumAttrCase<"SHT_NOTE",     7>,
                I64EnumAttrCase<"SHT_NOBITS",   8>,
                I64EnumAttrCase<"SHT_REL",      9>,
                I64EnumAttrCase<"SHT_SHLIB",    10>,
                I64EnumAttrCase<"SHT_DYNSYM",   11>,
                I64EnumAttrCase<"SHT_LOPROC",   0x70000000>,
                I64EnumAttrCase<"SHT_HIPROC",   0x7FFFFFFF>,
                I64EnumAttrCase<"SHT_LOUSER",   0x80000000>,
                I64EnumAttrCase<"SHT_HIUSER",   0xFFFFFFFF>
            ]
        > {
    let cppNamespace = "vpux::ELF";
    let genSpecializedAttr = 1;
}

//
// SymbolTypeAttr
//
// Refer to https://docs.oracle.com/cd/E23824_01/html/819-0690/chapter6-79797.html
// for the detailed description of the case values

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
            StrEnumAttrCase<"STT_LOPROC"     , 13>,
            StrEnumAttrCase<"STT_HIPROC"     , 15>,
        ]
    > {
    let cppNamespace = "vpux::ELF";
    let genSpecializedAttr = 1;
}

//
// SectionFlagsAttr
//
// Refer to
// https://docs.oracle.com/cd/E19120-01/open.solaris/819-0690/6n33n7fcj/index.html
// (or https://docs.oracle.com/cd/E19455-01/806-3773/elf-2/index.html)
// for the detailed description of the case values
//
// BitEnumAttr has uint32_t as underlying type, which is OK for all cases normally.
// See thirdparty/llvm-project/mlir/include/mlir/IR/OpBase.td, "class BitEnumAttr".
// Please refer to the ticket #29800 to address this small limitation.

def ELF_SectionFlagsAttr :
        BitEnumAttr<
            "SectionFlagsAttr",
            "Enum for describing ELF section header flags (we can use also the | operator)",
            [
                BitEnumAttrCase<"SHF_NONE",                0x0>,
                BitEnumAttrCase<"SHF_WRITE",               0x1>,
                BitEnumAttrCase<"SHF_ALLOC",               0x2>,
                BitEnumAttrCase<"SHF_EXECINSTR",           0x4>,
                BitEnumAttrCase<"SHF_INFO_LINK",           0x40>,
                BitEnumAttrCase<"VPU_SHF_JIT",             0x10000000>,
                BitEnumAttrCase<"VPU_SHF_USERINPUT",       0x20000000>,
                BitEnumAttrCase<"VPU_SHF_USEROUTPUT",      0x40000000>,
                BitEnumAttrCase<"SHF_MASKPROC",            0xF0000000>,
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
