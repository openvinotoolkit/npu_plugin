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

#pragma once

#include <vpux_elf/types/data_types.hpp>

namespace elf {

///
/// Refer to https://docs.oracle.com/cd/E26502_01/html/E26507/chapter6-35342.html#scrolltoc
/// for the detailed description of the values below
///

//! File identification
constexpr uint8_t EI_MAG0       = 0;
constexpr uint8_t EI_MAG1       = 1;
constexpr uint8_t EI_MAG2       = 2;
constexpr uint8_t EI_MAG3       = 3;
constexpr uint8_t EI_CLASS      = 4;
constexpr uint8_t EI_DATA       = 5;
constexpr uint8_t EI_VERSION    = 6;
constexpr uint8_t EI_OSABI      = 7;
constexpr uint8_t EI_ABIVERSION = 8;
constexpr uint8_t EI_PAD        = 9;
constexpr uint8_t EI_NIDENT     = 16;

//! ELF magic values
constexpr uint8_t ELFMAG0 = 0x7f;
constexpr uint8_t ELFMAG1 = 'E';
constexpr uint8_t ELFMAG2 = 'L';
constexpr uint8_t ELFMAG3 = 'F';

//! File class
constexpr uint8_t ELFCLASSNONE = 0;
constexpr uint8_t ELFCLASS32   = 1;
constexpr uint8_t ELFCLASS64   = 2;

//! Data encoding
constexpr uint8_t ELFDATANONE = 0;
constexpr uint8_t ELFDATA2LSB = 1;
constexpr uint8_t ELFDATA2MSB = 2;

///
/// Refer to https://docs.oracle.com/cd/E26502_01/html/E26507/chapter6-43405.html
/// for the detailed description of the values and structures below
///

//! File types
constexpr Elf_Half ET_NONE           = 0;
constexpr Elf_Half ET_REL            = 1;
constexpr Elf_Half ET_EXEC           = 2;
constexpr Elf_Half ET_DYN            = 3;
constexpr Elf_Half ET_CORE           = 4;
constexpr Elf_Half ET_LOSUNW         = 0xfefe;
constexpr Elf_Half ET_SUNW_ANCILLARY = 0xfefe;
constexpr Elf_Half ET_HISUNW         = 0xfefd;
constexpr Elf_Half ET_LOPROC         = 0xff00;
constexpr Elf_Half ET_HIPROC         = 0xffff;

constexpr Elf_Half EM_NONE        = 0;
constexpr Elf_Half EM_SPARC       = 2;
constexpr Elf_Half EM_386         = 3;
constexpr Elf_Half EM_SPARC32PLUS = 18;
constexpr Elf_Half EM_SPARCV9     = 43;
constexpr Elf_Half EM_AMD64       = 62;

constexpr Elf_Word EV_NONE = 0;

struct Elf64_Ehdr {
    unsigned char e_ident[EI_NIDENT];
    Elf_Half      e_type;
    Elf_Half      e_machine;
    Elf_Word      e_version;
    Elf64_Addr    e_entry;
    Elf64_Off     e_phoff;
    Elf64_Off     e_shoff;
    Elf_Word      e_flags;
    Elf_Half      e_ehsize;
    Elf_Half      e_phentsize;
    Elf_Half      e_phnum;
    Elf_Half      e_shentsize;
    Elf_Half      e_shnum;
    Elf_Half      e_shstrndx;
};

using ELFHeader = Elf64_Ehdr;

} // namespace elf
