// {% copyright %}

#pragma once

#include <elf/types/data_types.hpp>

namespace elf {

constexpr int EI_MAG0       = 0;
constexpr int EI_MAG1       = 1;
constexpr int EI_MAG2       = 2;
constexpr int EI_MAG3       = 3;
constexpr int EI_CLASS      = 4;
constexpr int EI_DATA       = 5;
constexpr int EI_VERSION    = 6;
constexpr int EI_OSABI      = 7;
constexpr int EI_ABIVERSION = 8;
constexpr int EI_PAD        = 9;
constexpr int EI_NIDENT     = 16;

constexpr int ELFCLASS32 = 1;
constexpr int ELFCLASS64 = 2;

struct Elf32_Ehdr {
    unsigned char e_ident[EI_NIDENT];
    Elf_Half      e_type;
    Elf_Half      e_machine;
    Elf_Word      e_version;
    Elf32_Addr    e_entry;
    Elf32_Off     e_phoff;
    Elf32_Off     e_shoff;
    Elf_Word      e_flags;
    Elf_Half      e_ehsize;
    Elf_Half      e_phentsize;
    Elf_Half      e_phnum;
    Elf_Half      e_shentsize;
    Elf_Half      e_shnum;
    Elf_Half      e_shstrndx;
};

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

} // namespace elf
