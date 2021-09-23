// {% copyright %}

#pragma once

#include <elf/types/elf_header.hpp>
#include <elf/types/section_header.hpp>
#include <elf/types/program_header.hpp>

namespace elf {

template <Elf_Half type>
struct HeaderTypes {};

template <>
struct HeaderTypes<ELFCLASS32> {
    using Elf_EHdr = Elf32_Ehdr;
    using Elf_PHdr = Elf32_Phdr;
    using Elf_SHdr = Elf32_Shdr;
};

template <>
struct HeaderTypes<ELFCLASS64> {
    using Elf_EHdr = Elf64_Ehdr;
    using Elf_PHdr = Elf64_Phdr;
    using Elf_SHdr = Elf64_Shdr;
};

} // namespace elf
