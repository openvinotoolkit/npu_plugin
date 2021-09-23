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
