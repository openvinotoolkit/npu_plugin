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
/// Refer to https://docs.oracle.com/cd/E23824_01/html/819-0690/chapter6-54839.html
/// for the detailed description of the values and structures below
///

struct Elf64_Rel {
    Elf64_Addr r_offset;
    Elf_Xword  r_info;
};

struct Elf64_Rela {
    Elf64_Addr r_offset;
    Elf_Xword  r_info;
    Elf_Sxword r_addend;
};

struct Elf32_Rel {
    Elf32_Addr r_offset;
    Elf_Word   r_info;
};

struct Elf32_Rela {
    Elf32_Addr r_offset;
    Elf_Word   r_info;
    Elf_Sword  r_addend;
};

using RelocationEntry = Elf64_Rel;
using RelocationAEntry = Elf64_Rela;
using RelocationEntry32 = Elf32_Rel;
using RelocationAEntry32 = Elf32_Rela;

//! Extract symbol index from info
Elf_Word elf64RSym(Elf_Xword info);

//! Extract relocation type from info
Elf_Word elf64RType(Elf_Xword info);

//! Pack relocation type and symbol index into info
Elf_Xword elf64RInfo(Elf_Word sym, Elf_Word type);

} // namespace elf
