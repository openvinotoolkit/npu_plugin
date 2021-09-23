// {% copyright %}

#pragma once

#include <cstdint>

namespace elf {

using Elf_Half = uint16_t;
using Elf_Sword = int32_t;
using Elf_Word = uint32_t;
using Elf_Sword = int32_t;

using Elf_Xword = uint64_t;
using Elf_Sxword = int64_t;

using Elf32_Addr = uint32_t;
using Elf32_Off = uint32_t;
using Elf64_Addr = uint64_t;
using Elf64_Off = uint64_t;

} // namespace elf
