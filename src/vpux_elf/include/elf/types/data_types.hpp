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
