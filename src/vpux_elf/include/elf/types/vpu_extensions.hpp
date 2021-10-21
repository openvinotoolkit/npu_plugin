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

#include <elf/types/data_types.hpp>
#include <elf/types/symbol_entry.hpp>
#include <elf/types/section_header.hpp>

namespace elf {

//
// Relocation types
//

constexpr Elf_Word R_VPU_64 = 0;
constexpr Elf_Word R_VPU_64_OR = 1;
constexpr Elf_Word R_VPU_64_OR_RTM = 2;
constexpr Elf_Word R_VPU_64_LSHIFT = 3;

//
// Symbol types
//

constexpr uint8_t VPU_STT_ENTRY = STT_LOOS; // TODO: temporary hack for the loader to easily find the entry;
constexpr uint8_t VPU_STT_INPUT = STT_LOOS + 1;
constexpr uint8_t VPU_STT_OUTPUT = STT_LOOS + 2;

//
// Relocation flags
//

const Elf_Xword VPU_SHF_JIT = SHF_MASKPROC;

//
// Section types
//

const Elf_Word VPU_SHT_DDR = SHT_LOPROC;
const Elf_Word VPU_SHT_TILES = SHT_LOPROC + 1;

//
// Special section indexes
//

constexpr Elf_Word VPU_RT_SYMTAB = SHN_LOOS;

//
// Run-time owned symtab indexes
//

constexpr Elf_Word NNRD_SYM_NNCXM_SLICE_BASE_ADDR = 0;
constexpr Elf_Word NNRD_SYM_RTM_IVAR = 1;
constexpr Elf_Word NNRD_SYM_RTM_ACT = 2;
constexpr Elf_Word NNRD_SYM_RTM_DMA0 = 3;
constexpr Elf_Word NNRD_SYM_RTM_DMA1 = 4;
constexpr Elf_Word NNRD_SYM_FIFO_BASE = 5;
constexpr Elf_Word NNRD_SYM_BARRIERS_START = 6;

}
