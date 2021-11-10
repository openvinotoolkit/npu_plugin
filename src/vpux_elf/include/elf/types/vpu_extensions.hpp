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

/// Originated from DMA src and sdt field https://github.com/movidius/vpuip_2/blob/develop/system/nn_mtl/inference_manager/src/nn_mvnci_executor.cpp#L661
/// Is used also for MappedInference fields (variants, invariants, dma, barriers) relocation
/// Formula: Dst = S + A
constexpr Elf_Word R_VPU_64 = 0;

/// Originated from link_address relocation in DMA in case of linking to DDR
/// https://github.com/movidius/vpuip_2/blob/develop/system/nn_mtl/inference_manager/src/nn_mvnci_executor.cpp#L707
/// Formula: Dst |= S + A
constexpr Elf_Word R_VPU_64_OR = 1;

/// Originated from link_address relocation in DMA in case of linking to CMX slot
/// https://github.com/movidius/vpuip_2/blob/develop/system/nn_mtl/inference_manager/src/nn_mvnci_executor.cpp#L708
/// Formula: Dst |= S + (A * (Dst & (Z - 1)))
/// A - contains the size of element being relocated
/// Dst - contains an index of CMX slot
/// Z - the size of CMX locator
constexpr Elf_Word R_VPU_64_OR_RTM = 2;

/// Originated from barriers relocation, e.g. https://github.com/movidius/vpuip_2/blob/develop/system/nn_mtl/inference_manager/src/nn_mvnci_executor.cpp#L696
/// Formula: Dst <<= S
constexpr Elf_Word R_VPU_64_LSHIFT = 3;

/// The same as R_VPU_64 but applied for the target address being reinterpreted as uint32_t*
constexpr Elf_Word R_VPU_32 = 4;

/// The same as R_VPU_64_OR_RTM but applied for the target address being reinterpreted as uint32_t*
constexpr Elf_Word R_VPU_32_OR_RTM = 5;

/// Originated from https://github.com/movidius/vpuip_2/blob/develop/system/nn_mtl/nce_lib/src/nn_nce_lib_utils.cpp#L196
/// and https://github.com/movidius/vpuip_2/blob/develop/system/nn_mtl/inference_manager/src/nn_mvnci_executor.cpp#L431
/// Formula: Dst += S + A
constexpr Elf_Word R_VPU_32_SUM = 6;

/// Originated from https://github.com/movidius/vpuip_2/blob/develop/system/nn_mtl/nce_lib/src/nn_nce_lib.cpp#L727
/// Formula: Dst = RelativeAddress::to_dpu_multicast_base(S + A)
constexpr Elf_Word R_VPU_32_MULTICAST_BASE = 7;

/// Originated from https://github.com/movidius/vpuip_2/blob/develop/system/nn_mtl/nce_lib/src/nn_nce_lib.cpp#L717
/// Formula: Dst = RelativeAddress::to_dpu_multicast_base(S + A) - Dst
constexpr Elf_Word R_VPU_32_MULTICAST_BASE_SUB = 8;

/// Originated from https://github.com/movidius/vpuip_2/blob/develop/system/nn_mtl/nce_lib/src/nn_nce_lib.cpp#L723
/// Formula: Dst |= offs[Dst] << 4
/// Where offs we get from:
///     unsigned int offs[3] = {SLICE_LENGTH >> 4, SLICE_LENGTH >> 4, SLICE_LENGTH >> 4}; // 1024 * 1024 >> 4 as HW requirement
//      RelativeAddress::to_dpu_multicast(S + A, offs[0], offs[1], offs[2]);
constexpr Elf_Word R_VPU_32_MULTICAST_OFFSET_4_BIT_SHIFT_OR = 9;

/// Originated from https://github.com/movidius/vpuip_2/blob/develop/system/nn_mtl/nce_lib/src/nn_nce_lib.cpp#L722
/// Formula: Dst |= offs[Dst] != 0
/// Where offs we get from:
///     unsigned int offs[3] = {SLICE_LENGTH >> 4, SLICE_LENGTH >> 4, SLICE_LENGTH >> 4};
//      RelativeAddress::to_dpu_multicast(S + A, offs[0], offs[1], offs[2]);
constexpr Elf_Word R_VPU_32_MULTICAST_OFFSET_CMP_OR = 10;

//
// Symbol types
//

constexpr uint8_t VPU_STT_ENTRY = STT_LOOS; // TODO: temporary hack for the loader to easily find the entry;
constexpr uint8_t VPU_STT_INPUT = STT_LOOS + 1;
constexpr uint8_t VPU_STT_OUTPUT = STT_LOOS + 2;

//
// Relocation flags
//

const Elf_Xword VPU_SHF_JIT = 0x10000000;
const Elf_Xword VPU_SHF_USERINPUT = 0x20000000;
const Elf_Xword VPU_SHF_USEROUTPUT = 0x40000000;


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
