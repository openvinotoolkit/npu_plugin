//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once
#include <cstdint>

namespace mvds {
namespace nce2p7 {

const uint64_t CMX_ADDRESS_MASK{ (1ULL << 21) - 1ULL };
const uint64_t CMX_BASE_ADDRESS[]{ 0x2E000000,0x2E200000 };
const uint64_t DRAM_BASE_ADRESS{ 0x80000000 };
const uint64_t PROGRAMMABLE_INPUT_BASE_ADRESS{ 0x8E000000 };
const uint64_t PROGRAMMABLE_OUTPUT_BASE_ADRESS{ 0x8F000000 };
const uint32_t ACT_KERNEL_RUNTIME_WINDOW{ 0x1C000000 };
const uint32_t ACT_KERNEL_CMX_WINDOW{ 0x1F000000 };
const uint32_t ACT_KERNEL_TEXT_WINDOW{ 0x1D000000 };
const uint32_t ACT_KERNEL_DATA_WINDOW{ 0x1E000000 };

} // namespace nce2p7
} // namespace mvds
