//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once
#include <cstdint>

namespace mvds {
namespace nce2p7 {

constexpr uint32_t CMX_ADDRESS_MASK{(1ULL << 21) - 1ULL};
constexpr uint32_t CMX_BASE_ADDRESS[]{0x2E000000, 0x2E200000};
constexpr uint32_t CMX_SLICE_SIZE{CMX_BASE_ADDRESS[1] - CMX_BASE_ADDRESS[0]};

constexpr uint32_t ACT_KERNEL_RUNTIME_WINDOW{0x1C000000};
constexpr uint32_t ACT_KERNEL_CMX_WINDOW{0x1F000000};
constexpr uint32_t ACT_KERNEL_TEXT_WINDOW{0x1D000000};
constexpr uint32_t ACT_KERNEL_DATA_WINDOW{0x1E000000};

}  // namespace nce2p7
}  // namespace mvds
