//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <stdint.h>
#include <utility>

#pragma once

namespace vpux {

std::pair<uint8_t, uint8_t> reduceWaitMaskTo8bit(uint64_t waitMask);

}  // namespace vpux
