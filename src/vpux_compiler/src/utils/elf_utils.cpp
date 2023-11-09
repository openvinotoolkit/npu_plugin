//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/elf_utils.hpp"

using namespace vpux;

std::pair<uint8_t, uint8_t> vpux::reduceWaitMaskTo8bit(uint64_t waitMask) {
    uint8_t barrier_group = 0;
    uint8_t barrier_mask = 0;
    for (uint64_t mask = waitMask, group = 1; mask > 0; mask >>= 8, ++group) {
        if (mask & 0xff) {
            if (barrier_group == 0) {
                barrier_group = static_cast<unsigned char>(group);
                barrier_mask = mask & 0xff;
            } else {
                barrier_group = 0;
                barrier_mask = 0;
                break;
            }
        }
    }
    return {barrier_group, barrier_mask};
}
