//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
#include "vpux/compiler/utils/compression_utils.hpp"

using namespace vpux;

int64_t vpux::updateSizeForCompression(int64_t size) {
    // In worst case scenario depending on the content of activation, its final size after
    // compression might be bigger than original size. Compiler before performing DDR
    // allocation needs to adjust required size by this buffer
    // Formula from HAS is following:
    //   DTS = X * Y * Z * (element size in bytes)
    //   denseSize = (DTS * (65/64)) + 1
    //   DDR Allocation (32B aligned) = denseSize + ( (denseSize % 32) ? (32 – (denseSize % 32) : 0)
    auto worstCaseSize = static_cast<int64_t>(size * 65 / 64) + 1;
    if (worstCaseSize % ACT_COMPRESSION_BUF_SIZE_ALIGNMENT) {
        worstCaseSize += ACT_COMPRESSION_BUF_SIZE_ALIGNMENT - worstCaseSize % ACT_COMPRESSION_BUF_SIZE_ALIGNMENT;
    }
    return worstCaseSize;
}
