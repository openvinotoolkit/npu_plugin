//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/Types.h>

namespace vpux {

//
// Activation compression
//

constexpr uint32_t ACT_COMPRESSION_RESERVED_MEM_SIZE = 64;
constexpr uint32_t ACT_COMPRESSION_LUT_SIZE = 32;
constexpr uint32_t ACT_COMPRESSION_BUF_SIZE_ALIGNMENT = 32;

// For compression reserved size of buffer needs to be updated for worst case compression
int64_t updateSizeForCompression(int64_t size);

}  // namespace vpux
