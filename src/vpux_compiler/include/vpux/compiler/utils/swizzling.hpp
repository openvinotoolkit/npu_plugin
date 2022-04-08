//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Types.h>
#include "vpux/compiler/dialect/VPU/types.hpp"
#include "vpux/utils/core/mem_size.hpp"

namespace vpux {

//
// Swizzling constants
//

// Swizzling key 5 requires swizzled buffer to have start address aligned to 16k
constexpr uint64_t SWIZZLING_KEY_5 = 5;

// Swizzled buffers need to have space in CMX of size aligned to 512, because this
// it the smallest unit of CMX RAM cut
constexpr int64_t SWIZZLING_SIZE_ALIGNMENT = 512;

// For swizzling buffer size needs to be aligned to 512 as required by HW
int64_t alignSizeForSwizzling(int64_t size);

/**
 * @brief Required alignment of buffers in CMX memory required swizzling operations
 *
 * @param swizzlingKey
 * @return alignment [bytes]
 */
int64_t getAlignmentForSwizzling(int64_t swizzlingKey);

/**
 * @brief calculate size of buffers with requested initial memory allocation offset and fixed minimal allocation
 * increments
 *
 * @param bufferSizes - reference to vector containing sizes [bytes] of buffers to be allocated
 * @param offsetAlignment - alignment of buffer start [bytes]. The default value corresponds to offset required for
 * vpux::SWIZZLING_KEY_5. Must be > 0. The start address ADDR of any allocated buffer yields
 * ADDR % offsetAlignment == 0.
 *
 * @param sizeAlignment - memory allocation increment. The memory is allocated  in units of sizeAlignment so that
 * bufferSize[i] % sizeAlignment == 0. The default value corresponds to
 * vpux::SWIZZLING_SIZE_ALIGNMENT.
 *
 * @return required memory taking into account the allocation requirements for swizzled buffers [bytes].
 *
 */
Byte calculateAlignedBuffersMemoryRequirement(
        mlir::SmallVector<Byte>& bufferSizes,
        const Byte offsetAlignment = Byte(getAlignmentForSwizzling(SWIZZLING_KEY_5)),
        const Byte sizeAlignment = Byte(SWIZZLING_SIZE_ALIGNMENT));

// Retrieve swizzling key setting embedded in layout with buffer types
mlir::IntegerAttr getSwizzlingKeyAttr(mlir::Type type);

int64_t getSwizzlingKey(mlir::Type type);

mlir::Type setSwizzlingKey(mlir::Type type, mlir::IntegerAttr swizzlingKeyAttr);

mlir::Type setSwizzlingKey(mlir::Type type, int64_t swizzlingKey);

}  // namespace vpux
