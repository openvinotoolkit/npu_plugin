//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"

#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/mem_size.hpp"

#include <llvm/Support/FormatVariadic.h>

namespace vpux {
namespace VPU {

namespace NCEInvariant {

//
// Constants
//

constexpr int64_t WEIGHT_TABLE_NUM_ELEMENTS_PER_OC = 4;

constexpr int64_t KMB_CMCONV_WIDTH_ALIGNMENT = 16;
constexpr int64_t KMB_CMCONV_CHANNELS_LIMIT = 16;

constexpr int64_t SUPPORTED_BATCH_SIZE = 1;
constexpr int64_t MAX_KERNEL_SIZE = 11;

//
// Precision checks
//

bool isPrecisionSupported(ArchKind arch, mlir::ValueRange vals, LogCb logCb = emptyLogCb);

//
// Attributes checks
//

bool isAttrsSupported(ArchKind arch, int64_t KY, int64_t KX, int64_t SY, int64_t SX, int64_t padTop, int64_t padBottom,
                      int64_t padLeft, int64_t padRight, LogCb logCb = emptyLogCb);

//
// Activation type checks
//

int64_t getAlignment(mlir::Type elemType);

bool isActTypeSupported(vpux::NDTypeInterface type, int64_t alignment, LogCb logCb = emptyLogCb);

//
// PostOp checks
//

bool isPostOpSupported(mlir::Operation* postOp);

//
// WeightsTable information
//

Byte getWeightsTableSize(int64_t OC);

//
// Channel major Convolution
//

bool isChannelMajorCompatible(ArchKind arch, vpux::NDTypeInterface inputType);

//
// Fuse PadOp check
//

bool verifyPads(mlir::ArrayAttr kernelSizeAttr, mlir::ArrayAttr padBeginAttr, mlir::ArrayAttr padEndAttr,
                LogCb logCb = emptyLogCb);
bool verifyPads(int64_t KY, int64_t KX, int64_t padTop, int64_t padBottom, int64_t padLeft, int64_t padRight,
                LogCb logCb = emptyLogCb);

}  // namespace NCEInvariant

}  // namespace VPU
}  // namespace vpux
