//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Dialect/Bufferization/Transforms/Bufferize.h>
#include <mlir/IR/Value.h>

#include "vpux/compiler/dialect/VPUIP/types.hpp"

namespace vpux {

SmallVector<mlir::Value> allocateBuffersOfType(const Logger& log, mlir::Location loc, mlir::OpBuilder& builder,
                                               mlir::Type bufferType, bool individualBuffers = false);

SmallVector<mlir::Value> allocateBuffersForValue(const Logger& log, mlir::Location loc, mlir::OpBuilder& builder,
                                                 mlir::TypeConverter& typeConverter, mlir::Value value,
                                                 bool individualBuffers = false);

SmallVector<mlir::Value> allocateBuffers(const Logger& log, mlir::Location loc, mlir::OpBuilder& builder,
                                         mlir::TypeConverter& typeConverter, mlir::ValueRange values,
                                         bool individualBuffers = false);

}  // namespace vpux
