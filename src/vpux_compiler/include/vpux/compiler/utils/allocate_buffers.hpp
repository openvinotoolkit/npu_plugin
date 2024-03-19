//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Bufferization/Transforms/Bufferize.h>
#include <mlir/IR/Value.h>

#include "vpux/compiler/dialect/VPUIP/types.hpp"

namespace vpux {

SmallVector<mlir::Value> allocateBuffersOfType(const Logger& log, mlir::Location loc, mlir::OpBuilder& builder,
                                               mlir::Type bufferType, bool individualBuffers = false);

//
// allocateBuffers & allocateBuffersForValue using bufferizable interface
//

SmallVector<mlir::Value> allocateBuffersForValue(const Logger& log, mlir::Location loc, mlir::OpBuilder& builder,
                                                 mlir::Value value,
                                                 const mlir::bufferization::BufferizationOptions& options,
                                                 bool individualBuffers = false);

SmallVector<mlir::Value> allocateBuffers(const Logger& log, mlir::Location loc, mlir::OpBuilder& builder,
                                         mlir::ValueRange values,
                                         const mlir::bufferization::BufferizationOptions& options,
                                         bool individualBuffers = false);

//
// allocateBuffers & allocateBuffersForValue using typeConverter & allocateBuffersAdaptor
// Note: remove after one-shot bufferization is fully implemented  E#102424
//

SmallVector<mlir::Value> allocateBuffersForValue(const Logger& log, mlir::Location loc, mlir::OpBuilder& builder,
                                                 mlir::TypeConverter& typeConverter, mlir::Value value,
                                                 bool individualBuffers = false);

SmallVector<mlir::Value> allocateBuffers(const Logger& log, mlir::Location loc, mlir::OpBuilder& builder,
                                         mlir::TypeConverter& typeConverter, mlir::ValueRange values,
                                         bool individualBuffers = false);

SmallVector<mlir::Value> allocateBuffersAdaptor(
        const Logger& log, mlir::Location loc, mlir::OpBuilder& builder, mlir::ValueRange values,
        const std::optional<mlir::bufferization::BufferizationOptions>& options,
        std::optional<std::reference_wrapper<mlir::TypeConverter>> typeConverter, bool individualBuffers);

}  // namespace vpux
