//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/allocate_buffers.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/types.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

//
// allocateBuffersOfType
//

SmallVector<mlir::Value> vpux::allocateBuffersOfType(const Logger& log, mlir::Location loc, mlir::OpBuilder& builder,
                                                     mlir::Type bufferType, bool individualBuffers) {
    auto createAllocOp = [&](mlir::Type type) {
        if (type == nullptr) {
            return mlir::Value();
        } else if (auto memref = type.dyn_cast<mlir::MemRefType>()) {
            return static_cast<mlir::Value>(builder.create<mlir::memref::AllocOp>(loc, memref).getMemref());
        } else if (auto distributedBuffer = type.dyn_cast<VPUIP::DistributedBufferType>()) {
            return builder.create<VPURT::AllocDistributed>(loc, distributedBuffer, nullptr, nullptr).getBuffer();
        }
        VPUX_THROW("Unexpected type to allocate: {0}", type);
    };

    if (bufferType.isa<mlir::MemRefType, VPUIP::DistributedBufferType>()) {
        log.trace("Allocating result buffer of type '{0}'", bufferType);
        return {createAllocOp(bufferType)};
    } else if (auto sparseBufferType = bufferType.dyn_cast<VPUIP::SparseBufferType>()) {
        log.trace("Allocating result buffers of type '{0}'", sparseBufferType);

        auto dataBuffer = createAllocOp(sparseBufferType.getData());
        auto sparsityMapBuffer = createAllocOp(sparseBufferType.getSparsityMap());
        auto seTableBuffer = createAllocOp(sparseBufferType.getStorageElementTable());

        if (!individualBuffers) {
            auto groupOp = builder.create<VPUIP::GroupSparseBufferOp>(
                    loc, dataBuffer, sparsityMapBuffer, seTableBuffer, sparseBufferType.getIsWeights(),
                    sparseBufferType.getCompressionScheme(), sparseBufferType.getSeAttr());
            return {groupOp.getOutput()};
        }

        SmallVector<mlir::Value> buffers{dataBuffer};
        if (sparsityMapBuffer != nullptr) {
            buffers.push_back(sparsityMapBuffer);
        }
        if (seTableBuffer != nullptr) {
            buffers.push_back(seTableBuffer);
        }
        return buffers;
    }
    VPUX_THROW("Unexpected type to allocate {0}", bufferType);
}

//
// allocateBuffers & allocateBuffersForValue using bufferizable interface
//

SmallVector<mlir::Value> vpux::allocateBuffersForValue(const Logger& log, mlir::Location loc, mlir::OpBuilder& builder,
                                                       mlir::Value value,
                                                       const mlir::bufferization::BufferizationOptions& options,
                                                       bool individualBuffers) {
    auto bufferType = vpux::getBufferType(value, options);
    log.nest().trace("Allocating result buffer of type '{0}' for value type '{1}'", bufferType, value.getType());
    return allocateBuffersOfType(log.nest(), loc, builder, bufferType, individualBuffers);
}

SmallVector<mlir::Value> vpux::allocateBuffers(const Logger& log, mlir::Location loc, mlir::OpBuilder& builder,
                                               mlir::ValueRange values,
                                               const mlir::bufferization::BufferizationOptions& options,
                                               bool individualBuffers) {
    SmallVector<mlir::Value> buffers;
    for (const auto& value : values) {
        const auto valueBuffers = allocateBuffersForValue(log, loc, builder, value, options, individualBuffers);
        buffers.append(valueBuffers.begin(), valueBuffers.end());
    }
    return buffers;
}

//
// allocateBuffers & allocateBuffersForValue using typeConverter & allocateBuffersAdaptor
// Note: remove after one-shot bufferization is fully implemented E#102424
//

SmallVector<mlir::Value> vpux::allocateBuffersForValue(const Logger& log, mlir::Location loc, mlir::OpBuilder& builder,
                                                       mlir::TypeConverter& typeConverter, mlir::Value value,
                                                       bool individualBuffers) {
    auto origType = value.getType();
    auto bufferType = typeConverter.convertType(origType);

    log.nest().trace("Allocating result buffer of type '{0}' for value type '{1}'", bufferType, value.getType());
    return allocateBuffersOfType(log.nest(), loc, builder, bufferType, individualBuffers);
}

SmallVector<mlir::Value> vpux::allocateBuffers(const Logger& log, mlir::Location loc, mlir::OpBuilder& builder,
                                               mlir::TypeConverter& typeConverter, mlir::ValueRange values,
                                               bool individualBuffers) {
    SmallVector<mlir::Value> buffers;
    for (const auto& value : values) {
        const auto valueBuffers = allocateBuffersForValue(log, loc, builder, typeConverter, value, individualBuffers);
        buffers.append(valueBuffers.begin(), valueBuffers.end());
    }
    return buffers;
}

SmallVector<mlir::Value> vpux::allocateBuffersAdaptor(
        const Logger& log, mlir::Location loc, mlir::OpBuilder& builder, mlir::ValueRange values,
        const std::optional<mlir::bufferization::BufferizationOptions>& options,
        std::optional<std::reference_wrapper<mlir::TypeConverter>> typeConverter, bool individualBuffers) {
    return options.has_value() ? allocateBuffers(log, loc, builder, values, options.value(), individualBuffers)
                               : allocateBuffers(log, loc, builder, typeConverter.value(), values, individualBuffers);
}
