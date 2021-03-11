//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/IERT/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Identifier.h>
#include <mlir/IR/Types.h>

using namespace vpux;

void vpux::IERT::RunTimeResourcesOp::build(mlir::OpBuilder& builder, ::mlir::OperationState& state) {
    ensureTerminator(*state.addRegion(), builder, state.location);
    ensureTerminator(*state.addRegion(), builder, state.location);
    ensureTerminator(*state.addRegion(), builder, state.location);
    ensureTerminator(*state.addRegion(), builder, state.location);
}

static IERT::ExecutorResourceOp getAvailableExecutor(::mlir::Region& executor, const mlir::Attribute& kind) {
    for (auto res : executor.getOps<IERT::ExecutorResourceOp>()) {
        if (res.kindAttr() == kind) {
            return res;
        }
    }

    return nullptr;
}

static IERT::ExecutorResourceOp addAvailableExecutor(mlir::MLIRContext* ctx, mlir::Location loc, mlir::Region& executor,
                                                     mlir::Attribute kind, uint32_t count) {
    auto countAttr = getInt32Attr(ctx, count);

    auto builder = mlir::OpBuilder::atBlockTerminator(&executor.front());
    auto execResource = builder.create<IERT::ExecutorResourceOp>(loc, kind, countAttr);
    vpux::IERT::ExecutorResourceOp::ensureTerminator(execResource.getRegion(), builder, loc);
    return execResource;
}

mlir::LogicalResult vpux::IERT::verifyOp(IERT::RunTimeResourcesOp op) {
    for (auto& resOp : op.availableMemory().getOps()) {
        if (!mlir::isa<IERT::MemoryResourceOp>(&resOp) && !mlir::isa<IERT::EndOp>(&resOp)) {
            return errorAt(op, "Got unsupported Operation '{0}' at '{1}' in 'availableMemory' region", resOp.getName(),
                           resOp.getLoc());
        }
    }
    for (auto& resOp : op.usedMemory().getOps()) {
        if (!mlir::isa<IERT::MemoryResourceOp>(&resOp) && !mlir::isa<IERT::EndOp>(&resOp)) {
            return errorAt(op, "Got unsupported Operation '{0}' at '{1}' in 'usedMemory' region", resOp.getName(),
                           resOp.getLoc());
        }
    }

    for (auto& resOp : op.availableExecutors().getOps()) {
        if (!mlir::isa<IERT::ExecutorResourceOp>(&resOp) && !mlir::isa<IERT::EndOp>(&resOp)) {
            return errorAt(op, "Got unsupported Operation '{0}' at '{1}' in 'availableExecutors' region",
                           resOp.getName(), resOp.getLoc());
        }
    }
    for (auto& resOp : op.usedExecutors().getOps()) {
        if (!mlir::isa<IERT::ExecutorResourceOp>(&resOp) && !mlir::isa<IERT::EndOp>(&resOp)) {
            return errorAt(op, "Got unsupported Operation '{0}' at '{1}' in 'usedExecutors' region", resOp.getName(),
                           resOp.getLoc());
        }
    }

    return mlir::success();
}

IERT::RunTimeResourcesOp vpux::IERT::RunTimeResourcesOp::getFromModule(mlir::ModuleOp module) {
    auto ops = to_small_vector(module.getOps<IERT::RunTimeResourcesOp>());

    if (ops.empty()) {
        return nullptr;
    }

    VPUX_THROW_UNLESS(ops.size() == 1,
                      "Can't have more than one 'IERT::RunTimeResources' Operation in Module, got '{0}'", ops.size());

    return ops.front();
}

IERT::MemoryResourceOp vpux::IERT::RunTimeResourcesOp::addAvailableMemory(mlir::Attribute kind, Byte size) {
    VPUX_THROW_UNLESS(size.count() > 0, "Trying to set zero size of memory kind '{0}'", kind);

    for (auto res : availableMemory().getOps<IERT::MemoryResourceOp>()) {
        VPUX_THROW_UNLESS(kind != res.kindAttr(), "Available memory kind '{0}' was already added", kind);
    }

    auto byteSizeAttr = getInt64Attr(getContext(), size.count());

    auto builder = mlir::OpBuilder::atBlockTerminator(&availableMemory().front());
    return builder.create<IERT::MemoryResourceOp>(getLoc(), kind, byteSizeAttr);
}

IERT::MemoryResourceOp vpux::IERT::RunTimeResourcesOp::getAvailableMemory(mlir::Attribute kind) {
    for (auto res : availableMemory().getOps<IERT::MemoryResourceOp>()) {
        if (res.kindAttr() == kind) {
            return res;
        }
    }

    return nullptr;
}

IERT::MemoryResourceOp vpux::IERT::RunTimeResourcesOp::setUsedMemory(mlir::Attribute kind, Byte size) {
    auto available = getAvailableMemory(kind);
    VPUX_THROW_UNLESS(available != nullptr, "Memory kind '{0}' is not registered as available", kind);
    VPUX_THROW_UNLESS(size <= available.size(), "Memory kind '{0}' used size '{1}' exceeds available size '{2}'", kind,
                      size, available.size());

    auto byteSizeAttr = getInt64Attr(getContext(), size.count());

    for (auto res : usedMemory().getOps<IERT::MemoryResourceOp>()) {
        if (res.kindAttr() == kind) {
            res.byteSizeAttr(byteSizeAttr);
            return res;
        }
    }

    auto builder = mlir::OpBuilder::atBlockTerminator(&usedMemory().front());
    return builder.create<IERT::MemoryResourceOp>(getLoc(), kind, byteSizeAttr);
}

IERT::MemoryResourceOp vpux::IERT::RunTimeResourcesOp::getUsedMemory(mlir::Attribute kind) {
    for (auto res : usedMemory().getOps<IERT::MemoryResourceOp>()) {
        if (res.kindAttr() == kind) {
            return res;
        }
    }

    return nullptr;
}

IERT::ExecutorResourceOp vpux::IERT::RunTimeResourcesOp::addAvailableExecutor(mlir::Attribute kind, uint32_t count) {
    VPUX_THROW_UNLESS(count > 0, "Trying to set zero count of executor kind '{0}'", kind);

    for (auto res : availableExecutors().getOps<IERT::ExecutorResourceOp>()) {
        VPUX_THROW_UNLESS(kind != res.kindAttr(), "Available executor kind '{0}' was already added", kind);
    }
    return ::addAvailableExecutor(getContext(), getLoc(), availableExecutors(), kind, count);
}

IERT::ExecutorResourceOp vpux::IERT::RunTimeResourcesOp::getAvailableExecutor(mlir::Attribute kind) {
    return ::getAvailableExecutor(availableExecutors(), kind);
}

IERT::ExecutorResourceOp vpux::IERT::RunTimeResourcesOp::setUsedExecutor(mlir::Attribute kind, uint32_t count) {
    auto available = getAvailableExecutor(kind);
    VPUX_THROW_UNLESS(available != nullptr, "Executor kind '{0}' is not registered as available", kind);
    VPUX_THROW_UNLESS(count <= available.count(), "Executor kind '{0}' used count '{1}' exceeds available count '{2}'",
                      kind, count, available.count());

    auto countAttr = getInt32Attr(getContext(), count);

    for (auto res : usedExecutors().getOps<IERT::ExecutorResourceOp>()) {
        if (res.kindAttr() == kind) {
            res.countAttr(countAttr);
            return res;
        }
    }

    auto builder = mlir::OpBuilder::atBlockTerminator(&usedExecutors().front());
    auto execResource = builder.create<IERT::ExecutorResourceOp>(getLoc(), kind, countAttr);
    ensureTerminator(execResource.getRegion(), builder, getLoc());
    return execResource;
}

IERT::ExecutorResourceOp vpux::IERT::RunTimeResourcesOp::getUsedExecutor(mlir::Attribute kind) {
    for (auto res : usedExecutors().getOps<IERT::ExecutorResourceOp>()) {
        if (res.kindAttr() == kind) {
            return res;
        }
    }

    return nullptr;
}

IERT::ExecutorResourceOp vpux::IERT::ExecutorResourceOp::addAvailableExecutor(mlir::Attribute kind, uint32_t count) {
    VPUX_THROW_UNLESS(count > 0, "Trying to set zero count of executor kind '{0}'", kind);

    for (auto res : subExecutors().getOps<IERT::ExecutorResourceOp>()) {
        VPUX_THROW_UNLESS(kind != res.kindAttr(), "Available executor kind '{0}' was already added", kind);
    }
    return ::addAvailableExecutor(getContext(), getLoc(), subExecutors(), kind, count);
}

IERT::ExecutorResourceOp vpux::IERT::ExecutorResourceOp::getAvailableExecutor(mlir::Attribute kind) {
    return ::getAvailableExecutor(subExecutors(), kind);
}
