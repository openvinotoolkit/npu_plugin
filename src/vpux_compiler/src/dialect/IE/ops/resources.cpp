//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/IERT/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Identifier.h>
#include <mlir/IR/Types.h>

using namespace vpux;

//
// Common utilities
//

namespace {

IE::ExecutorResourceOp getExecutor(mlir::Region& executor, mlir::Attribute kind) {
    for (auto res : executor.getOps<IE::ExecutorResourceOp>()) {
        if (res.kind() == kind) {
            return res;
        }
    }

    return nullptr;
}

IE::ExecutorResourceOp addExecutor(mlir::Location loc, mlir::Region& executor, mlir::Attribute kind, uint32_t count,
                                   bool withSubRegion) {
    VPUX_THROW_UNLESS(count > 0, "Trying to set zero count of executor kind '{0}'", kind);

    for (auto res : executor.getOps<IE::ExecutorResourceOp>()) {
        VPUX_THROW_UNLESS(kind != res.kind(), "Executor kind '{0}' was already added", kind);
    }

    auto countAttr = getIntAttr(loc.getContext(), count);
    auto builder = mlir::OpBuilder::atBlockEnd(&executor.front());
    auto resOp = builder.create<IE::ExecutorResourceOp>(loc, kind, countAttr, withSubRegion ? 1 : 0);
    if (withSubRegion) {
        resOp.subExecutors().front().emplaceBlock();
    }
    return resOp;
}

}  // namespace

//
// RunTimeResourcesOp
//

void vpux::IE::RunTimeResourcesOp::build(mlir::OpBuilder&, mlir::OperationState& state) {
    state.addRegion()->emplaceBlock();
    state.addRegion()->emplaceBlock();
    state.addRegion()->emplaceBlock();
}

mlir::LogicalResult vpux::IE::verifyOp(IE::RunTimeResourcesOp op) {
    for (auto& resOp : op.availableMemory().getOps()) {
        if (!mlir::isa<IE::MemoryResourceOp>(&resOp)) {
            return errorAt(op, "Got unsupported Operation '{0}' at '{1}' in 'availableMemory' region", resOp.getName(),
                           resOp.getLoc());
        }
    }

    for (auto& resOp : op.usedMemory().getOps()) {
        if (!mlir::isa<IE::MemoryResourceOp>(&resOp)) {
            return errorAt(op, "Got unsupported Operation '{0}' at '{1}' in 'usedMemory' region", resOp.getName(),
                           resOp.getLoc());
        }
    }

    for (auto& resOp : op.executors().getOps()) {
        if (!mlir::isa<IE::ExecutorResourceOp>(&resOp)) {
            return errorAt(op, "Got unsupported Operation '{0}' at '{1}' in 'executors' region", resOp.getName(),
                           resOp.getLoc());
        }
    }

    return mlir::success();
}

IE::RunTimeResourcesOp vpux::IE::RunTimeResourcesOp::getFromModule(mlir::ModuleOp module) {
    auto ops = to_small_vector(module.getOps<IE::RunTimeResourcesOp>());

    if (ops.empty()) {
        return nullptr;
    }

    VPUX_THROW_UNLESS(ops.size() == 1, "Can't have more than one 'IE::RunTimeResources' Operation in Module, got '{0}'",
                      ops.size());

    return ops.front();
}

IE::MemoryResourceOp vpux::IE::RunTimeResourcesOp::addAvailableMemory(mlir::Attribute kind, Byte size) {
    VPUX_THROW_UNLESS(size.count() > 0, "Trying to set zero size of memory kind '{0}'", kind);

    for (auto res : getAvailableMemory()) {
        VPUX_THROW_UNLESS(kind != res.kind(), "Available memory kind '{0}' was already added", kind);
    }

    auto byteSizeAttr = getIntAttr(getContext(), size.count());

    auto builder = mlir::OpBuilder::atBlockEnd(&availableMemory().front());
    return builder.create<IE::MemoryResourceOp>(getLoc(), kind, byteSizeAttr);
}

IE::MemoryResourceOp vpux::IE::RunTimeResourcesOp::getAvailableMemory(mlir::Attribute kind) {
    for (auto res : getAvailableMemory()) {
        if (res.kind() == kind) {
            return res;
        }
    }

    return nullptr;
}

IE::MemoryResourceOp vpux::IE::RunTimeResourcesOp::setUsedMemory(mlir::Attribute kind, Byte size) {
    auto available = getAvailableMemory(kind);
    VPUX_THROW_UNLESS(available != nullptr, "Memory kind '{0}' is not registered as available", kind);
    VPUX_THROW_UNLESS(size <= available.size(), "Memory kind '{0}' used size '{1}' exceeds available size '{2}'", kind,
                      size, available.size());

    auto byteSizeAttr = getIntAttr(getContext(), size.count());

    for (auto res : getUsedMemory()) {
        if (res.kind() == kind) {
            res.byteSizeAttr(byteSizeAttr);
            return res;
        }
    }

    auto builder = mlir::OpBuilder::atBlockEnd(&usedMemory().front());
    return builder.create<IE::MemoryResourceOp>(getLoc(), kind, byteSizeAttr);
}

IE::MemoryResourceOp vpux::IE::RunTimeResourcesOp::getUsedMemory(mlir::Attribute kind) {
    for (auto res : getUsedMemory()) {
        if (res.kind() == kind) {
            return res;
        }
    }

    return nullptr;
}

IE::ExecutorResourceOp vpux::IE::RunTimeResourcesOp::addExecutor(mlir::Attribute kind, uint32_t count,
                                                                 bool withSubRegion) {
    return ::addExecutor(getLoc(), executors(), kind, count, withSubRegion);
}

IE::ExecutorResourceOp vpux::IE::RunTimeResourcesOp::getExecutor(mlir::Attribute kind) {
    return ::getExecutor(executors(), kind);
}

//
// ExecutorResourceOp
//

mlir::LogicalResult vpux::IE::verifyOp(IE::ExecutorResourceOp op) {
    if (!op.subExecutors().empty() && op.subExecutors().size() != 1) {
        return errorAt(op, "Can't have more than one 'subExecutors' region");
    }

    if (!op.subExecutors().empty()) {
        for (auto& resOp : op.subExecutors().front().getOps()) {
            if (!mlir::isa<IE::ExecutorResourceOp>(&resOp)) {
                return errorAt(op, "Got unsupported Operation '{0}' at '{1}' in 'subExecutors' region", resOp.getName(),
                               resOp.getLoc());
            }
        }
    }

    return mlir::success();
}

IE::ExecutorResourceOp vpux::IE::ExecutorResourceOp::addSubExecutor(mlir::Attribute kind, uint32_t count,
                                                                    bool withSubRegion) {
    VPUX_THROW_UNLESS(!subExecutors().empty(), "Executor '{0}' doesn't support sub executors", this->kind());
    return ::addExecutor(getLoc(), subExecutors().front(), kind, count, withSubRegion);
}

IE::ExecutorResourceOp vpux::IE::ExecutorResourceOp::getSubExecutor(mlir::Attribute kind) {
    return subExecutors().empty() ? nullptr : ::getExecutor(subExecutors().front(), kind);
}
