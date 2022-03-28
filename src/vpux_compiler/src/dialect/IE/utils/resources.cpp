//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/Builders.h>

using namespace vpux;

//
// MemoryResourceOp
//

IE::MemoryResourceOp vpux::IE::addAvailableMemory(mlir::ModuleOp mainModule, mlir::StringAttr memSpace, Byte size) {
    VPUX_THROW_UNLESS(size.count() > 0, "Trying to set zero size of memory kind '{0}'", memSpace);

    auto res = mainModule.lookupSymbol<MemoryResourceOp>(memSpace);
    VPUX_THROW_UNLESS(res == nullptr, "Available memory kind '{0}' was already added", memSpace);

    const auto byteSizeAttr = getIntAttr(mainModule->getContext(), size.count());
    auto builder = mlir::OpBuilder::atBlockBegin(mainModule.getBody());
    return builder.create<IE::MemoryResourceOp>(mainModule.getLoc(), memSpace, byteSizeAttr);
}

IE::MemoryResourceOp vpux::IE::getAvailableMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace) {
    return mainModule.lookupSymbol<IE::MemoryResourceOp>(memSpace);
}

IE::MemoryResourceOp vpux::IE::setUsedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace, Byte size) {
    auto available = IE::getAvailableMemory(mainModule, memSpace);
    VPUX_THROW_UNLESS(available != nullptr, "Memory kind '{0}' is not registered as available", memSpace);
    VPUX_THROW_UNLESS(size <= available.size(), "Memory kind '{0}' used size '{1}' exceeds available size '{2}'",
                      memSpace, size, available.size());

    auto byteSizeAttr = getIntAttr(mainModule->getContext(), size.count());
    auto memSpaceAttr = memSpace.getLeafReference();
    auto mainBuilder = mlir::OpBuilder::atBlockBegin(mainModule.getBody());

    auto usedMemModule = mainModule.lookupSymbol<mlir::ModuleOp>(usedMemModuleName);
    if (usedMemModule == nullptr) {
        usedMemModule = mainBuilder.create<mlir::ModuleOp>(mainModule->getLoc(), usedMemModuleName);
    }

    auto res = usedMemModule.lookupSymbol<IE::MemoryResourceOp>(memSpace.getLeafReference());
    if (res != nullptr) {
        res.byteSizeAttr(byteSizeAttr);
        return res;
    }

    auto innerBuilder = mlir::OpBuilder::atBlockBegin(usedMemModule.getBody());
    return innerBuilder.create<IE::MemoryResourceOp>(usedMemModule->getLoc(), memSpaceAttr, byteSizeAttr);
}

IE::MemoryResourceOp vpux::IE::getUsedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace) {
    auto usedMemModule = mainModule.lookupSymbol<mlir::ModuleOp>(usedMemModuleName);
    if (usedMemModule == nullptr) {
        return nullptr;
    }

    return usedMemModule.lookupSymbol<IE::MemoryResourceOp>(memSpace);
}

SmallVector<IE::MemoryResourceOp> vpux::IE::getUsedMemory(mlir::ModuleOp mainModule) {
    auto usedMemModule = mainModule.lookupSymbol<mlir::ModuleOp>(usedMemModuleName);
    if (usedMemModule == nullptr) {
        return {};
    }

    return to_small_vector(usedMemModule.getOps<IE::MemoryResourceOp>());
}

//
// ExecutorResourceOp
//

IE::ExecutorResourceOp vpux::IE::details::addExecutor(mlir::SymbolTable mainModule, mlir::Region& region,
                                                      mlir::StringAttr executorAttr, uint32_t count) {
    VPUX_THROW_UNLESS(count > 0, "Trying to set zero count of executor kind '{0}'", executorAttr);

    auto* ctx = region.getContext();
    auto res = mainModule.lookup<ExecutorResourceOp>(executorAttr);
    VPUX_THROW_UNLESS(res == nullptr, "Available executor kind '{0}' was already added", executorAttr);

    const auto countAttr = getIntAttr(ctx, count);
    auto builder = mlir::OpBuilder::atBlockBegin(&region.front());
    auto resOp = builder.create<IE::ExecutorResourceOp>(region.getLoc(), executorAttr, countAttr);

    // Operations with a 'SymbolTable' must have exactly one block
    resOp.getRegion().emplaceBlock();
    return resOp;
}

IE::ExecutorResourceOp vpux::IE::getAvailableExecutor(mlir::ModuleOp mainModule, mlir::SymbolRefAttr executorAttr) {
    return mlir::dyn_cast_or_null<IE::ExecutorResourceOp>(mlir::SymbolTable::lookupSymbolIn(mainModule, executorAttr));
}

IE::ExecutorResourceOp vpux::IE::ExecutorResourceOp::addSubExecutor(mlir::StringAttr executorAttr, uint32_t count) {
    return details::addExecutor(getOperation(), getRegion(), executorAttr, count);
}

IE::ExecutorResourceOp vpux::IE::ExecutorResourceOp::getSubExecutor(mlir::StringAttr executorAttr) {
    return lookupSymbol<IE::ExecutorResourceOp>(executorAttr);
}
