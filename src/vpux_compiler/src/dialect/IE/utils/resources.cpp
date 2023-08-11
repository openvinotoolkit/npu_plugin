//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/Builders.h>

using namespace vpux;

//
// MemoryResourceOp
//

IE::MemoryResourceOp vpux::IE::addAvailableMemory(mlir::ModuleOp mainModule, mlir::StringAttr memSpace, Byte size) {
    VPUX_THROW_UNLESS(size.count() > 0, "Trying to set zero size of memory kind '{0}'", memSpace);

    const auto byteSizeAttr = getIntAttr(mainModule->getContext(), size.count());
    auto builder = mlir::OpBuilder::atBlockBegin(mainModule.getBody());
    return builder.create<IE::MemoryResourceOp>(mainModule.getLoc(), memSpace, byteSizeAttr, nullptr);
}

bool vpux::IE::hasAvailableMemory(mlir::ModuleOp mainModule, mlir::StringAttr memSpace) {
    auto res = mainModule.lookupSymbol<IE::MemoryResourceOp>(memSpace);
    return res != nullptr;
}

IE::MemoryResourceOp vpux::IE::getAvailableMemory(mlir::ModuleOp mainModule, mlir::StringAttr memSpace) {
    return mainModule.lookupSymbol<IE::MemoryResourceOp>(memSpace);
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
    return innerBuilder.create<IE::MemoryResourceOp>(usedMemModule->getLoc(), memSpaceAttr, byteSizeAttr, nullptr);
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
// Reserved memory resources
//
SmallVector<IE::MemoryResourceOp> vpux::IE::getReservedMemoryResources(mlir::ModuleOp mainModule,
                                                                       mlir::SymbolRefAttr memSpace) {
    auto resMemModule = mainModule.lookupSymbol<mlir::ModuleOp>(resMemModuleName);
    if (resMemModule == nullptr) {
        return {};
    }

    SmallVector<IE::MemoryResourceOp> resMemVec;

    for (auto resMemModuleOp : resMemModule.getOps<mlir::ModuleOp>()) {
        resMemVec.push_back(resMemModuleOp.lookupSymbol<IE::MemoryResourceOp>(memSpace));
    }

    return resMemVec;
}

//
// DMA profiling reserved memory
//

IE::MemoryResourceOp vpux::IE::setDmaProfilingReservedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace,
                                                             int64_t size) {
    auto* ctx = mainModule->getContext();
    auto byteSizeAttr = getIntAttr(ctx, size);
    auto memSpaceAttr = memSpace.getLeafReference();
    auto mainBuilder = mlir::OpBuilder::atBlockBegin(mainModule.getBody());

    auto resMemModule = mainModule.lookupSymbol<mlir::ModuleOp>(resMemModuleName);
    if (resMemModule == nullptr) {
        resMemModule = mainBuilder.create<mlir::ModuleOp>(mainModule->getLoc(), resMemModuleName);
    }

    auto resMemBuilder = mlir::OpBuilder::atBlockBegin(resMemModule.getBody());

    auto dmaProfilingResMemModule = resMemModule.lookupSymbol<mlir::ModuleOp>(dmaProfilingResMemModuleName);
    if (dmaProfilingResMemModule == nullptr) {
        dmaProfilingResMemModule =
                resMemBuilder.create<mlir::ModuleOp>(resMemModule->getLoc(), dmaProfilingResMemModuleName);
    }

    auto res = dmaProfilingResMemModule.lookupSymbol<IE::MemoryResourceOp>(memSpace.getLeafReference());
    if (res != nullptr) {
        res.byteSizeAttr(byteSizeAttr);
        return res;
    }

    auto innerBuilder = mlir::OpBuilder::atBlockBegin(dmaProfilingResMemModule.getBody());
    return innerBuilder.create<IE::MemoryResourceOp>(dmaProfilingResMemModule->getLoc(), memSpaceAttr, byteSizeAttr,
                                                     nullptr);
}

IE::MemoryResourceOp vpux::IE::getDmaProfilingReservedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace) {
    auto resMemModule = mainModule.lookupSymbol<mlir::ModuleOp>(resMemModuleName);
    if (resMemModule == nullptr) {
        return nullptr;
    }

    auto dmaProfilingResMemModule = resMemModule.lookupSymbol<mlir::ModuleOp>(dmaProfilingResMemModuleName);
    if (dmaProfilingResMemModule == nullptr) {
        return nullptr;
    }

    return dmaProfilingResMemModule.lookupSymbol<IE::MemoryResourceOp>(memSpace);
}

SmallVector<IE::MemoryResourceOp> vpux::IE::getDmaProfilingReservedMemory(mlir::ModuleOp mainModule) {
    auto resMemModule = mainModule.lookupSymbol<mlir::ModuleOp>(resMemModuleName);
    if (resMemModule == nullptr) {
        return {};
    }

    auto dmaProfilingResMemModule = resMemModule.lookupSymbol<mlir::ModuleOp>(dmaProfilingResMemModuleName);
    if (dmaProfilingResMemModule == nullptr) {
        return {};
    }

    return to_small_vector(dmaProfilingResMemModule.getOps<IE::MemoryResourceOp>());
}

//
// Compressed DMA reserved memory
//

IE::MemoryResourceOp vpux::IE::setCompressDmaReservedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace,
                                                            int64_t size) {
    auto* ctx = mainModule->getContext();
    auto byteSizeAttr = getIntAttr(ctx, size);
    auto memSpaceAttr = memSpace.getLeafReference();
    auto mainBuilder = mlir::OpBuilder::atBlockBegin(mainModule.getBody());

    auto resMemModule = mainModule.lookupSymbol<mlir::ModuleOp>(resMemModuleName);
    if (resMemModule == nullptr) {
        resMemModule = mainBuilder.create<mlir::ModuleOp>(mainModule->getLoc(), resMemModuleName);
    }

    auto resMemBuilder = mlir::OpBuilder::atBlockBegin(resMemModule.getBody());

    auto compressDmaResMemModule = resMemModule.lookupSymbol<mlir::ModuleOp>(compressDmaResMemModuleName);
    if (compressDmaResMemModule == nullptr) {
        compressDmaResMemModule =
                resMemBuilder.create<mlir::ModuleOp>(resMemModule->getLoc(), compressDmaResMemModuleName);
    }

    auto res = compressDmaResMemModule.lookupSymbol<IE::MemoryResourceOp>(memSpace.getLeafReference());
    if (res != nullptr) {
        res.byteSizeAttr(byteSizeAttr);
        return res;
    }

    auto innerBuilder = mlir::OpBuilder::atBlockBegin(compressDmaResMemModule.getBody());
    return innerBuilder.create<IE::MemoryResourceOp>(compressDmaResMemModule->getLoc(), memSpaceAttr, byteSizeAttr,
                                                     nullptr);
}

IE::MemoryResourceOp vpux::IE::getCompressDmaReservedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace) {
    auto resMemModule = mainModule.lookupSymbol<mlir::ModuleOp>(resMemModuleName);
    if (resMemModule == nullptr) {
        return nullptr;
    }

    auto compressDmaResMemModule = resMemModule.lookupSymbol<mlir::ModuleOp>(compressDmaResMemModuleName);
    if (compressDmaResMemModule == nullptr) {
        return nullptr;
    }

    return compressDmaResMemModule.lookupSymbol<IE::MemoryResourceOp>(memSpace);
}

SmallVector<IE::MemoryResourceOp> vpux::IE::getCompressDmaReservedMemory(mlir::ModuleOp mainModule) {
    auto resMemModule = mainModule.lookupSymbol<mlir::ModuleOp>(resMemModuleName);
    if (resMemModule == nullptr) {
        return {};
    }

    auto compressDmaResMemModule = resMemModule.lookupSymbol<mlir::ModuleOp>(compressDmaResMemModuleName);
    if (compressDmaResMemModule == nullptr) {
        return {};
    }

    return to_small_vector(compressDmaResMemModule.getOps<IE::MemoryResourceOp>());
}

//
// ExecutorResourceOp
//

IE::ExecutorResourceOp vpux::IE::details::addExecutor(mlir::Region& region, mlir::StringAttr executorAttr,
                                                      uint32_t count) {
    VPUX_THROW_UNLESS(count > 0, "Trying to set zero count of executor kind '{0}'", executorAttr);

    auto* ctx = region.getContext();
    const auto countAttr = getIntAttr(ctx, count);
    auto builder = mlir::OpBuilder::atBlockBegin(&region.front());
    auto resOp = builder.create<IE::ExecutorResourceOp>(region.getLoc(), executorAttr, countAttr, nullptr);

    // Operations with a 'SymbolTable' must have exactly one block
    resOp.getRegion().emplaceBlock();
    return resOp;
}

bool vpux::IE::details::hasExecutor(mlir::SymbolTable mainModule, mlir::StringAttr executorAttr) {
    auto res = mainModule.lookup<IE::ExecutorResourceOp>(executorAttr);
    return res != nullptr;
}

IE::ExecutorResourceOp vpux::IE::getAvailableExecutor(mlir::ModuleOp mainModule, mlir::SymbolRefAttr executorAttr) {
    return mlir::dyn_cast_or_null<IE::ExecutorResourceOp>(mlir::SymbolTable::lookupSymbolIn(mainModule, executorAttr));
}

IE::ExecutorResourceOp vpux::IE::ExecutorResourceOp::addSubExecutor(mlir::StringAttr executorAttr, uint32_t count) {
    return details::addExecutor(getRegion(), executorAttr, count);
}

bool vpux::IE::ExecutorResourceOp::hasSubExecutor(mlir::StringAttr executorAttr) {
    return details::hasExecutor(getOperation(), executorAttr);
}

IE::ExecutorResourceOp vpux::IE::ExecutorResourceOp::getSubExecutor(mlir::StringAttr executorAttr) {
    return lookupSymbol<IE::ExecutorResourceOp>(executorAttr);
}
