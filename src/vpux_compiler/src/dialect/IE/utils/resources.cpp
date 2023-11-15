//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/Builders.h>

using namespace vpux;

//
// Hierarchy aware utils
//

bool isNceTileMemory(mlir::StringAttr memSpace) {
    auto memSpaceStr = memSpace.getValue();
    return memSpaceStr == stringifyEnum(VPU::MemoryKind::CMX_NN) || memSpaceStr == VPU::CMX_NN_FragmentationAware;
}

bool isNceTileExecutor(mlir::StringAttr executor) {
    auto nceExecutorList = {stringifyEnum(VPU::ExecutorKind::DPU), stringifyEnum(VPU::ExecutorKind::SHAVE_ACT),
                            stringifyEnum(VPU::ExecutorKind::SHAVE_NN)};
    auto executorStr = executor.getValue();
    return std::find(nceExecutorList.begin(), nceExecutorList.end(), executorStr) != nceExecutorList.end();
}

//
// MemoryResourceOp
//

IE::MemoryResourceOp vpux::IE::details::addAvailableMemory(mlir::Region& region, mlir::StringAttr memSpace, Byte size) {
    VPUX_THROW_UNLESS(size.count() > 0, "Trying to set zero size of memory kind '{0}'", memSpace);
    const auto byteSizeAttr = getIntAttr(region.getContext(), size.count());
    auto builder = mlir::OpBuilder::atBlockBegin(&region.front());
    return builder.create<IE::MemoryResourceOp>(region.getLoc(), memSpace, byteSizeAttr, nullptr);
}

IE::MemoryResourceOp vpux::IE::addAvailableMemory(mlir::ModuleOp mainModule, mlir::StringAttr memSpace, Byte size) {
    return details::addAvailableMemory(mainModule.getBodyRegion(), memSpace, size);
}

bool vpux::IE::details::hasAvailableMemory(mlir::SymbolTable symbolTable, mlir::StringAttr memSpace) {
    auto res = symbolTable.lookup<IE::MemoryResourceOp>(memSpace);
    return res != nullptr;
}

bool vpux::IE::hasAvailableMemory(mlir::ModuleOp mainModule, mlir::StringAttr memSpace) {
    if (isNceTileMemory(memSpace)) {
        auto nceTile = IE::getAvailableExecutor(mainModule, VPU::ExecutorKind::NCE);
        VPUX_THROW_UNLESS(nceTile != nullptr, "Expected nceTile executor in order to query '{0}' memspace.", memSpace);
        return nceTile.hasAvailableMemory(memSpace);
    }
    return details::hasAvailableMemory(mainModule.getOperation(), memSpace);
}

IE::MemoryResourceOp vpux::IE::details::getAvailableMemory(mlir::SymbolTable symbolTable, mlir::StringAttr memSpace) {
    return symbolTable.lookup<IE::MemoryResourceOp>(memSpace);
}

IE::MemoryResourceOp vpux::IE::getAvailableMemory(mlir::ModuleOp mainModule, mlir::StringAttr memSpace) {
    if (isNceTileMemory(memSpace)) {
        auto nceTile = IE::getAvailableExecutor(mainModule, VPU::ExecutorKind::NCE);
        VPUX_THROW_UNLESS(nceTile != nullptr, "Expected nceTile executor in order to query '{0}' memspace.", memSpace);
        return nceTile.getAvailableMemory(memSpace);
    }
    return details::getAvailableMemory(mainModule.getOperation(), memSpace);
}

IE::MemoryResourceOp vpux::IE::setUsedMemory(mlir::ModuleOp mainModule, mlir::StringAttr memSpace, Byte size) {
    if (isNceTileMemory(memSpace)) {
        auto nceTile = IE::getAvailableExecutor(mainModule, VPU::ExecutorKind::NCE);
        VPUX_THROW_UNLESS(nceTile != nullptr, "Expected nceTile executor in order to set '{0}' memspace.", memSpace);
        return nceTile.setUsedMemory(memSpace, size);
    }

    auto available = details::getAvailableMemory(mainModule.getOperation(), memSpace);
    VPUX_THROW_UNLESS(available != nullptr, "Memory kind '{0}' is not registered as available", memSpace);
    VPUX_THROW_UNLESS(size <= available.size(), "Memory kind '{0}' used size '{1}' exceeds available size '{2}'",
                      memSpace, size, available.size());

    auto byteSizeAttr = getIntAttr(mainModule->getContext(), size.count());
    auto mainBuilder = mlir::OpBuilder::atBlockBegin(mainModule.getBody());

    auto usedMemModule = mainModule.lookupSymbol<mlir::ModuleOp>(usedMemModuleName);
    if (usedMemModule == nullptr) {
        usedMemModule = mainBuilder.create<mlir::ModuleOp>(mainModule->getLoc(), usedMemModuleName);
    }

    auto res = usedMemModule.lookupSymbol<IE::MemoryResourceOp>(memSpace);
    if (res != nullptr) {
        res.byteSizeAttr(byteSizeAttr);
        return res;
    }

    auto innerBuilder = mlir::OpBuilder::atBlockBegin(usedMemModule.getBody());
    return innerBuilder.create<IE::MemoryResourceOp>(usedMemModule->getLoc(), memSpace, byteSizeAttr, nullptr);
}

IE::MemoryResourceOp vpux::IE::details::getUsedMemory(mlir::SymbolTable symbolTable, mlir::StringAttr memSpace) {
    auto usedMemModule = symbolTable.lookup<mlir::ModuleOp>(usedMemModuleName);
    if (usedMemModule == nullptr) {
        return nullptr;
    }

    return usedMemModule.lookupSymbol<IE::MemoryResourceOp>(memSpace);
}

IE::MemoryResourceOp vpux::IE::getUsedMemory(mlir::ModuleOp mainModule, mlir::StringAttr memSpace) {
    return details::getUsedMemory(mainModule.getOperation(), memSpace);
}

SmallVector<IE::MemoryResourceOp> vpux::IE::getUsedMemory(mlir::ModuleOp mainModule) {
    auto usedMemModule = mainModule.lookupSymbol<mlir::ModuleOp>(usedMemModuleName);
    if (usedMemModule == nullptr) {
        return {};
    }
    auto usedMem = to_small_vector(usedMemModule.getOps<IE::MemoryResourceOp>());
    auto nceTile = IE::getAvailableExecutor(mainModule, VPU::ExecutorKind::NCE);
    if (nceTile == nullptr) {
        return usedMem;
    }
    auto nceUsedMemModule = nceTile.lookupSymbol<mlir::ModuleOp>(usedMemModuleName);
    if (nceUsedMemModule == nullptr) {
        return usedMem;
    }
    auto nceUsedMem = to_small_vector(nceUsedMemModule.getOps<IE::MemoryResourceOp>());
    usedMem.append(nceUsedMem);

    return usedMem;
}

//
// Reserved memory resources
//
SmallVector<IE::MemoryResourceOp> vpux::IE::getReservedMemoryResources(mlir::ModuleOp mainModule,
                                                                       mlir::StringAttr memSpace) {
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

IE::MemoryResourceOp vpux::IE::setDmaProfilingReservedMemory(mlir::ModuleOp mainModule, mlir::StringAttr memSpace,
                                                             int64_t size) {
    auto* ctx = mainModule->getContext();
    auto byteSizeAttr = getIntAttr(ctx, size);
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

    auto res = dmaProfilingResMemModule.lookupSymbol<IE::MemoryResourceOp>(memSpace);
    if (res != nullptr) {
        res.byteSizeAttr(byteSizeAttr);
        return res;
    }

    auto innerBuilder = mlir::OpBuilder::atBlockBegin(dmaProfilingResMemModule.getBody());
    return innerBuilder.create<IE::MemoryResourceOp>(dmaProfilingResMemModule->getLoc(), memSpace, byteSizeAttr,
                                                     nullptr);
}

IE::MemoryResourceOp vpux::IE::getDmaProfilingReservedMemory(mlir::ModuleOp mainModule, mlir::StringAttr memSpace) {
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

IE::MemoryResourceOp vpux::IE::setCompressDmaReservedMemory(mlir::ModuleOp mainModule, mlir::StringAttr memSpace,
                                                            int64_t size) {
    auto* ctx = mainModule->getContext();
    auto byteSizeAttr = getIntAttr(ctx, size);
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

    auto res = compressDmaResMemModule.lookupSymbol<IE::MemoryResourceOp>(memSpace);
    if (res != nullptr) {
        res.byteSizeAttr(byteSizeAttr);
        return res;
    }

    auto innerBuilder = mlir::OpBuilder::atBlockBegin(compressDmaResMemModule.getBody());
    return innerBuilder.create<IE::MemoryResourceOp>(compressDmaResMemModule->getLoc(), memSpace, byteSizeAttr,
                                                     nullptr);
}

IE::MemoryResourceOp vpux::IE::getCompressDmaReservedMemory(mlir::ModuleOp mainModule, mlir::StringAttr memSpace) {
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
    auto executorStr = executorAttr.getLeafReference();
    if (isNceTileExecutor(executorStr)) {
        auto nceTile = IE::getAvailableExecutor(mainModule, VPU::ExecutorKind::NCE);
        VPUX_THROW_UNLESS(nceTile != nullptr, "Expected nceTile executor in order to query '{0}' executor.",
                          executorAttr);
        return nceTile.getSubExecutor(executorStr);
    }
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

IE::MemoryResourceOp vpux::IE::ExecutorResourceOp::addAvailableMemory(mlir::StringAttr memSpace, Byte size) {
    return details::addAvailableMemory(getRegion(), memSpace, size);
}

bool vpux::IE::ExecutorResourceOp::hasAvailableMemory(mlir::StringAttr memSpace) {
    return details::hasAvailableMemory(getOperation(), memSpace);
}

IE::MemoryResourceOp vpux::IE::ExecutorResourceOp::getAvailableMemory(mlir::StringAttr memSpace) {
    return lookupSymbol<IE::MemoryResourceOp>(memSpace);
}

IE::MemoryResourceOp vpux::IE::ExecutorResourceOp::setUsedMemory(mlir::StringAttr memSpace, Byte size) {
    auto available = getAvailableMemory(memSpace);
    VPUX_THROW_UNLESS(available != nullptr, "Memory kind '{0}' is not registered as available", memSpace);
    VPUX_THROW_UNLESS(size <= available.size(), "Memory kind '{0}' used size '{1}' exceeds available size '{2}'",
                      memSpace, size, available.size());

    auto byteSizeAttr = getIntAttr(getContext(), size.count());
    auto mainBuilder = mlir::OpBuilder::atBlockBegin(&getRegion().front());

    auto usedMemModule = lookupSymbol<mlir::ModuleOp>(usedMemModuleName);
    if (usedMemModule == nullptr) {
        usedMemModule = mainBuilder.create<mlir::ModuleOp>(getLoc(), usedMemModuleName);
    }

    auto res = usedMemModule.lookupSymbol<IE::MemoryResourceOp>(memSpace);
    if (res != nullptr) {
        res.byteSizeAttr(byteSizeAttr);
        return res;
    }

    auto innerBuilder = mlir::OpBuilder::atBlockBegin(usedMemModule.getBody());
    return innerBuilder.create<IE::MemoryResourceOp>(usedMemModule->getLoc(), memSpace, byteSizeAttr, nullptr);
}

IE::MemoryResourceOp vpux::IE::ExecutorResourceOp::getUsedMemory(mlir::StringAttr memSpace) {
    return details::getUsedMemory(getOperation(), memSpace);
}

//
// ShaveResources
//

int64_t vpux::IE::getTotalNumOfActShaveEngines(mlir::ModuleOp moduleOp) {
    auto nceTileResOp = IE::getAvailableExecutor(moduleOp, VPU::ExecutorKind::NCE);
    VPUX_THROW_UNLESS(nceTileResOp != nullptr, "Expected NCE executor in order to query SHAVE_ACT executor.");
    VPUX_THROW_UNLESS(nceTileResOp.hasSubExecutor(VPU::ExecutorKind::SHAVE_ACT),
                      "No SHAVE_ACT executor found, check your arch");
    auto actShavePerTileResOp = nceTileResOp.getSubExecutor(VPU::ExecutorKind::SHAVE_ACT);
    return nceTileResOp.count() * actShavePerTileResOp.count();
}
