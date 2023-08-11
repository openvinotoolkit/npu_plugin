//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/ops.hpp"

namespace vpux {
namespace IE {

//
// MemoryResourceOp
//

static constexpr StringLiteral usedMemModuleName = "UsedMemory";

MemoryResourceOp addAvailableMemory(mlir::ModuleOp mainModule, mlir::StringAttr memSpace, Byte size);
bool hasAvailableMemory(mlir::ModuleOp mainModule, mlir::StringAttr memSpace);

template <typename Enum, typename OutT = MemoryResourceOp>
using memory_resource_if = enable_t<OutT, std::is_enum<Enum>, details::HasStringifyEnum<Enum>>;

template <typename Enum>
memory_resource_if<Enum> addAvailableMemory(mlir::ModuleOp mainModule, Enum kind, Byte size) {
    return addAvailableMemory(mainModule, mlir::StringAttr::get(mainModule.getContext(), stringifyEnum(kind)), size);
}

template <typename Enum, typename OutT = bool>
using bool_if = enable_t<OutT, std::is_enum<Enum>, details::HasStringifyEnum<Enum>>;

template <typename Enum>
bool_if<Enum> hasAvailableMemory(mlir::ModuleOp mainModule, Enum kind) {
    return hasAvailableMemory(mainModule, mlir::StringAttr::get(mainModule.getContext(), stringifyEnum(kind)));
}

MemoryResourceOp getAvailableMemory(mlir::ModuleOp mainModule, mlir::StringAttr memSpace);
MemoryResourceOp getAvailableMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace);

template <typename Enum>
memory_resource_if<Enum> getAvailableMemory(mlir::ModuleOp mainModule, Enum kind) {
    return mainModule.template lookupSymbol<MemoryResourceOp>(stringifyEnum(kind));
}

MemoryResourceOp setUsedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace, Byte size);

template <typename Enum>
memory_resource_if<Enum> setUsedMemory(mlir::ModuleOp mainModule, Enum kind, Byte size) {
    return setUsedMemory(mainModule, mlir::SymbolRefAttr::get(mainModule.getContext(), stringifyEnum(kind)), size);
}

MemoryResourceOp getUsedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace);

template <typename Enum>
memory_resource_if<Enum> getUsedMemory(mlir::ModuleOp mainModule, Enum kind) {
    return getUsedMemory(mainModule, mlir::SymbolRefAttr::get(mainModule.getContext(), stringifyEnum(kind)));
}

SmallVector<MemoryResourceOp> getUsedMemory(mlir::ModuleOp mainModule);

//
// Reserved memory resource
//
static constexpr StringLiteral resMemModuleName = "ReservedMemory";

SmallVector<IE::MemoryResourceOp> getReservedMemoryResources(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace);

//
// DMA profiling reserved memory
//
static constexpr StringLiteral dmaProfilingResMemModuleName = "DmaProfilingReservedMemory";

IE::MemoryResourceOp setDmaProfilingReservedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace,
                                                   int64_t size);

IE::MemoryResourceOp getDmaProfilingReservedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace);

template <typename Enum>
memory_resource_if<Enum> getDmaProfilingReservedMemory(mlir::ModuleOp mainModule, Enum kind) {
    return getDmaProfilingReservedMemory(mainModule,
                                         mlir::SymbolRefAttr::get(mainModule.getContext(), stringifyEnum(kind)));
}

SmallVector<MemoryResourceOp> getDmaProfilingReservedMemory(mlir::ModuleOp mainModule);

//
// Compressed DMA reserved memory
//
static constexpr StringLiteral compressDmaResMemModuleName = "CompressDmaReservedMemory";

IE::MemoryResourceOp setCompressDmaReservedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace,
                                                  int64_t size);

IE::MemoryResourceOp getCompressDmaReservedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace);

template <typename Enum>
memory_resource_if<Enum> getCompressDmaReservedMemory(mlir::ModuleOp mainModule, Enum kind) {
    return getCompressDmaReservedMemory(mainModule,
                                        mlir::SymbolRefAttr::get(mainModule.getContext(), stringifyEnum(kind)));
}

SmallVector<MemoryResourceOp> getCompressDmaReservedMemory(mlir::ModuleOp mainModule);

//
// ExecutorResourceOp
//

namespace details {

ExecutorResourceOp addExecutor(mlir::Region& region, mlir::StringAttr executorAttr, uint32_t count);

bool hasExecutor(mlir::SymbolTable mainModule, mlir::StringAttr executorAttr);

}  // namespace details

template <typename Enum, typename OutT = ExecutorResourceOp>
using exec_resource_if = enable_t<OutT, std::is_enum<Enum>, vpux::details::HasStringifyEnum<Enum>>;

template <typename Enum>
exec_resource_if<Enum> addAvailableExecutor(mlir::ModuleOp mainModule, Enum kind, uint32_t count) {
    const auto executorAttr = mlir::StringAttr::get(mainModule->getContext(), stringifyEnum(kind));
    return details::addExecutor(mainModule.getBodyRegion(), executorAttr, count);
}

template <typename Enum>
bool_if<Enum> hasExecutor(mlir::ModuleOp mainModule, Enum kind) {
    const auto executorAttr = mlir::StringAttr::get(mainModule->getContext(), stringifyEnum(kind));
    return details::hasExecutor(mainModule.getOperation(), executorAttr);
}

ExecutorResourceOp getAvailableExecutor(mlir::ModuleOp mainModule, mlir::SymbolRefAttr executorAttr);

template <typename Enum>
exec_resource_if<Enum> getAvailableExecutor(mlir::ModuleOp mainModule, Enum kind) {
    return getAvailableExecutor(mainModule, mlir::SymbolRefAttr::get(mainModule->getContext(), stringifyEnum(kind)));
}

}  // namespace IE
}  // namespace vpux
