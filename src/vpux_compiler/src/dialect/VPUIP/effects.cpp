//
// Copyright 2020 Intel Corporation.
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

#include "vpux/compiler/dialect/VPUIP/effects.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"

using namespace vpux;

mlir::SideEffects::Resource* vpux::VPUIP::getMemoryResource(PhysicalMemory mem) {
    switch (mem) {
    case PhysicalMemory::DDR:
        return MemoryResource<PhysicalMemory::DDR>::get();
    case PhysicalMemory::CSRAM:
        return MemoryResource<PhysicalMemory::CSRAM>::get();
    case PhysicalMemory::CMX_UPA:
        return MemoryResource<PhysicalMemory::CMX_UPA>::get();
    case PhysicalMemory::CMX_NN:
        return MemoryResource<PhysicalMemory::CMX_NN>::get();
    default:
        VPUX_THROW("Unsupported PhysicalMemory '{0}' for MemoryResource", mem);
    }
}

mlir::FailureOr<mlir::SideEffects::Resource*> vpux::VPUIP::getMemoryResource(mlir::MemRefType memref) {
    auto mem = getPhysicalMemory(memref);
    if (mlir::failed(mem)) {
        return mlir::failure();
    }

    return getMemoryResource(mem.getValue());
}
