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
