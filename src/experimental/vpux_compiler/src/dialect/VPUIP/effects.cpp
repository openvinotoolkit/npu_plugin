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

mlir::SideEffects::Resource*
        vpux::VPUIP::getMemoryResource(MemoryLocation location) {
    switch (location) {
    case MemoryLocation::VPU_DDR_Heap:
        return MemoryResource<MemoryLocation::VPU_DDR_Heap>::get();
    case MemoryLocation::VPU_CMX_NN:
        return MemoryResource<MemoryLocation::VPU_CMX_NN>::get();
    case MemoryLocation::VPU_CMX_UPA:
        return MemoryResource<MemoryLocation::VPU_CMX_UPA>::get();
    case MemoryLocation::VPU_DDR_BSS:
        return MemoryResource<MemoryLocation::VPU_DDR_BSS>::get();
    case MemoryLocation::VPU_CSRAM:
        return MemoryResource<MemoryLocation::VPU_CSRAM>::get();
    default:
        VPUX_THROW("Unsupported MemoryLocation '{0}' for MemoryResource",
                   location);
    }
}

mlir::SideEffects::Resource*
        vpux::VPUIP::getMemoryResource(mlir::MemRefType memref) {
    return getMemoryResource(MemoryLocationAttr::fromMemRef(memref));
}
