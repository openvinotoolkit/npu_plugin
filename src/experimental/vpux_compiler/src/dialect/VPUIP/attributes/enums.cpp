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

#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"

#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/StringExtras.h>

using namespace vpux;

//
// MemoryLocation utilities
//

VPUIP::MemoryLocation vpux::VPUIP::getDefaultMemoryLocation(PhysicalMemory mem) {
    switch (mem) {
    case PhysicalMemory::DDR:
        return MemoryLocation::VPU_DDR_Heap;
    case PhysicalMemory::CSRAM:
        return MemoryLocation::VPU_CSRAM;
    case PhysicalMemory::CMX_UPA:
        return MemoryLocation::VPU_CMX_UPA;
    case PhysicalMemory::CMX_NN:
        return MemoryLocation::VPU_CMX_NN;
    default:
        VPUX_THROW("Unsupported PhysicalMemory : {0}", mem);
    }
}

VPUIP::MemoryLocation vpux::VPUIP::getDefaultMemoryLocation(mlir::MemRefType memref) {
    return getDefaultMemoryLocation(getPhysicalMemory(memref));
}

VPUIP::PhysicalMemory vpux::VPUIP::getPhysicalMemory(MemoryLocation location) {
    switch (location) {
    case MemoryLocation::ProgrammableInput:
    case MemoryLocation::ProgrammableOutput:
    case MemoryLocation::GraphFile:
    case MemoryLocation::VPU_DDR_Heap:
    case MemoryLocation::VPU_DDR_BSS:
        return PhysicalMemory::DDR;
    case MemoryLocation::VPU_CSRAM:
        return PhysicalMemory::CSRAM;
    case MemoryLocation::VPU_CMX_UPA:
        return PhysicalMemory::CMX_UPA;
    case MemoryLocation::VPU_CMX_NN:
        return PhysicalMemory::CMX_NN;
    default:
        VPUX_THROW("Unsupported MemoryLocation : {0}", location);
    }
}

VPUIP::PhysicalMemory vpux::VPUIP::getPhysicalMemory(mlir::MemRefType memref) {
    const auto memSpace = memref.getMemorySpace();

    if (memSpace == nullptr) {
        return PhysicalMemory::DDR;
    }

    if (memSpace.isa<VPUIP::PhysicalMemoryAttr>()) {
        return memSpace.cast<VPUIP::PhysicalMemoryAttr>().getValue();
    }

    VPUX_THROW_UNLESS(memSpace.isa<VPUIP::MemoryLocationAttr>(), "Unsupported memory space {0}", memSpace);

    return getPhysicalMemory(memSpace.cast<VPUIP::MemoryLocationAttr>().getValue());
}

bool vpux::VPUIP::isMemoryCompatible(MemoryLocation location, mlir::MemRefType memref) {
    return getPhysicalMemory(location) == getPhysicalMemory(memref);
}

//
// Generated
//

#include <vpux/compiler/dialect/VPUIP/generated/attributes/enums.cpp.inc>
