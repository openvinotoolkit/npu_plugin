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

using namespace vpux;

//
// PhysicalProcessorAttr
//

StringRef vpux::VPUIP::PhysicalProcessorAttr::getMnemonic() {
    return "PhysicalProcessor";
}

//
// DMAEngineAttr
//

StringRef vpux::VPUIP::DMAEngineAttr::getMnemonic() {
    return "DMAEngine";
}

//
// PhysicalMemoryAttr
//

StringRef vpux::VPUIP::PhysicalMemoryAttr::getMnemonic() {
    return "PhysicalMemory";
}

//
// ArchKindAttr
//

StringRef vpux::VPUIP::ArchKindAttr::getMnemonic() {
    return "ArchKind";
}

//
// MemoryLocationAttr
//

StringRef vpux::VPUIP::MemoryLocationAttr::getMnemonic() {
    return "MemoryLocation";
}

VPUIP::MemoryLocationAttr vpux::VPUIP::MemoryLocationAttr::fromPhysicalMemory(
        mlir::MLIRContext* ctx,
        PhysicalMemory mem) {
    switch (mem) {
    case PhysicalMemory::DDR:
        return get(ctx, MemoryLocation::VPU_DDR_Heap);
    case PhysicalMemory::CSRAM:
        return get(ctx, MemoryLocation::VPU_CSRAM);
    case PhysicalMemory::CMX_UPA:
        return get(ctx, MemoryLocation::VPU_CMX_UPA);
    case PhysicalMemory::CMX_NN:
        return get(ctx, MemoryLocation::VPU_CMX_NN);
    default:
        VPUX_THROW("Unsupported PhysicalMemory : {0}", mem);
    }
}

VPUIP::PhysicalMemory vpux::VPUIP::MemoryLocationAttr::toPhysicalMemory(
        mlir::MemRefType memref) {
    const auto memSpace = memref.getMemorySpace();

    switch (memSpace) {
    case 0:
        return PhysicalMemory::DDR;
    case 1:
        return PhysicalMemory::CSRAM;
    case 2:
        return PhysicalMemory::CMX_UPA;
    case 3:
        return PhysicalMemory::CMX_NN;
    default:
        VPUX_THROW("Unsupported MemRef memory space : {0}", memSpace);
    }
}

VPUIP::PhysicalMemory vpux::VPUIP::MemoryLocationAttr::toPhysicalMemory(
        MemoryLocation location) {
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

VPUIP::PhysicalMemory
        vpux::VPUIP::MemoryLocationAttr::toPhysicalMemory() const {
    return toPhysicalMemory(getValue());
}

VPUIP::MemoryLocationAttr
        vpux::VPUIP::MemoryLocationAttr::fromMemRef(mlir::MemRefType memref) {
    return fromPhysicalMemory(memref.getContext(), toPhysicalMemory(memref));
}

bool vpux::VPUIP::MemoryLocationAttr::isCompatibleWith(
        MemoryLocation location,
        mlir::MemRefType memref) {
    return toPhysicalMemory(location) == toPhysicalMemory(memref);
}

bool vpux::VPUIP::MemoryLocationAttr::isCompatibleWith(
        mlir::MemRefType memref) const {
    return isCompatibleWith(getValue(), memref);
}

//
// ExecutionFlagAttr
//

StringRef vpux::VPUIP::ExecutionFlagAttr::getMnemonic() {
    return "ExecutionFlag";
}

//
// TaskTypeAttr
//

StringRef vpux::VPUIP::TaskTypeAttr::getMnemonic() {
    return "TaskType";
}

//
// Generated
//

#include <vpux/compiler/dialect/VPUIP/generated/attributes/enums.cpp.inc>
