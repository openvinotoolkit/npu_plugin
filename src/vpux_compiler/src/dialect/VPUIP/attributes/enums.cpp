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

#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"

#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/StringExtras.h>

using namespace vpux;

namespace {
constexpr StringLiteral compilationModeAttrName = "VPUIP.compilationMode";
};

//
// MemoryLocation utilities
//

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
    case MemoryLocation::AbsoluteAddr:
        return PhysicalMemory::Register;
    default:
        VPUX_THROW("Unsupported MemoryLocation : {0}", location);
    }
}

VPUIP::MemoryLocation vpux::VPUIP::getDefaultMemoryLocation(VPUIP::PhysicalMemory memory) {
    switch (memory) {
    case PhysicalMemory::DDR:
        return MemoryLocation::VPU_DDR_Heap;
    case PhysicalMemory::CSRAM:
        return MemoryLocation::VPU_CSRAM;
    case PhysicalMemory::CMX_UPA:
        return MemoryLocation::VPU_CMX_UPA;
    case PhysicalMemory::CMX_NN:
        return MemoryLocation::VPU_CMX_NN;
    case PhysicalMemory::Register:
        return MemoryLocation::AbsoluteAddr;
    default:
        VPUX_THROW("Unsupported PhysicalMemory : {0}", memory);
    }
}

mlir::FailureOr<VPUIP::PhysicalMemory> vpux::VPUIP::getPhysicalMemory(mlir::MemRefType memref) {
    const auto memSpace = memref.getMemorySpace();

    if (memSpace == nullptr) {
        return PhysicalMemory::DDR;
    }

    if (memSpace.isa<VPUIP::PhysicalMemoryAttr>()) {
        return memSpace.cast<VPUIP::PhysicalMemoryAttr>().getValue();
    }

    if (!memSpace.isa<VPUIP::MemoryLocationAttr>()) {
        return mlir::failure();
    }

    return getPhysicalMemory(memSpace.cast<VPUIP::MemoryLocationAttr>().getValue());
}

mlir::FailureOr<VPUIP::MemoryLocation> vpux::VPUIP::getMemoryLocation(mlir::MemRefType memref) {
    const auto memSpace = memref.getMemorySpace();

    if (memSpace == nullptr) {
        return VPUIP::MemoryLocation::VPU_DDR_Heap;
    }

    if (memSpace.isa<VPUIP::MemoryLocationAttr>()) {
        return memSpace.cast<VPUIP::MemoryLocationAttr>().getValue();
    }

    if (!memSpace.isa<VPUIP::PhysicalMemoryAttr>()) {
        return mlir::failure();
    }

    return getDefaultMemoryLocation(memSpace.cast<VPUIP::PhysicalMemoryAttr>().getValue());
}

bool vpux::VPUIP::isMemoryCompatible(MemoryLocation location, mlir::MemRefType memref) {
    return getPhysicalMemory(location) == getPhysicalMemory(memref);
}

void vpux::VPUIP::setCompilationMode(mlir::ModuleOp module, CompilationMode compilationMode) {
    module->setAttr(compilationModeAttrName, VPUIP::CompilationModeAttr::get(module.getContext(), compilationMode));
}

VPUIP::CompilationMode vpux::VPUIP::getCompilationMode(mlir::ModuleOp module) {
    auto attr = module->getAttr(compilationModeAttrName);
    VPUX_THROW_UNLESS(attr != nullptr, "Module doesn't contain '{0}' attribute", compilationModeAttrName);
    VPUX_THROW_UNLESS(attr.isa<VPUIP::CompilationModeAttr>(), "Module attribute '{0}' has unsupported value '{1}'",
                      compilationModeAttrName, attr);
    return attr.cast<VPUIP::CompilationModeAttr>().getValue();
}

//
// Generated
//

#include <vpux/compiler/dialect/VPUIP/generated/attributes/enums.cpp.inc>
