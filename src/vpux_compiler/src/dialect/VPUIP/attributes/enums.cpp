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
    default:
        VPUX_THROW("Unsupported MemoryLocation : {0}", location);
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
