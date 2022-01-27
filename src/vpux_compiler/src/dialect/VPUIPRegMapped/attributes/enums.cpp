//
// Copyright Intel Corporation.
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

#include "vpux/compiler/dialect/VPUIPRegMapped/attributes/enums.hpp"

#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/StringExtras.h>
#include "llvm/Support/Debug.h"                                   // 2022_01_25
#include "vpux/compiler/core/attributes/indexed_symbol_attr.hpp"  // 2022_01_25

using namespace vpux;

namespace {

constexpr StringLiteral compilationModeAttrName = "VPUIPRegMapped.compilationMode";

}  // namespace

//
// MemoryLocation utilities
//

VPUIPRegMapped::PhysicalMemory vpux::VPUIPRegMapped::getPhysicalMemory(MemoryLocation location) {
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
    case MemoryLocation::MAC_Accumulators:
        return PhysicalMemory::Register;
    default:
        VPUX_THROW("Unsupported MemoryLocation : {0}", location);
    }
}

VPUIPRegMapped::MemoryLocation vpux::VPUIPRegMapped::getDefaultMemoryLocation(VPUIPRegMapped::PhysicalMemory memory) {
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

mlir::FailureOr<VPUIPRegMapped::PhysicalMemory> vpux::VPUIPRegMapped::getPhysicalMemory(mlir::MemRefType memref) {
    const auto memSpace = memref.getMemorySpace();  // Note: returned type is mlir::Attribute
    // const mlir::Attribute memSpace2 = memref.getMemorySpace();

    //#define DEBUG_ALEX

#ifdef DEBUG_ALEX
    llvm::dbgs() << "vpux::VPUIPRegMapped::getPhysicalMemory(): memSpace = " << memSpace << "\n";  // 2022_01_25
    // Inspired from https://stackoverflow.com/questions/3649278/how-can-i-get-the-class-name-from-a-c-object
    llvm::dbgs() << "typeid(memSpace).name() = " << typeid(memSpace).name()
                 << "\n";  // 2022_01_25
                           // It writes: "N4mlir9AttributeE" (mangled string for mlir::Attribute).
    llvm::dbgs().flush();  // 2022_01_25
#endif

    if (memSpace == nullptr) {
#ifdef DEBUG_ALEX
        llvm::dbgs() << "vpux::VPUIPRegMapped::getPhysicalMemory(): return PhysicalMemory::DDR"
                     << "\n";  // 2022_01_25
        llvm::dbgs().flush();  // 2022_01_25
#endif

        return PhysicalMemory::DDR;
    }

    if (memSpace.isa<VPUIPRegMapped::PhysicalMemoryAttr>()) {
#ifdef DEBUG_ALEX
        llvm::dbgs() << "vpux::VPUIPRegMapped::getPhysicalMemory(): return ...getValue()"
                     << "\n";  // 2022_01_25
        llvm::dbgs().flush();  // 2022_01_25
#endif

        return memSpace.cast<VPUIPRegMapped::PhysicalMemoryAttr>().getValue();
    }

    /*
    // memSpace is normally a mlir::SymbolRefAttr.
    if (memSpace.isa<mlir::SymbolRefAttr>()) {
#ifdef DEBUG_ALEX
        llvm::dbgs() << "vpux::VPUIPRegMapped::getPhysicalMemory(): memSpace.isa<mlir::SymbolRefAttr>()"
                     << "\n";  // 2022_01_25
        llvm::dbgs().flush();  // 2022_01_25
#endif
    }
    */

    if (memSpace.isa<vpux::IndexedSymbolAttr>()) {
#ifdef DEBUG_ALEX
        llvm::dbgs() << "vpux::VPUIPRegMapped::getPhysicalMemory(): memSpace.isa<vpux::IndexedSymbolAttr>()"
                     << "\n";  // 2022_01_25

        llvm::dbgs()
                << "vpux::VPUIPRegMapped::getPhysicalMemory(): memSpace.cast<vpux::IndexedSymbolAttr>().getName() = "
                << memSpace.cast<vpux::IndexedSymbolAttr>().getName() << "\n";  // 2022_01_25
#endif

        if (memSpace.cast<vpux::IndexedSymbolAttr>().getName() == "DDR") {
#ifdef DEBUG_ALEX
            llvm::dbgs() << "vpux::VPUIPRegMapped::getPhysicalMemory(): "
                            "memSpace.cast<vpux::IndexedSymbolAttr>().getName() is DDR"
                         << "\n";  // 2022_01_25
#endif

            return PhysicalMemory::DDR;  // 2022_01_25
        }
        // TODO: treat the other cases besides DDR

#ifdef DEBUG_ALEX
        llvm::dbgs().flush();  // 2022_01_25
#endif
    }

    /*
    if (!memSpace.isa<VPUIPRegMapped::MemoryLocationAttr>()) {
#ifdef DEBUG_ALEX
        llvm::dbgs() << "vpux::VPUIPRegMapped::getPhysicalMemory(): return mlir::failure()"
                     << "\n";  // 2022_01_25
        llvm::dbgs().flush();  // 2022_01_25
#endif

        //return PhysicalMemory::DDR;  // 2022_01_25
        return mlir::failure();
    }

#ifdef DEBUG_ALEX
    llvm::dbgs() << "vpux::VPUIPRegMapped::getPhysicalMemory(): return getPhysicalMemory(...)"
                 << "\n";  // 2022_01_25
    llvm::dbgs().flush();  // 2022_01_25
#endif

    return getPhysicalMemory(memSpace.cast<VPUIPRegMapped::MemoryLocationAttr>().getValue());
    */

    return mlir::failure();
}

mlir::FailureOr<VPUIPRegMapped::MemoryLocation> vpux::VPUIPRegMapped::getMemoryLocation(mlir::MemRefType memref) {
    const auto memSpace = memref.getMemorySpace();

    if (memSpace == nullptr) {
        return VPUIPRegMapped::MemoryLocation::VPU_DDR_Heap;
    }

    if (memSpace.isa<VPUIPRegMapped::MemoryLocationAttr>()) {
        return memSpace.cast<VPUIPRegMapped::MemoryLocationAttr>().getValue();
    }

    if (!memSpace.isa<VPUIPRegMapped::PhysicalMemoryAttr>()) {
        return mlir::failure();
    }

    return getDefaultMemoryLocation(memSpace.cast<VPUIPRegMapped::PhysicalMemoryAttr>().getValue());
}

bool vpux::VPUIPRegMapped::isMemoryCompatible(MemoryLocation location, mlir::MemRefType memref) {
    /*
    llvm::dbgs() << "vpux::VPUIPRegMapped::isMemoryCompatible(): getPhysicalMemory(location) = "
                 << getPhysicalMemory(location) << "\n";  // 2022_01_25
    llvm::dbgs() << "vpux::VPUIPRegMapped::isMemoryCompatible(): getPhysicalMemory(memref) = "
                 << getPhysicalMemory(memref) << "\n";  // 2022_01_25
    */
    llvm::dbgs().flush();  // 2022_01_25

    return getPhysicalMemory(location) == getPhysicalMemory(memref);
}

void vpux::VPUIPRegMapped::setCompilationMode(mlir::ModuleOp module, CompilationMode compilationMode) {
    module->setAttr(compilationModeAttrName,
                    VPUIPRegMapped::CompilationModeAttr::get(module.getContext(), compilationMode));
}

VPUIPRegMapped::CompilationMode vpux::VPUIPRegMapped::getCompilationMode(mlir::Operation* op) {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    VPUX_THROW_UNLESS(module != nullptr, "Can't get parent Module from Operation '{0}' at '{1}'", op->getName(),
                      op->getLoc());

    auto attr = module->getAttr(compilationModeAttrName);
    VPUX_THROW_UNLESS(attr != nullptr, "Module doesn't contain '{0}' attribute", compilationModeAttrName);
    VPUX_THROW_UNLESS(attr.isa<VPUIPRegMapped::CompilationModeAttr>(),
                      "Module attribute '{0}' has unsupported value '{1}'", compilationModeAttrName, attr);

    return attr.cast<VPUIPRegMapped::CompilationModeAttr>().getValue();
}

//
// Generated
//

#include <vpux/compiler/dialect/VPUIPRegMapped/generated/attributes/enums.cpp.inc>
