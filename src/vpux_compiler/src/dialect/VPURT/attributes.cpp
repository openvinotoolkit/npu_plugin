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

#include "vpux/compiler/dialect/VPURT/attributes.hpp"

#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/StringExtras.h>

using namespace vpux;

//
// BufferSection/MemoryKind conversion
//

VPU::MemoryKind vpux::VPURT::getMemoryKind(BufferSection section) {
    switch (section) {
    case BufferSection::NetworkInput:
    case BufferSection::NetworkOutput:
    case BufferSection::ProfilingOutput:
    case BufferSection::Constant:
    case BufferSection::DDR:
        return VPU::MemoryKind::DDR;
    case BufferSection::CSRAM:
        return VPU::MemoryKind::CSRAM;
    case BufferSection::CMX_UPA:
        return VPU::MemoryKind::CMX_UPA;
    case BufferSection::CMX_NN:
        return VPU::MemoryKind::CMX_NN;
    case BufferSection::Register:
    case BufferSection::MAC_Accumulators:
        return VPU::MemoryKind::Register;
    default:
        VPUX_THROW("Unsupported BufferSection : {0}", section);
    }
}

VPURT::BufferSection vpux::VPURT::getBufferSection(VPU::MemoryKind memKind) {
    switch (memKind) {
    case VPU::MemoryKind::DDR:
        return BufferSection::DDR;
    case VPU::MemoryKind::CSRAM:
        return BufferSection::CSRAM;
    case VPU::MemoryKind::CMX_UPA:
        return BufferSection::CMX_UPA;
    case VPU::MemoryKind::CMX_NN:
        return BufferSection::CMX_NN;
    case VPU::MemoryKind::Register:
        return BufferSection::Register;
    default:
        VPUX_THROW("Unsupported MemoryKind : {0}", memKind);
    }
}

bool vpux::VPURT::isMemoryCompatible(BufferSection section, mlir::MemRefType memref) {
    const auto ndType = memref.cast<vpux::NDTypeInterface>();
    return VPURT::getMemoryKind(section) == ndType.getMemoryKind();
}

//
// Generated
//

#include <vpux/compiler/dialect/VPURT/generated/attributes/enums.cpp.inc>
