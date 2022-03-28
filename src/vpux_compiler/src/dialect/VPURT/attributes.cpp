//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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

bool vpux::VPURT::isMemoryCompatible(BufferSection section, vpux::NDTypeInterface ndType) {
    return VPURT::getMemoryKind(section) == ndType.getMemoryKind();
}

//
// Generated
//

#include <vpux/compiler/dialect/VPURT/generated/attributes/enums.cpp.inc>
