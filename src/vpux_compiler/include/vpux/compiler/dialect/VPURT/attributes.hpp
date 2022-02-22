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

#pragma once

#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"

#include <mlir/IR/BuiltinAttributes.h>

//
// Generated
//

#include <vpux/compiler/dialect/VPURT/generated/attributes/enums.hpp.inc>

//
// BufferSection/MemoryKind conversion
//

namespace vpux {
namespace VPURT {

VPU::MemoryKind getMemoryKind(BufferSection section);
BufferSection getBufferSection(VPU::MemoryKind memKind);

bool isMemoryCompatible(BufferSection section, vpux::NDTypeInterface ndType);

}  // namespace VPURT
}  // namespace vpux
