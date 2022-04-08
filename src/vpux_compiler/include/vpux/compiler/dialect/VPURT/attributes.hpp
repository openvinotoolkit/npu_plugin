//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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
