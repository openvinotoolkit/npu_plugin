//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include <mlir/IR/Types.h>

namespace vpux {
namespace VPU {
bool isNCEEltwiseSupported(VPU::ArchKind arch, vpux::NDTypeInterface input1Type, vpux::NDTypeInterface input2Type,
                           vpux::NDTypeInterface outputType, bool allowDifferentScales, bool allowDifferentZp,
                           bool checkLayout, bool checkChannelAlignment, LogCb logCb);
}
}  // namespace vpux
