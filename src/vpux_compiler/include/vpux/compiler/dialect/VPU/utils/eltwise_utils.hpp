//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include <mlir/IR/Types.h>

namespace vpux {
namespace VPU {
bool isNCEEltwiseSupported(VPU::ArchKind arch, mlir::ValueRange operands, mlir::Value result, bool allowDifferentScales,
                           bool allowDifferentZp, LogCb logCb);
}
}  // namespace vpux
