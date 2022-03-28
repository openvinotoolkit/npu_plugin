//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/core/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPURT/attributes.hpp"
#include "vpux/compiler/dialect/VPURT/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPURT/types.hpp"

#include <mlir/IR/BuiltinTypes.h>

// Generated
//

#include <vpux/compiler/dialect/VPURT/generated/dialect.hpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPURT/generated/ops.hpp.inc>

//
// Operation verifiers
//

namespace vpux {
namespace VPURT {

mlir::LogicalResult verifyOp(DeclareBufferOp op);

}  // namespace VPURT
}  // namespace vpux
