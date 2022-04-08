//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/core/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/attributes/enums.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/small_vector.hpp"

namespace vpux {
namespace IE {

mlir::FailureOr<SmallVector<int64_t>> broadcastEltwiseShape(ArrayRef<int64_t> shape1, ArrayRef<int64_t> shape2,
                                                            AutoBroadcastType broadcastType, mlir::Location loc);

mlir::FailureOr<SmallVector<int64_t>> broadcastEltwiseShape(ArrayRef<ArrayRef<int64_t>> shapes,
                                                            AutoBroadcastType broadcastType, mlir::Location loc);

mlir::FailureOr<SmallVector<int64_t>> constInputToData(mlir::Location loc, const mlir::Value& value);

}  // namespace IE
}  // namespace vpux
