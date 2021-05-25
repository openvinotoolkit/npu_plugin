//
// Copyright 2020 Intel Corporation.
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

#include "vpux/compiler/core/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/attributes/enums.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/small_vector.hpp"

namespace vpux {
namespace IE {

mlir::FailureOr<SmallVector<int64_t>> broadcastEltwiseShape(ArrayRef<int64_t> shape1, ArrayRef<int64_t> shape2,
                                                            AutoBroadcastType broadcastType, mlir::Location loc);

mlir::FailureOr<SmallVector<int64_t>> broadcastEltwiseShape(ArrayRef<ArrayRef<int64_t>> shapes,
                                                            AutoBroadcastType broadcastType, mlir::Location loc);
}  // namespace IE
}  // namespace vpux
