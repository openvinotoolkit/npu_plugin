//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinTypes.h>

namespace vpux {
namespace IE {

llvm::Optional<int64_t> getFQAxisIndex(IE::FakeQuantizeOp fq);
llvm::Optional<int64_t> getQuantAxisIndex(mlir::Operation* fq);

}  // namespace IE
}  // namespace vpux