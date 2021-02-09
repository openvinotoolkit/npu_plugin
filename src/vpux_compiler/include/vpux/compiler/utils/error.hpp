//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

#include "vpux/utils/core/format.hpp"

#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>

namespace vpux {

template <typename... Args>
mlir::LogicalResult errorAt(mlir::Location loc, StringRef format, Args&&... args) {
    return printTo(mlir::emitError(loc), format.data(), std::forward<Args>(args)...);
}

template <typename... Args>
mlir::LogicalResult errorAt(mlir::Operation* op, StringRef format, Args&&... args) {
    return printTo(op->emitError(), format.data(), std::forward<Args>(args)...);
}

}  // namespace vpux
