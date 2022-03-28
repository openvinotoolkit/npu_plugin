//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/logger.hpp"

#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
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

template <typename... Args>
mlir::LogicalResult matchFailed(mlir::RewriterBase& rewriter, mlir::Operation* op, StringRef format, Args&&... args) {
    const auto msg = llvm::formatv(format.data(), std::forward<Args>(args)...);
    return rewriter.notifyMatchFailure(op, msg.str());
}

template <typename... Args>
mlir::LogicalResult matchFailed(Logger log, mlir::RewriterBase& rewriter, mlir::Operation* op, StringRef format,
                                Args&&... args) {
    log.trace(format, std::forward<Args>(args)...);
    return matchFailed(rewriter, op, format, std::forward<Args>(args)...);
}

}  // namespace vpux
