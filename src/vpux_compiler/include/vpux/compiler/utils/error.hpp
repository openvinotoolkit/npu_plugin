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
