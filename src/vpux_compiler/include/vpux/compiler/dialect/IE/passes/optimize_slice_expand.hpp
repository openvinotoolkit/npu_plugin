//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/passes.hpp"

namespace vpux {
namespace IE {

mlir::LogicalResult genericOptimizeSliceImplicitExpand(IE::ExpandOp layerOp, mlir::Operation* implicitOp,
                                                       mlir::PatternRewriter& rewriter);
//
// OptimizeSliceImplicitExpand
//

template <class ImplicitLayer>
class OptimizeSliceImplicitExpand : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    OptimizeSliceImplicitExpand(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log) {
        setDebugName("OptimizeSliceImplicitExpand");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp origOp, mlir::PatternRewriter& rewriter) const final {
        auto implicitOp = origOp.input().getDefiningOp<ImplicitLayer>();
        if (implicitOp == nullptr) {
            return mlir::failure();
        }
        return genericOptimizeSliceImplicitExpand(origOp, implicitOp.getOperation(), rewriter);
    }

private:
    Logger _log;
};

//
// OptimizeSingleSliceConcatExpand
//

class OptimizeSingleSliceConcatExpand final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    OptimizeSingleSliceConcatExpand(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log) {
        setDebugName("OptimizeSingleSliceConcatExpand");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp layerOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// OptimizeSliceExpand
//

class OptimizeSliceExpand final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    OptimizeSliceExpand(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log) {
        setDebugName("OptimizeSliceExpand");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp layerOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// OptimizeExpandSlice
//

class OptimizeExpandSlice final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    OptimizeExpandSlice(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log) {
        setDebugName("OptimizeExpandSlice");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp layerOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

}  // namespace IE
}  // namespace vpux
