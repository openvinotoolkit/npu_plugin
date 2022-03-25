//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/TypeSwitch.h>
#include <vpux/compiler/conversion.hpp>

using namespace vpux;

namespace {

//
// TransposeToPermuteCast
//

class TransposeToPermuteCast final : public IE::TransposeToPermuteCastBase<TransposeToPermuteCast> {
public:
    explicit TransposeToPermuteCast(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    class TransposeOpConverter;

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

//
// TransposeOpConverter
//

class TransposeToPermuteCast::TransposeOpConverter final : public mlir::OpRewritePattern<IE::TransposeOp> {
public:
    TransposeOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::TransposeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TransposeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Consider the following tensor transposition:
// 1x16x32x64 -> 1x64x16x32:
// 1    16  32  64  ->  1   64  16  32
// d0   d1  d2  d3  ->  d0  d3  d1  d2
// This transposition is described as:
// (d0, d1, d2, d3) -> (d0, d3, d1, d2)
// To find out the order of target tensor, one must inverse applied affine map:
// d0, d3, d1, d2   ->  d0, d1, d2, d3
// aN, aC, aH, aW   ->  aN, aH, aW, aC
// Thus, target order of dimensions is actually NHWC.
DimsOrder deduceTargetOrder(IE::TransposeOp op) {
    const auto orderAttr = op.order_valueAttr();
    const auto order = DimsOrder::fromAffineMap(orderAttr.getValue());

    // Given the example above, inputOrder is NCHW.
    const auto inputOrder = vpux::DimsOrder::fromValue(op.input());
    // The order is (d0, d1, d2, d3) -> (d0, d3, d1, d2), NWCH
    auto targetPermutation = order.toPermutation();
    // The following loop in case of NCHW just goes over d0(N), d1(C), d2(H), d3(W).
    // N is trivial enough (it is d0 in both NCHW and NWCH).
    // Now C resides at index 1 in NCHW and index 2 in NWCH, so targetPermutation[1] = d2
    // Applying this for H gives: index 2 in NCHW and index 3 in NWCH => targetPermutation[2] = d3
    // Finally, for W: index 3 in NCHW and index 1 in NWCH => targetPermutation[2] = d1
    // The result is: d0, d2, d3, d1
    for (const auto& perm : inputOrder.toPermutation()) {
        const auto permInd = perm.ind();
        targetPermutation[permInd] = Dim(order.dimPos(Dim(permInd)));
    }

    return DimsOrder::fromPermutation(targetPermutation);
}

mlir::LogicalResult TransposeToPermuteCast::TransposeOpConverter::matchAndRewrite(
        IE::TransposeOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto transposeIn = origOp.input();
    const auto targetDimsOrder = deduceTargetOrder(origOp);
    const auto dstOrder = mlir::AffineMapAttr::get(targetDimsOrder.toAffineMap(rewriter.getContext()));
    const auto origOutOrder = DimsOrder::fromValue(origOp.output());
    const auto numDims = checked_cast<unsigned>(origOutOrder.numDims());
    const auto memPerm =
            mlir::AffineMapAttr::get(mlir::AffineMap::getMinorIdentityMap(numDims, numDims, rewriter.getContext()));
    auto permuteCast = rewriter.create<IE::PermuteCastOp>(origOp->getLoc(), transposeIn, dstOrder, memPerm);
    rewriter.replaceOpWithNewOp<IE::ReorderOp>(
            origOp, permuteCast.output(), mlir::AffineMapAttr::get(origOutOrder.toAffineMap(rewriter.getContext())));

    return mlir::success();
}

void TransposeToPermuteCast::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<IE::TransposeOp>();
    target.addLegalOp<IE::ReorderOp>();
    target.addLegalOp<IE::PermuteCastOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<TransposeToPermuteCast::TransposeOpConverter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createTransposeToPermuteCastPass(Logger log) {
    return std::make_unique<TransposeToPermuteCast>(log);
}
