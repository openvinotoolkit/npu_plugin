//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

//
// TransposeInterpolation
//

class TransposeInterpolation final : public mlir::OpRewritePattern<IE::InterpolateOp> {
public:
    TransposeInterpolation(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::InterpolateOp>(ctx), _log(log) {
        this->setDebugName("TransposeInterpolation");
    }

    SmallVector<int64_t> getInterpolationAxes(IE::InterpolateOp op) const;
    bool isSpatialInterpolation(ArrayRef<int64_t> interpolationAxes, int64_t shapeRank) const;

private:
    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Get the real interpolation axes by IO shapes
// There are cases where the axis attribute contains all four dimensions but the interpolation is done only on the
// spatial axes, so only the IO shapes can define the interpolation.
SmallVector<int64_t> TransposeInterpolation::getInterpolationAxes(IE::InterpolateOp op) const {
    const auto inputShape = getShape(op.getInput());
    const auto outputShape = getShape(op.getOutput());
    const auto rank = inputShape.size();

    SmallVector<int64_t> interpolationAxes;
    for (size_t i = 0; i < checked_cast<size_t>(rank); i++) {
        if (inputShape[Dim(i)] != outputShape[Dim(i)]) {
            interpolationAxes.push_back(checked_cast<int64_t>(i));
        }
    }

    // Adjust for single interpolation axis
    if (interpolationAxes.size() == 1) {
        if (interpolationAxes[0] == checked_cast<int64_t>(rank) - 1) {
            interpolationAxes.insert(interpolationAxes.begin(), interpolationAxes[0] - 1);
        } else {
            interpolationAxes.push_back(interpolationAxes[0] + 1);
        }
    }

    return interpolationAxes;
}

bool TransposeInterpolation::isSpatialInterpolation(ArrayRef<int64_t> interpolationAxes, int64_t shapeRank) const {
    for (size_t i = 0; i < interpolationAxes.size(); i++) {
        if (interpolationAxes[interpolationAxes.size() - 1 - i] != shapeRank - 1 - checked_cast<int64_t>(i)) {
            return false;
        }
    }

    return true;
}

mlir::LogicalResult TransposeInterpolation::matchAndRewrite(IE::InterpolateOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    if (!origOp.getSizesAttr().has_value() || !origOp.getScalesAttr().has_value() ||
        !origOp.getAxesAttr().has_value()) {
        _log.trace("InterpolateOp {0} at {1} has no size scale or axes attribute", origOp->getName(), origOp->getLoc());
        return mlir::failure();
    }

    const auto realInterpolationAxes = getInterpolationAxes(origOp);
    const auto inputShape = getShape(origOp.getInput());
    const auto rank = inputShape.size();

    if (isSpatialInterpolation(realInterpolationAxes, rank)) {
        _log.trace("Bypass {0} at {1}, which is spatial interpolaton already", origOp->getName(), origOp->getLoc());
        return mlir::failure();
    }

    auto ctx = rewriter.getContext();

    _log.trace("Got non-spatial interpolation {0} at {1}, interpolation axes {2}", origOp->getName(), origOp->getLoc(),
               realInterpolationAxes);

    // Create input Transpose
    SmallVector<uint32_t> inMemPerm;
    for (size_t i = 0; i < rank; i++) {
        if (std::find(realInterpolationAxes.begin(), realInterpolationAxes.end(), i) == realInterpolationAxes.end()) {
            inMemPerm.push_back(checked_cast<uint32_t>(i));
        }
    }
    for (const auto& axis : realInterpolationAxes) {
        inMemPerm.push_back(checked_cast<uint32_t>(axis));
    }

    auto inOrderMap = mlir::AffineMap::getPermutationMap(inMemPerm, ctx);
    auto inOrderAttr = mlir::AffineMapAttr::get(inOrderMap);

    _log.nest().trace("Create input Transpose with order attribute {0}", inOrderAttr);
    auto inTranspose = rewriter.create<IE::TransposeOp>(origOp->getLoc(), origOp.getInput(), nullptr, inOrderAttr);

    // Create new Interpolate
    auto origAxesValue = parseIntArrayAttr<int64_t>(origOp.getAxesAttrAttr());
    auto origScalesValue = parseFPArrayAttr<double>(origOp.getScalesAttrAttr());
    auto origSizesValue = parseIntArrayAttr<int64_t>(origOp.getSizesAttrAttr());

    SmallVector<int64_t> newAxesValue;
    SmallVector<double> newScalesValue;
    SmallVector<int64_t> newSizesValue;
    if (origAxesValue.size() == 2) {
        for (size_t i = 0; i < origAxesValue.size(); i++) {
            newAxesValue.insert(newAxesValue.begin(), checked_cast<int64_t>(rank - 1 - i));
        }
        newScalesValue.assign(origScalesValue);
        newSizesValue.assign(origSizesValue);
    } else if (origAxesValue.size() == rank) {
        newAxesValue.assign(origAxesValue);
        for (size_t i = 0; i < origScalesValue.size(); i++) {
            newScalesValue.push_back(origScalesValue[inMemPerm[i]]);
        }
        for (size_t i = 0; i < origSizesValue.size(); i++) {
            newSizesValue.push_back(origSizesValue[inMemPerm[i]]);
        }
    } else {
        return matchFailed(_log, rewriter, origOp, "InterpolateOp {0} has invalid axes attribute", origOp->getLoc());
    }

    const auto newAxesAttr = getIntArrayAttr(ctx, newAxesValue);
    const auto newScalesAttr = getFPArrayAttr(ctx, newScalesValue);
    const auto newSizesAttr = getIntArrayAttr(ctx, newSizesValue);
    _log.nest().trace("Create new Interpolate with axes attr {0}, scales attr {1}, sizes attr {2}", newAxesAttr,
                      newScalesAttr, newSizesAttr);
    auto newInterpolate = rewriter.create<IE::InterpolateOp>(
            origOp->getLoc(), inTranspose.getOutput(), origOp.getSizes(), origOp.getScales(), origOp.getAxes(),
            newSizesAttr, newScalesAttr, newAxesAttr, origOp.getTileOffsetAttrAttr(),
            origOp.getInitialInputDimsAttrAttr(), origOp.getInitialOutputDimsAttrAttr(), origOp.getAttr());

    // Create output Transpose
    auto outOrderMap = mlir::inversePermutation(inOrderMap);
    auto outOrderAttr = mlir::AffineMapAttr::get(outOrderMap);

    _log.nest().trace("Create output Transpose with order attribute {0}", outOrderAttr);
    auto outTranspose =
            rewriter.create<IE::TransposeOp>(origOp->getLoc(), newInterpolate.getOutput(), nullptr, outOrderAttr);

    _log.trace("Finished replacement at {0}", origOp->getLoc());
    rewriter.replaceOp(origOp, outTranspose.getOutput());

    return mlir::success();
}

//
// ConvertToSpatialInterpolationPass
//

class ConvertToSpatialInterpolationPass final :
        public IE::ConvertToSpatialInterpolationBase<ConvertToSpatialInterpolationPass> {
public:
    explicit ConvertToSpatialInterpolationPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

//
// safeRunOnFunc
//

void ConvertToSpatialInterpolationPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<TransposeInterpolation>(&ctx, _log);
    IE::InterpolateOp::getCanonicalizationPatterns(patterns, &ctx);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertToSpatialInterpolationPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertToSpatialInterpolationPass(Logger log) {
    return std::make_unique<ConvertToSpatialInterpolationPass>(log);
}
