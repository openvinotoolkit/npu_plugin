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

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/TypeSwitch.h>
#include <vpux/compiler/conversion.hpp>

using namespace vpux;

namespace {

//
// CollapseMutualTransposes
//

class CollapseMutualTransposes final : public IE::CollapseMutualTransposesBase<CollapseMutualTransposes> {
public:
    explicit CollapseMutualTransposes(Logger log): _log(log) {
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

class CollapseMutualTransposes::TransposeOpConverter final : public mlir::OpRewritePattern<IE::TransposeOp> {
public:
    TransposeOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::TransposeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TransposeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult CollapseMutualTransposes::TransposeOpConverter::matchAndRewrite(
        IE::TransposeOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto lastTransposeIn = origOp.input();
    VPUX_THROW_UNLESS(lastTransposeIn != nullptr, "TransposeOpConverter: transpose input is a null pointer");
    auto maybeReshapeOp = lastTransposeIn.getDefiningOp<IE::AffineReshapeOp>();
    VPUX_THROW_UNLESS(maybeReshapeOp != nullptr, "TransposeOpConverter: transpose input is not a reshape");
    auto reshapeIn = maybeReshapeOp.input();
    VPUX_THROW_UNLESS(reshapeIn != nullptr, "TransposeOpConverter: reshape input is a null pointer");
    auto firstTranspose = reshapeIn.getDefiningOp<IE::TransposeOp>();
    VPUX_THROW_UNLESS(firstTranspose != nullptr, "TransposeOpConverter: rehsape input is not a transpose");
    auto firstTransposeIn = firstTranspose.input();
    VPUX_THROW_UNLESS(firstTransposeIn != nullptr,
                      "TransposeOpConverter: input of the first transpose is a null pointer");

    const auto shape = origOp.output().getType().cast<mlir::ShapedType>().getShape();
    const auto newShape = to_small_vector(shape);
    const auto newShapeAttr = getIntArrayAttr(rewriter.getContext(), newShape);
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, firstTransposeIn, nullptr, false, newShapeAttr);

    return mlir::success();
}

bool isTrivialReshape(const ShapeRef inShape, const ShapeRef outShape) {
    // Remove all trivial dimensions.
    std::vector<int64_t> inShapeVec;
    for (size_t ind = 0; ind < inShape.size(); ind++) {
        const auto dimVal = inShape[Dim(ind)];
        if (dimVal > 1) {
            inShapeVec.push_back(dimVal);
        }
    }

    std::vector<int64_t> outShapeVec;
    for (size_t ind = 0; ind < outShape.size(); ind++) {
        const auto dimVal = outShape[Dim(ind)];
        if (dimVal > 1) {
            outShapeVec.push_back(dimVal);
        }
    }

    if (inShapeVec.size() != inShapeVec.size()) {
        return false;
    }

    for (size_t ind = 0; ind < inShapeVec.size() && ind < inShapeVec.size(); ind++) {
        if (inShapeVec[ind] != outShapeVec[ind]) {
            return false;
        }
    }

    return true;
}

bool canBeCollapsed(IE::TransposeOp op) {
    // First, check whether the input of that transpose operation is a reshape operation.
    const auto lastTransposeIn = op.input();
    if (!lastTransposeIn) {
        return false;
    }

    auto maybeReshapeOp = lastTransposeIn.getDefiningOp<IE::AffineReshapeOp>();
    if (maybeReshapeOp == nullptr) {
        return false;
    }

    // Now, find out whether this reshape operation has another transpose as an input.
    const auto reshapeIn = maybeReshapeOp.input();
    if (!reshapeIn) {
        return false;
    }

    auto firstTranspose = reshapeIn.getDefiningOp<IE::TransposeOp>();
    if (firstTranspose == nullptr) {
        return false;
    }

    // Only trivial reshapes can be collapsed.
    // Trivial means that all dimensions larger than 1 preserve order.
    // Examples:
    // 1x1x28x70 -> Reshape -> 1x28x70 -- trivial
    // 1x28x70x1 -> Reshape -> 1x28x70 -- trivial
    // 1x28x1x70 -> Reshape -> 1x28x70 -- trivial
    // 1x28x1x70 -> Reshape -> 1x70x28 -- non-trivial, since the order is not preserved.
    if (!isTrivialReshape(getShape(maybeReshapeOp.input()), getShape(maybeReshapeOp.output()))) {
        return false;
    }

    // Check that the second reshape is the inverse of the first one.
    if (!isTrivialReshape(getShape(firstTranspose.input()), getShape(op.output()))) {
        return false;
    }

    return true;
}

void CollapseMutualTransposes::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    const auto isLegalTranspose = [](IE::TransposeOp op) -> bool {
        // The possibility to collapse that transpose makes it illegal
        return !canBeCollapsed(op);
    };
    target.addDynamicallyLegalOp<IE::TransposeOp>(isLegalTranspose);
    target.addLegalOp<IE::ReshapeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<CollapseMutualTransposes::TransposeOpConverter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createCollapseMutualTransposes(Logger log) {
    return std::make_unique<CollapseMutualTransposes>(log);
}
