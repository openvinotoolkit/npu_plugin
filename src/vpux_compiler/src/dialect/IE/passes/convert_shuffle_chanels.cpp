//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// ConvertShuffleChannelsPass
//

class ConvertShuffleChannelsPass final : public IE::ConvertShuffleChannelsBase<ConvertShuffleChannelsPass> {
public:
    explicit ConvertShuffleChannelsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class ShuffleChannelsOpConverter;

private:
    void safeRunOnFunc() final;
};

//
// ShuffleChannelsOpConverter
//

class ConvertShuffleChannelsPass::ShuffleChannelsOpConverter final :
        public mlir::OpRewritePattern<IE::ShuffleChannelsOp> {
public:
    ShuffleChannelsOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ShuffleChannelsOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ShuffleChannelsOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertShuffleChannelsPass::ShuffleChannelsOpConverter::matchAndRewrite(
        IE::ShuffleChannelsOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto inputShape = origOp.input().getType().cast<vpux::NDTypeInterface>().getShape().raw();
    const auto outShape = origOp.output().getType().cast<vpux::NDTypeInterface>().getShape().raw();
    const auto axis = origOp.axis();
    const auto group = origOp.group();

    // Compute 1st shape ( e.g. for inputShape = {N,C,H,W}, axis=1
    // => shape1 = {N, group, C / group, H * W} )
    std::array<int64_t, 4> shape1 = {1, 1, 1, 1};
    // Allow negative axis
    const auto _axis = axis >= 0 ? axis : axis + inputShape.size();
    // All dims before 'axis' dim
    for (size_t i = 0; i < _axis; i++) {
        shape1[0] *= inputShape[i];
    }
    shape1[1] = group;
    shape1[2] = inputShape[_axis] / group;
    // All dims after 'axis' dim
    for (size_t i = _axis + 1; i < inputShape.size(); i++) {
        shape1[3] *= inputShape[i];
    }

    const auto shape1Attr = getIntArrayAttr(getContext(), shape1);
    auto reShape1Op = rewriter.create<IE::ReshapeOp>(origOp->getLoc(), origOp.input(), nullptr, false, shape1Attr);

    const SmallVector<uint32_t> permuteNdOrder{0, 2, 1, 3};
    const auto permutationMap = mlir::AffineMap::getPermutationMap(makeArrayRef(permuteNdOrder), getContext());
    auto transpOp = rewriter.create<IE::TransposeOp>(origOp->getLoc(), reShape1Op.output(), nullptr,
                                                     mlir::AffineMapAttr::get(permutationMap));

    const auto outShapeAttr = getIntArrayAttr(getContext(), outShape);
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, transpOp.output(), nullptr, false, outShapeAttr);

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertShuffleChannelsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<IE::ShuffleChannelsOp>();
    target.addLegalOp<IE::TransposeOp>();
    target.addLegalOp<IE::ReshapeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ShuffleChannelsOpConverter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertShuffleChannelsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertShuffleChannelsPass(Logger log) {
    return std::make_unique<ConvertShuffleChannelsPass>(log);
}
