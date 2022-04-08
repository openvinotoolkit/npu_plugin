//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/Value.h>

using namespace vpux;

namespace {

//
// ReplaceInfFqPass
//

class ReplaceInfFqPass final : public IE::ReplaceInfFqBase<ReplaceInfFqPass> {
public:
    explicit ReplaceInfFqPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    class InfFqToClampRewriter;

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

class ReplaceInfFqPass::InfFqToClampRewriter final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    InfFqToClampRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReplaceInfFqPass::InfFqToClampRewriter::matchAndRewrite(IE::FakeQuantizeOp originOp,
                                                                            mlir::PatternRewriter& rewriter) const {
    auto ctx = originOp.getContext();
    auto inLowConst = originOp.input_low().getDefiningOp<Const::DeclareOp>();
    auto inHighConst = originOp.input_high().getDefiningOp<Const::DeclareOp>();
    auto outLowConst = originOp.output_low().getDefiningOp<Const::DeclareOp>();
    auto outHighConst = originOp.output_high().getDefiningOp<Const::DeclareOp>();

    if (inLowConst == nullptr || inHighConst == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
        return mlir::failure();
    }

    const auto inLowAttr = inLowConst.content();
    const auto inHighAttr = inHighConst.content();
    const auto outLowAttr = outLowConst.content();
    const auto outHighAttr = outHighConst.content();

    if (!inLowAttr.isSplat() || !inHighAttr.isSplat() || !outLowAttr.isSplat() || !outHighAttr.isSplat()) {
        return mlir::failure();
    }

    const auto inLowVal = inLowAttr.getSplatValue<double>();
    const auto inHighVal = inHighAttr.getSplatValue<double>();
    const auto outLowVal = outLowAttr.getSplatValue<double>();
    const auto outHighVal = outHighAttr.getSplatValue<double>();

    if (inLowVal != outLowVal || inHighVal != outHighVal) {
        return mlir::failure();
    }

    if (inLowVal != -std::numeric_limits<double>::infinity() && inHighVal != std::numeric_limits<double>::infinity()) {
        return mlir::failure();
    }

    const auto fp16Max = 65504.0f;
    const auto min = inLowVal == -std::numeric_limits<double>::infinity() ? -fp16Max : inLowVal;
    const auto max = inHighVal == std::numeric_limits<double>::infinity() ? fp16Max : inHighVal;
    const auto minAttr = getFPAttr(ctx, min);
    const auto maxAttr = getFPAttr(ctx, max);

    auto clampOp = rewriter.create<IE::ClampOp>(originOp.getLoc(), originOp.input(), minAttr, maxAttr);
    rewriter.replaceOp(originOp, clampOp.output());

    return mlir::success();
}

void ReplaceInfFqPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<InfFqToClampRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createReplaceInfFqPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createReplaceInfFqPass(Logger log) {
    return std::make_unique<ReplaceInfFqPass>(log);
}
