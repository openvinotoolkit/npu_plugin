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

#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// QuantizeConstPass
//

class QuantizeConstPass final : public IE::QuantizeConstBase<QuantizeConstPass> {
public:
    explicit QuantizeConstPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class QuantizeConst;

private:
    void safeRunOnFunc() final;
};

//
// QuantizeConst
//

class QuantizeConstPass::QuantizeConst final : public mlir::OpRewritePattern<mlir::quant::QuantizeCastOp> {
public:
    QuantizeConst(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<mlir::quant::QuantizeCastOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::quant::QuantizeCastOp qCastOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult QuantizeConstPass::QuantizeConst::matchAndRewrite(mlir::quant::QuantizeCastOp qCastOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    auto inputConst = qCastOp.arg().getDefiningOp<ConstantInterface>();

    if (inputConst == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got QuantizeCast Operation '{0}' with Constant input '{1}'", qCastOp->getLoc(), inputConst.getLoc());

    const auto qType = qCastOp.getType().cast<mlir::ShapedType>();

    const auto constAttr = quantize(inputConst.getContent(), qType, qCastOp.getLoc());
    if (constAttr == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::ConstantOp>(qCastOp, qType, constAttr);
    return mlir::success();
}

//
// safeRunOnFunc
//

void QuantizeConstPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<QuantizeConst>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createQuantizeConstPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createQuantizeConstPass(Logger log) {
    return std::make_unique<QuantizeConstPass>(log);
}
