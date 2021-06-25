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
// DequantizeConstPass
//

class DequantizeConstPass final : public IE::DequantizeConstBase<DequantizeConstPass> {
public:
    explicit DequantizeConstPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class DequantizeConst;

private:
    void safeRunOnFunc() final;
};

//
// DequantizeConst
//

class DequantizeConstPass::DequantizeConst final : public mlir::OpRewritePattern<mlir::quant::DequantizeCastOp> {
public:
    DequantizeConst(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<mlir::quant::DequantizeCastOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::quant::DequantizeCastOp dCastOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DequantizeConstPass::DequantizeConst::matchAndRewrite(mlir::quant::DequantizeCastOp dCastOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    auto inputConst = dCastOp.arg().getDefiningOp<ConstantInterface>();

    if (inputConst == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got DequantizeCast Operation '{0}' with Constant input '{1}'", dCastOp->getLoc(), inputConst.getLoc());

    const auto qType = inputConst.getActualType();
    const auto qElemType = qType.getElementType().cast<mlir::quant::QuantizedType>();

    const auto newConstType =
            mlir::RankedTensorType::getChecked(dCastOp.getLoc(), qType.getShape(), qElemType.getExpressedType());
    if (newConstType == nullptr) {
        return mlir::failure();
    }

    const auto constAttr = dequantize(inputConst.getContent(), qType, dCastOp.getLoc());
    if (constAttr == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::ConstantOp>(dCastOp, newConstType, constAttr);
    return mlir::success();
}

//
// safeRunOnFunc
//

void DequantizeConstPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<DequantizeConst>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createDequantizeConstPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createDequantizeConstPass(Logger log) {
    return std::make_unique<DequantizeConstPass>(log);
}
