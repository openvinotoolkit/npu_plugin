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

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// UseFakeQuant
//

class UseFakeQuant final : public mlir::OpRewritePattern<mlir::quant::DequantizeCastOp> {
public:
    UseFakeQuant(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<mlir::quant::DequantizeCastOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::quant::DequantizeCastOp dCastOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult UseFakeQuant::matchAndRewrite(mlir::quant::DequantizeCastOp dCastOp,
                                                  mlir::PatternRewriter& rewriter) const {
    auto qCastOp = dCastOp.arg().getDefiningOp<mlir::quant::QuantizeCastOp>();

    if (qCastOp == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got QuantizeCast ('{0}') -> DequantizeCast ('{1}') pair", qCastOp.getLoc(), dCastOp.getLoc());

    const auto qType = dCastOp.arg().getType().cast<mlir::ShapedType>();

    int64_t levels = 0;
    mlir::RankedTensorType attrType;
    mlir::DenseElementsAttr rMinAttr, rMaxAttr;
    if (mlir::failed(getFakeQuantParams(qType, levels, attrType, rMinAttr, rMaxAttr, dCastOp.getLoc()))) {
        return mlir::failure();
    }

    auto rMinOp = rewriter.create<Const::DeclareOp>(dCastOp.getLoc(), attrType, Const::ContentAttr::get(rMinAttr));
    auto rMaxOp = rewriter.create<Const::DeclareOp>(dCastOp.getLoc(), attrType, Const::ContentAttr::get(rMaxAttr));

    rewriter.replaceOpWithNewOp<IE::FakeQuantizeOp>(dCastOp, qCastOp.arg(), rMinOp.output(), rMaxOp.output(),
                                                    rMinOp.output(), rMaxOp.output(), levels,
                                                    IE::AutoBroadcastType::NUMPY);

    return mlir::success();
}

//
// MergeFakeQuantPass
//

class MergeFakeQuantPass final : public IE::MergeFakeQuantBase<MergeFakeQuantPass> {
public:
    explicit MergeFakeQuantPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void MergeFakeQuantPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<UseFakeQuant>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMergeFakeQuantPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createMergeFakeQuantPass(Logger log) {
    return std::make_unique<MergeFakeQuantPass>(log);
}
