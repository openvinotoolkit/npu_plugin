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

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

using namespace vpux;

namespace {

class CanFuseDeQuantAndQuantIntoConv final : public mlir::OpRewritePattern<mlir::quant::QuantizeCastOp> {
public:
    CanFuseDeQuantAndQuantIntoConv(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<mlir::quant::QuantizeCastOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::quant::QuantizeCastOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult CanFuseDeQuantAndQuantIntoConv::matchAndRewrite(mlir::quant::QuantizeCastOp origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    //     convInputDeQuantize
    //          |
    //          |
    //         conv --- dequant -- weights
    //          |
    //          |
    //      outputQuantize

    auto conv = origOp.arg().getDefiningOp<IE::ConvolutionOp>();
    if (conv == nullptr) {
        return mlir::failure();
    }

    auto dequantizeWeights = conv.filter().getDefiningOp<mlir::quant::DequantizeCastOp>();
    if (dequantizeWeights == nullptr) {
        return mlir::failure();
    }

    auto convInputDeQuantize = conv.input().getDefiningOp<mlir::quant::DequantizeCastOp>();
    if (convInputDeQuantize == nullptr) {
        return mlir::failure();
    }

    auto newConv = rewriter.create<IE::ConvolutionOp>(
            conv.getLoc(), origOp.getType(), convInputDeQuantize.arg(), dequantizeWeights.arg(), conv.bias(),
            conv.strides(), conv.pads_begin(), conv.pads_end(), conv.dilations(), conv.post_opAttr());

    rewriter.replaceOp(origOp, newConv->getResults());
    return mlir::success();
}

class CanFuseDeQuantAndQuantIntoPool final : public mlir::OpRewritePattern<mlir::quant::QuantizeCastOp> {
public:
    CanFuseDeQuantAndQuantIntoPool(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<mlir::quant::QuantizeCastOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::quant::QuantizeCastOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult CanFuseDeQuantAndQuantIntoPool::matchAndRewrite(mlir::quant::QuantizeCastOp origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    //     convInputDeQuantize
    //          |
    //          |
    //         pool
    //          |
    //          |
    //      outputQuantize

    auto maxPool = origOp.arg().getDefiningOp<IE::MaxPoolOp>();
    if (maxPool == nullptr) {
        return mlir::failure();
    }

    auto maxPoolInputDeQuantize = maxPool.input().getDefiningOp<mlir::quant::DequantizeCastOp>();
    if (maxPoolInputDeQuantize == nullptr) {
        return mlir::failure();
    }

    auto newMaxPool = rewriter.create<IE::MaxPoolOp>(
            maxPool.getLoc(), origOp.getType(), maxPoolInputDeQuantize.arg(), maxPool.kernel_size(), maxPool.strides(),
            maxPool.pads_begin(), maxPool.pads_end(), maxPool.rounding_type(), maxPool.post_opAttr());

    rewriter.replaceOp(origOp, newMaxPool->getResults());
    return mlir::success();
}

//
// PropagateQuantizeDequantize
//

class PropagateQuantizeDequantizePass final :
        public IE::PropagateQuantizeDequantizeBase<PropagateQuantizeDequantizePass> {
public:
    explicit PropagateQuantizeDequantizePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void PropagateQuantizeDequantizePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    mlir::OwningRewritePatternList patterns(&ctx);
    patterns.insert<CanFuseDeQuantAndQuantIntoConv>(&ctx, _log);
    patterns.insert<CanFuseDeQuantAndQuantIntoPool>(&ctx, _log);
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
        signalPassFailure();
    }
}
}  // namespace

//
// createSplitFakeQuantPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateQuantizeDequantizePass(Logger log) {
    return std::make_unique<PropagateQuantizeDequantizePass>(log);
}
