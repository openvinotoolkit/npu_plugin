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

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// FuseWithConv
//

//
//       [input]
//          |
//     (dequantize)
//          |
//        (conv) --- (dequantize) -- [filter]
//          |
//       [output]
//          |
//      (quantize)
//

class FuseWithConv final : public mlir::OpRewritePattern<mlir::quant::QuantizeCastOp> {
public:
    FuseWithConv(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<mlir::quant::QuantizeCastOp>(ctx), _log(log) {
        setDebugName("FuseWithConv");
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::quant::QuantizeCastOp quantizeOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseWithConv::matchAndRewrite(mlir::quant::QuantizeCastOp quantizeOp,
                                                  mlir::PatternRewriter& rewriter) const {
    auto convOp = quantizeOp.arg().getDefiningOp<IE::ConvolutionOp>();
    if (convOp == nullptr) {
        return mlir::failure();
    }

    auto inputDequantizeOp = convOp.input().getDefiningOp<mlir::quant::DequantizeCastOp>();
    if (inputDequantizeOp == nullptr) {
        return mlir::failure();
    }

    auto filterDequantizeOp = convOp.filter().getDefiningOp<mlir::quant::DequantizeCastOp>();
    if (filterDequantizeOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::ConvolutionOp>(
            quantizeOp, quantizeOp.getType(), inputDequantizeOp.arg(), filterDequantizeOp.arg(), convOp.bias(),
            convOp.strides(), convOp.pads_begin(), convOp.pads_end(), convOp.dilations(), convOp.post_opAttr());

    return mlir::success();
}

//
// FuseWithMaxPool
//

//
//       [input]
//          |
//     (dequantize)
//          |
//        (pool)
//          |
//       [output]
//          |
//      (quantize)
//

class FuseWithMaxPool final : public mlir::OpRewritePattern<mlir::quant::QuantizeCastOp> {
public:
    FuseWithMaxPool(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<mlir::quant::QuantizeCastOp>(ctx), _log(log) {
        setDebugName("FuseWithMaxPool");
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::quant::QuantizeCastOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseWithMaxPool::matchAndRewrite(mlir::quant::QuantizeCastOp quantizeOp,
                                                     mlir::PatternRewriter& rewriter) const {
    auto maxPoolOp = quantizeOp.arg().getDefiningOp<IE::MaxPoolOp>();
    if (maxPoolOp == nullptr) {
        return mlir::failure();
    }

    auto inputDequantizeOp = maxPoolOp.input().getDefiningOp<mlir::quant::DequantizeCastOp>();
    if (inputDequantizeOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::MaxPoolOp>(
            quantizeOp, quantizeOp.getType(), inputDequantizeOp.arg(), maxPoolOp.kernel_size(), maxPoolOp.strides(),
            maxPoolOp.pads_begin(), maxPoolOp.pads_end(), maxPoolOp.rounding_type(), maxPoolOp.post_opAttr());

    return mlir::success();
}

//
// PropagateQuantizeDequantizePass
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

    mlir::OwningRewritePatternList patterns(&ctx);
    patterns.add<FuseWithConv>(&ctx, _log);
    patterns.add<FuseWithMaxPool>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
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
