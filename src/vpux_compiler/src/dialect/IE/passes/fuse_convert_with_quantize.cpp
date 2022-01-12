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

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/IE/loop.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// LayerRewriter
//

class LayerRewriter final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    LayerRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        setDebugName("LayerRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

class LayerRewriterDequantize final : public mlir::OpRewritePattern<IE::ConvertOp> {
public:
    LayerRewriterDequantize(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConvertOp>(ctx), _log(log) {
        setDebugName("LayerRewriterDequantize");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvertOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult LayerRewriter::matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const {
    auto convertOp = quantizeOp.input().getDefiningOp<IE::ConvertOp>();
    if (convertOp == nullptr) {
        return mlir::failure();
    }

    auto inElemType = convertOp.input().getType().cast<mlir::ShapedType>().getElementType();
    if (!inElemType.isInteger(CHAR_BIT)) {
        return mlir::failure();
    }

    const double inDataScale = 1.0;
    const int64_t inDataZP = 0;
    mlir::quant::QuantizedType dstType;
    auto originDstType = quantizeOp.dstElemType();
    if (const auto uniformType = originDstType.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        dstType = mlir::quant::UniformQuantizedType::getChecked(
                quantizeOp.getLoc(), uniformType.isSigned(), uniformType.getStorageType(),
                uniformType.getExpressedType(), inDataScale, inDataZP, uniformType.getStorageTypeMin(),
                uniformType.getStorageTypeMax());
    } else if (const auto perAxisType = originDstType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        SmallVector<double> scales(perAxisType.getScales().size(), inDataScale);
        SmallVector<int64_t> zeroPoints(perAxisType.getScales().size(), inDataZP);

        dstType = mlir::quant::UniformQuantizedPerAxisType::getChecked(
                quantizeOp.getLoc(), perAxisType.isSigned(), perAxisType.getStorageType(),
                perAxisType.getExpressedType(), scales, zeroPoints, perAxisType.getQuantizedDimension(),
                perAxisType.getStorageTypeMin(), perAxisType.getStorageTypeMax());
    } else {
        VPUX_THROW("Unsupported Quantized Type {0}", originDstType);
    }

    rewriter.replaceOpWithNewOp<IE::QuantizeCastOp>(quantizeOp, convertOp.input(), dstType);
    return mlir::success();
}

mlir::LogicalResult LayerRewriterDequantize::matchAndRewrite(IE::ConvertOp convertOp, mlir::PatternRewriter& rewriter) const {
    auto dequantizeOp = convertOp.input().getDefiningOp<IE::DequantizeOp>();
    if (dequantizeOp == nullptr) {
        return mlir::failure();
    }

    // Next op is either IE.Quantize (identity qp) or IE.QuantizeCast (non-identity qp)
    auto quantizeCastOp = dequantizeOp.input().getDefiningOp<IE::QuantizeCastOp>();
    auto quantizeOp = dequantizeOp.input().getDefiningOp<IE::QuantizeOp>();
    if (quantizeCastOp == nullptr && quantizeOp == nullptr) {
        return mlir::failure();
    } else if (quantizeOp) {
        rewriter.replaceOpWithNewOp<IE::QuantizeCastOp>(dequantizeOp, quantizeOp.output(), convertOp.output().getType().cast<mlir::ShapedType>().getElementType());
    } else if (quantizeCastOp) {
        quantizeOp = quantizeCastOp.input().getDefiningOp<IE::QuantizeOp>();
        if (quantizeOp == nullptr) {
            return mlir::failure();
        } else {
            rewriter.replaceOpWithNewOp<IE::QuantizeCastOp>(dequantizeOp, quantizeOp.output(), convertOp.output().getType().cast<mlir::ShapedType>().getElementType());
        }
    }

    return mlir::success();
}

//
// FuseConvertWithQuantizePass
//

class FuseConvertWithQuantizePass final : public IE::FuseConvertWithQuantizeBase<FuseConvertWithQuantizePass> {
public:
    explicit FuseConvertWithQuantizePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void FuseConvertWithQuantizePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::OwningRewritePatternList patterns(&ctx);
    patterns.add<LayerRewriter>(&ctx, _log);
    patterns.add<LayerRewriterDequantize>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}
}  // namespace

//
// createFuseConvertWithQuantizePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseConvertWithQuantizePass(Logger log) {
    return std::make_unique<FuseConvertWithQuantizePass>(log);
}
