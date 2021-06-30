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

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Transforms/DialectConversion.h>
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

namespace {

//
// UseQuantDequant
//

class UseQuantDequant final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    UseQuantDequant(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
        setDebugName("UseQuantDequant");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult UseQuantDequant::matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got FakeQuantize Operation '{1}'", getDebugName(), origOp->getLoc());
    auto innerLog = _log.nest();

    auto inLowConst = origOp.input_low().getDefiningOp<Const::DeclareOp>();
    auto inHighConst = origOp.input_high().getDefiningOp<Const::DeclareOp>();
    auto outLowConst = origOp.output_low().getDefiningOp<Const::DeclareOp>();
    auto outHighConst = origOp.output_high().getDefiningOp<Const::DeclareOp>();

    if (inLowConst == nullptr || inHighConst == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
        return matchFailed(innerLog, rewriter, origOp, "Got non constant parameters");
    }

    if (inLowConst.contentAttr() != outLowConst.contentAttr() ||
        inHighConst.contentAttr() != outHighConst.contentAttr()) {
        return matchFailed(innerLog, rewriter, origOp, "Input/output parameters mismatch");
    }

    innerLog.trace("Try to use quantize/dequantize pair");

    const auto realType = origOp.input().getType().cast<mlir::ShapedType>();
    const auto realElemType = realType.getElementType().cast<mlir::FloatType>();

    const auto qElemType = getQuantizedType(outLowConst.contentAttr(), outHighConst.contentAttr(), origOp.levels(),
                                            realElemType, origOp.getLoc());
    if (qElemType == nullptr) {
        return mlir::failure();
    }

    innerLog.trace("Use quantized element type '{0}'", qElemType);

    const auto qType = changeElemType(realType, qElemType);

    auto quantOp = rewriter.create<mlir::quant::QuantizeCastOp>(origOp.getLoc(), qType, origOp.input());
    rewriter.replaceOpWithNewOp<mlir::quant::DequantizeCastOp>(origOp, realType, quantOp.getResult());

    return mlir::success();
}

//
// UseConstDequant
//

class UseConstDequant final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    UseConstDequant(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
        setDebugName("UseConstDequant");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

static bool is_equal(float a, float b) {
    float epsilon = std::numeric_limits<float>::epsilon();
    return (std::fabs(a - b) <= epsilon);
};

static float compare_limits(const vpux::Const::details::ContentRange<float>& vec1,
                            const vpux::Const::details::ContentRange<float>& vec2) {
    float ratio = 1.;

    // check if FQ input and output limits are equal
    bool result = std::equal(vec1.begin(), vec1.end(), vec2.begin(), is_equal);
    if (!result) {
        // check that all ratios are equal
        std::vector<float> ratios;
        std::transform(vec1.begin(), vec1.end(), vec2.begin(), std::back_inserter(ratios), std::divides<>{});
        if (std::adjacent_find(ratios.begin(), ratios.end(), [](float a, float b) {
                return !is_equal(a, b);
            }) == ratios.end()) {
            ratio = ratios[0];
        } else {
            // Input and output limits has per channel ratio
            return -1.;
        }
    }

    return ratio;
}

mlir::LogicalResult UseConstDequant::matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got FakeQuantize Operation '{1}'", getDebugName(), origOp->getLoc());
    auto innerLog = _log.nest();

    auto inConst = origOp.input().getDefiningOp<Const::DeclareOp>();
    if (inConst == nullptr) {
        return matchFailed(innerLog, rewriter, origOp, "Got non constant input");
    }

    auto inLowConst = origOp.input_low().getDefiningOp<Const::DeclareOp>();
    auto inHighConst = origOp.input_high().getDefiningOp<Const::DeclareOp>();
    auto outLowConst = origOp.output_low().getDefiningOp<Const::DeclareOp>();
    auto outHighConst = origOp.output_high().getDefiningOp<Const::DeclareOp>();

    if (inLowConst == nullptr || inHighConst == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
        return matchFailed(innerLog, rewriter, origOp, "Got non constant parameters");
    }

    const auto inConstAttr = inConst.contentAttr();
    const auto inBaseVals = inConstAttr.getBaseContent();
    const auto inBaseElemType = inBaseVals.getType().getElementType();

    // TODO: make this check more reliable
    if (!inBaseElemType.isa<mlir::IntegerType>()) {
        const auto inLowContent = inLowConst.content();
        const auto inHighContent = inHighConst.content();

        if (!inLowContent.isSplat() || !inHighContent.isSplat()) {
            innerLog.warning("Legacy model, original input values are not integer");

            // Workaround for Float weights, it lacks generality but is ok for old networks
            // Check if FQ can be removed for float weights
            const auto outHighContent = outHighConst.content();
            const auto outLowContent = outLowConst.content();

            const auto inHighVals = inHighContent.getValues<float>();
            const auto outHighVals = outHighContent.getValues<float>();
            float ratioHigh = compare_limits(inHighVals, outHighVals);

            const auto inLowVals = inLowContent.getValues<float>();
            const auto outLowVals = outLowContent.getValues<float>();
            float ratioLow = compare_limits(inLowVals, outLowVals);

            if (ratioHigh < 0) {
                innerLog.warning("In and out limits differ and has per channel ratio, do not support");
                return mlir::failure();
            } else if (!is_equal(ratioHigh, ratioLow)) {
                innerLog.warning("Unsupported case, ratioHigh={0} != ratioLow={1}", ratioHigh, ratioLow);
                return mlir::failure();
            } else if (ratioHigh == 1.) {
                // FQ input and output ranges are equal, only remove FQ
                rewriter.replaceOpWithNewOp<Const::DeclareOp>(origOp, origOp.getType(), inConst.contentAttr());
            } else {
                // FQ input and output ranges are NOT equal, rescale weights
                innerLog.trace("Rescale weights");
                auto ctx = getContext();
                mlir::FloatAttr scale = vpux::getFP32Attr(ctx, checked_cast<float>(ratioHigh));
                const auto newConstAttr = inConst.contentAttr().rescale(scale);
                rewriter.replaceOpWithNewOp<Const::DeclareOp>(origOp, origOp.getType(), newConstAttr);
            }

            return mlir::success();
        }
    }

    innerLog.trace("Try to use constant dequantize");

    const auto realType = inConstAttr.getType();
    const auto realElemType = realType.getElementType().cast<mlir::FloatType>();

    const auto qElemType = getQuantizedType(outLowConst.contentAttr(), outHighConst.contentAttr(), origOp.levels(),
                                            realElemType, origOp.getLoc());
    if (qElemType == nullptr) {
        return mlir::failure();
    }

    innerLog.trace("Use quantized element type '{0}'", qElemType);

    const auto qType = changeElemType(realType, qElemType);

    const auto newInConstAttr =
            inConstAttr.convertElemType(normalizeQuantStorageType(qElemType.getStorageType())).quantCast(qElemType);
    auto newInOp = rewriter.create<Const::DeclareOp>(inConst->getLoc(), qType, newInConstAttr);

    rewriter.replaceOpWithNewOp<mlir::quant::DequantizeCastOp>(origOp, origOp.getType(), newInOp.output());
    return mlir::success();
}

//
// SplitFakeQuantPass
//

class SplitFakeQuantPass final : public IE::SplitFakeQuantBase<SplitFakeQuantPass> {
public:
    explicit SplitFakeQuantPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void SplitFakeQuantPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<IE::FakeQuantizeOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<mlir::quant::QuantizeCastOp>();
    target.addLegalOp<mlir::quant::DequantizeCastOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<UseQuantDequant>(&ctx, _log);
    patterns.insert<UseConstDequant>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createSplitFakeQuantPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createSplitFakeQuantPass(Logger log) {
    return std::make_unique<SplitFakeQuantPass>(log);
}
