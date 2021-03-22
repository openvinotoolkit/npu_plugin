//
// Copyright Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/utils/quantization.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// SplitFakeQuantPass
//

class SplitFakeQuantPass final : public IE::SplitFakeQuantBase<SplitFakeQuantPass> {
public:
    explicit SplitFakeQuantPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    void runOnFunction() final;

public:
    class UseQuantDequant;
    class UseConstDequant;

private:
    void passBody();

private:
    Logger _log;
};

void SplitFakeQuantPass::runOnFunction() {
    try {
        _log.trace("Run on Function '@{0}'", getFunction().sym_name());

        passBody();
    } catch (const std::exception& e) {
        (void)errorAt(getOperation(), "{0} Pass failed : {1}", getName(), e.what());
        signalPassFailure();
    }
}

//
// UseQuantDequant
//

class SplitFakeQuantPass::UseQuantDequant final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    UseQuantDequant(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SplitFakeQuantPass::UseQuantDequant::matchAndRewrite(IE::FakeQuantizeOp origOp,
                                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("Got FakeQuantize Operation '{0}'", origOp->getLoc());
    auto innerLog = _log.nest();

    auto inLowConst = origOp.input_low().getDefiningOp<ConstantInterface>();
    auto inHighConst = origOp.input_high().getDefiningOp<ConstantInterface>();
    auto outLowConst = origOp.output_low().getDefiningOp<ConstantInterface>();
    auto outHighConst = origOp.output_high().getDefiningOp<ConstantInterface>();

    if (inLowConst == nullptr || inHighConst == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
        innerLog.trace("Got non constant parameters");
        return mlir::failure();
    }

    const auto outLowAttr = outLowConst.getContent();
    const auto outHighAttr = outHighConst.getContent();

    if (inLowConst.getContent() != outLowAttr || inHighConst.getContent() != outHighAttr) {
        innerLog.trace("Input/output parameters mismatch");
        return mlir::failure();
    }

    innerLog.trace("Try to use quantize/dequantize pair");

    const auto realType = origOp.input().getType().cast<mlir::ShapedType>();
    const auto realElemType = realType.getElementType().cast<mlir::FloatType>();

    const auto qElemType = getQuantizedType(outLowConst, outHighConst, origOp.levels(), realElemType, origOp.getLoc());
    if (qElemType == nullptr) {
        return mlir::failure();
    }

    innerLog.trace("Use quantized element type '{0}'", qElemType);

    const auto qType = mlir::RankedTensorType::getChecked(origOp.getLoc(), realType.getShape(), qElemType);
    if (qType == nullptr) {
        return mlir::failure();
    }

    auto quantOp = rewriter.create<mlir::quant::QuantizeCastOp>(origOp.getLoc(), qType, origOp.input());
    rewriter.replaceOpWithNewOp<mlir::quant::DequantizeCastOp>(origOp, realType, quantOp.getResult());

    return mlir::success();
}

//
// UseConstDequant
//

class SplitFakeQuantPass::UseConstDequant final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    UseConstDequant(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SplitFakeQuantPass::UseConstDequant::matchAndRewrite(IE::FakeQuantizeOp origOp,
                                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("Got FakeQuantize Operation '{0}'", origOp->getLoc());
    auto innerLog = _log.nest();

    auto inConst = origOp.input().getDefiningOp<ConstantInterface>();
    if (inConst == nullptr) {
        innerLog.trace("Got non constant input");
        return mlir::failure();
    }

    auto inLowConst = origOp.input_low().getDefiningOp<ConstantInterface>();
    auto inHighConst = origOp.input_high().getDefiningOp<ConstantInterface>();
    auto outLowConst = origOp.output_low().getDefiningOp<ConstantInterface>();
    auto outHighConst = origOp.output_high().getDefiningOp<ConstantInterface>();

    if (inLowConst == nullptr || inHighConst == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
        innerLog.trace("Got non constant parameters");
        return mlir::failure();
    }

    const auto inLowAttr = inLowConst.getContent();
    const auto inHighAttr = inHighConst.getContent();

    if (!inLowAttr.isSplat() || !inHighAttr.isSplat()) {
        innerLog.trace("Input min/max are not splat values");
        return mlir::failure();
    }

    // TODO: should we check the inLowAttr/inHighAttr values some how?

    innerLog.trace("Try to use constant dequantize");

    const auto realType = inConst.getActualType();
    const auto realElemType = realType.getElementType().cast<mlir::FloatType>();

    const auto qElemType = getQuantizedType(outLowConst, outHighConst, origOp.levels(), realElemType, origOp.getLoc());
    if (qElemType == nullptr) {
        return mlir::failure();
    }

    innerLog.trace("Use quantized element type '{0}'", qElemType);

    const auto qType = mlir::RankedTensorType::getChecked(origOp.getLoc(), realType.getShape(), qElemType);
    if (qType == nullptr) {
        return mlir::failure();
    }

    auto newInOp = rewriter.create<IE::ConstantOp>(inConst->getLoc(), qType, inConst.getContent());
    rewriter.replaceOpWithNewOp<mlir::quant::DequantizeCastOp>(origOp, origOp.getType(), newInOp.output());

    return mlir::success();
}

//
// passBody
//

void SplitFakeQuantPass::passBody() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<IE::FakeQuantizeOp>();
    target.addLegalOp<IE::ConstantOp>();
    target.addLegalOp<mlir::quant::QuantizeCastOp>();
    target.addLegalOp<mlir::quant::DequantizeCastOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<UseQuantDequant>(&ctx, _log.nest());
    patterns.insert<UseConstDequant>(&ctx, _log.nest());

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
