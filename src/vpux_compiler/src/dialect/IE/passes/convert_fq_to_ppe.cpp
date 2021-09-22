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

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/Value.h>

using namespace vpux;

namespace {

std::pair<mlir::FloatAttr, mlir::FloatAttr> getLowHighAttrs(IE::FakeQuantizeOp fakeQuant) {
    auto outputLow = fakeQuant.output_low().getDefiningOp<vpux::Const::DeclareOp>();
    auto outputHigh = fakeQuant.output_high().getDefiningOp<vpux::Const::DeclareOp>();

    const auto outputLowAttribute = outputLow.contentAttr().fold();
    const auto outputHighAttribute = outputHigh.contentAttr().fold();

    const auto lowAttr = getFPAttr(fakeQuant.getContext(), outputLowAttribute.getSplatValue<double>());
    const auto highAttr = getFPAttr(fakeQuant.getContext(), outputHighAttribute.getSplatValue<double>());

    return {lowAttr, highAttr};
}

//
// FuseWithLayer
//

class FuseWithLayer final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    FuseWithLayer(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx, benefitHigh), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseWithLayer::matchAndRewrite(IE::FakeQuantizeOp originOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got FakeQuantize operation '{0}'", originOp->getLoc());

    if (!originOp.input().hasOneUse()) {
        return matchFailed(_log, rewriter, originOp, "FakeQuantize is not the only user of its input Value");
    }

    auto producerOp = originOp.input().getDefiningOp<IE::LayerWithPostOpInterface>();
    if (producerOp == nullptr) {
        return matchFailed(_log, rewriter, originOp,
                           "FakeQuantize input was not produced by another Operation or the producer does not support "
                           "post-processing");
    }

    auto lowHighAttrs = getLowHighAttrs(originOp);

    producerOp.setClipOp(lowHighAttrs.first, lowHighAttrs.second);
    rewriter.replaceOp(originOp, producerOp->getResult(0));

    _log.trace("Fuse with op '{0}'", producerOp->getLoc());

    return mlir::success();
}

//
// ConvertToMaxPool
//

class ConvertToMaxPool final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    ConvertToMaxPool(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx, benefitLow), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertToMaxPool::matchAndRewrite(IE::FakeQuantizeOp originOp,
                                                      mlir::PatternRewriter& rewriter) const {
    _log.trace("Got FakeQuantize operation '{0}'", originOp->getLoc());

    auto lowHighAttrs = getLowHighAttrs(originOp);

    const auto attrClipOp = IE::ClipOp::get(lowHighAttrs.first, lowHighAttrs.second, getContext());
    const auto attrKernelSize = getIntArrayAttr(getContext(), SmallVector<int64_t>{1, 1});
    const auto attrKernelStrides = getIntArrayAttr(getContext(), SmallVector<int64_t>{1, 1});
    const auto attrPadsBegin = getIntArrayAttr(getContext(), SmallVector<int64_t>{0, 0});
    const auto attrPadsEnd = getIntArrayAttr(getContext(), SmallVector<int64_t>{0, 0});
    const auto attrRoundingType = IE::RoundingTypeAttr::get(getContext(), IE::RoundingType::FLOOR);

    // FakeQuantize with the same input and output ranges produces a tensor
    // with the same shape and element type(as input),
    // but which will be clamped according to the operation specification.
    // So let's convert FakeQuantize to a trivial MaxPool with a clip operation to improve performance
    rewriter.replaceOpWithNewOp<IE::MaxPoolOp>(originOp, originOp.input(), attrKernelSize, attrKernelStrides,
                                               attrPadsBegin, attrPadsEnd, attrRoundingType, nullptr, attrClipOp);

    _log.trace("Replace with MaxPool");

    return mlir::success();
}

//
// ConvertFakeQuantizeToPPEOpPass
//

class ConvertFakeQuantizeToPPEOpPass final : public IE::ConvertFakeQuantizeToPPEOpBase<ConvertFakeQuantizeToPPEOpPass> {
public:
    explicit ConvertFakeQuantizeToPPEOpPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void ConvertFakeQuantizeToPPEOpPass::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto isLegal = [&](IE::FakeQuantizeOp op) {
        auto inputLow = op.input_low().getDefiningOp<vpux::Const::DeclareOp>();
        VPUX_THROW_UNLESS(inputLow != nullptr, "Only constant input is supported for input_low");

        auto inputHigh = op.input_high().getDefiningOp<vpux::Const::DeclareOp>();
        VPUX_THROW_UNLESS(inputLow != nullptr, "Only constant input is supported for input_high");

        auto outputLow = op.output_low().getDefiningOp<vpux::Const::DeclareOp>();
        VPUX_THROW_UNLESS(inputLow != nullptr, "Only constant input is supported for output_low");

        auto outputHigh = op.output_high().getDefiningOp<vpux::Const::DeclareOp>();
        VPUX_THROW_UNLESS(inputLow != nullptr, "Only constant input is supported for output_high");

        const auto inputLowAttribute = inputLow.contentAttr().fold();
        const auto inputHighAttribute = inputHigh.contentAttr().fold();
        const auto outputLowAttribute = outputLow.contentAttr().fold();
        const auto outputHighAttribute = outputHigh.contentAttr().fold();

        if (!inputLowAttribute.isSplat() || !inputHighAttribute.isSplat() || !outputLowAttribute.isSplat() ||
            !outputHighAttribute.isSplat()) {
            return true;
        }

        auto inputLowVal = inputLowAttribute.getSplatValue<double>();
        auto inputHighVal = inputHighAttribute.getSplatValue<double>();
        auto outputLowVal = outputLowAttribute.getSplatValue<double>();
        auto outputHighVal = outputHighAttribute.getSplatValue<double>();

        if (inputLowVal != outputLowVal || inputHighVal != outputHighVal) {
            return true;
        }

        return false;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::FakeQuantizeOp>(isLegal);
    target.addLegalOp<IE::MaxPoolOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<ConvertToMaxPool>(&ctx, _log);
    patterns.insert<FuseWithLayer>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertFakeQuantizeToPPEOpPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertFakeQuantizeToPPEOpPass(Logger log) {
    return std::make_unique<ConvertFakeQuantizeToPPEOpPass>(log);
}
