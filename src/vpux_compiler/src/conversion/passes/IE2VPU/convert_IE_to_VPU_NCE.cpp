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

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// ConvToNCE
//

class ConvToNCE final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    ConvToNCE(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
        setDebugName("ConvToNCE");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvToNCE::matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto logCb = [&](const llvm::formatv_object_base& msg) {
        std::ignore = matchFailed(_log, rewriter, origOp, "[{0}] {1}", getDebugName(), msg.str());
    };

    const auto inOrder = DimsOrder::fromValue(origOp.input());
    if (inOrder != DimsOrder::NCHW && inOrder != DimsOrder::NHWC) {
        return matchFailed(_log, rewriter, origOp, "Operation at '{0}' has unsupported input layout '{1}'",
                           origOp->getLoc(), inOrder);
    }
    if (!VPU::NCEConvolutionOp::isSupported(origOp, logCb)) {
        return mlir::failure();
    }

    Const::ContentAttr bias;
    if (origOp.bias() != nullptr) {
        auto biasConstOp = origOp.bias().getDefiningOp<Const::DeclareOp>();
        if (biasConstOp == nullptr) {
            return matchFailed(_log, rewriter, origOp, "[{0}] '{1}' at '{2}' has non constant biases", getDebugName(),
                               origOp->getName(), origOp->getLoc());
        }

        bias = biasConstOp.contentAttr();
    }

    auto filter = origOp.filter();
    if (inOrder == DimsOrder::NCHW) {
        filter = VPU::alignChannelMajorWeightsTensor(rewriter, origOp->getLoc(), filter);
    }
    const auto rawFilterShape = getIntArrayAttr(rewriter, getShape(origOp.filter()));

    const auto padAttr = VPU::getPaddingAttr(getContext(), PadInfo(origOp.pads_begin(), origOp.pads_end()));

    auto nceOp = rewriter.create<VPU::NCEConvolutionOp>(origOp->getLoc(), origOp.getType(), origOp.input(), filter,
                                                        bias, origOp.stridesAttr(), padAttr, origOp.post_opAttr(),
                                                        /*ppe=*/nullptr, rawFilterShape);

    rewriter.replaceOp(origOp, nceOp.output());
    return mlir::success();
}

//
// DepthConvToNCE
//

class DepthConvToNCE final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    DepthConvToNCE(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DepthConvToNCE::matchAndRewrite(IE::GroupConvolutionOp origOp,
                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto logCb = [&](const llvm::formatv_object_base& msg) {
        std::ignore = matchFailed(_log, rewriter, origOp, "[{0}] {1}", getDebugName(), msg.str());
    };

    if (!VPU::NCEDepthConvolutionOp::isSupported(origOp, logCb)) {
        return mlir::failure();
    }

    Const::ContentAttr bias;
    if (origOp.bias() != nullptr) {
        auto biasConstOp = origOp.bias().getDefiningOp<Const::DeclareOp>();
        if (biasConstOp == nullptr) {
            return matchFailed(_log, rewriter, origOp, "[{0}] '{1}' at '{2}' has non constant biases", getDebugName(),
                               origOp->getName(), origOp->getLoc());
        }

        bias = biasConstOp.contentAttr();
    }

    const auto rawFilterShape = getIntArrayAttr(rewriter, getShape(origOp.filter()));
    const auto alignedFilter = VPU::alignDepthWiseWeightsTensor(rewriter, origOp.getLoc(), origOp.filter());

    const auto padAttr = VPU::getPaddingAttr(getContext(), PadInfo(origOp.pads_begin(), origOp.pads_end()));

    auto nceOp = rewriter.create<VPU::NCEDepthConvolutionOp>(origOp->getLoc(), origOp.getType(), origOp.input(),
                                                             alignedFilter, bias, origOp.stridesAttr(), padAttr,
                                                             origOp.post_opAttr(),
                                                             /*ppe=*/nullptr, rawFilterShape);

    rewriter.replaceOp(origOp, nceOp.output());
    return mlir::success();
}

//
// MaxPoolToNCE
//

class MaxPoolToNCE final : public mlir::OpRewritePattern<IE::MaxPoolOp> {
public:
    MaxPoolToNCE(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MaxPoolOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MaxPoolToNCE::matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto logCb = [&](const llvm::formatv_object_base& msg) {
        std::ignore = matchFailed(_log, rewriter, origOp, "[{0}] {1}", getDebugName(), msg.str());
    };

    if (!VPU::NCEMaxPoolOp::isSupported(origOp, logCb)) {
        return mlir::failure();
    }

    const auto padAttr = VPU::getPaddingAttr(getContext(), PadInfo(origOp.pads_begin(), origOp.pads_end()));

    auto nceOp = rewriter.create<VPU::NCEMaxPoolOp>(origOp->getLoc(), origOp.getType(), origOp.input(),
                                                    origOp.kernel_sizeAttr(), origOp.stridesAttr(), padAttr,
                                                    origOp.post_opAttr(),
                                                    /*ppe=*/nullptr);

    rewriter.replaceOp(origOp, nceOp.output());
    return mlir::success();
}

//
// EltwiseToNCE
//

template <class ConcreteOp>
class EltwiseToNCE final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    EltwiseToNCE<ConcreteOp>(mlir::MLIRContext* ctx, VPU::EltwiseType opType, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _opType(opType), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    VPU::EltwiseType _opType;
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult EltwiseToNCE<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    const auto logCb = [&](const llvm::formatv_object_base& msg) {
        std::ignore = matchFailed(_log, rewriter, origOp, "[{0}] {1}", this->getDebugName(), msg.str());
    };

    const bool allowDifferentScales = _opType == VPU::EltwiseType::MULTIPLY;
    const bool allowDifferentZp = true;

    if (!VPU::NCEEltwiseOp::isSupported(origOp, allowDifferentScales, allowDifferentZp, logCb)) {
        return mlir::failure();
    }

    const auto input1 = origOp.input1();
    const auto input2 = origOp.input1() == origOp.input2() ? origOp.input1() : origOp.input2();

    auto nceOp = rewriter.create<VPU::NCEEltwiseOp>(origOp->getLoc(), origOp.getType(), input1, input2,
                                                    VPU::EltwiseTypeAttr::get(this->getContext(), _opType),
                                                    origOp.post_opAttr(),
                                                    /*ppe=*/nullptr);

    rewriter.replaceOp(origOp, nceOp.output());
    return mlir::success();
}

//
// ConvertIEToVPUNCEPass
//

class ConvertIEToVPUNCEPass final : public ConvertIEToVPUNCEBase<ConvertIEToVPUNCEPass> {
public:
    explicit ConvertIEToVPUNCEPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void ConvertIEToVPUNCEPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::OwningRewritePatternList patterns(&ctx);
    patterns.add<ConvToNCE>(&ctx, _log);
    patterns.add<DepthConvToNCE>(&ctx, _log);
    patterns.add<MaxPoolToNCE>(&ctx, _log);
    patterns.add<EltwiseToNCE<IE::AddOp>>(&ctx, VPU::EltwiseType::ADD, _log);
    patterns.add<EltwiseToNCE<IE::MultiplyOp>>(&ctx, VPU::EltwiseType::MULTIPLY, _log);
    patterns.add<EltwiseToNCE<IE::SubtractOp>>(&ctx, VPU::EltwiseType::SUBTRACT, _log);
    patterns.add<EltwiseToNCE<IE::AndOp>>(&ctx, VPU::EltwiseType::AND, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertIEToVPUNCENCEPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertIEToVPUNCEPass(Logger log) {
    return std::make_unique<ConvertIEToVPUNCEPass>(log);
}
