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

#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/TypeSwitch.h>
#include <vpux/compiler/conversion.hpp>

using namespace vpux;

namespace {

//
// SwapConcatWithEltwise
//

class SwapConcatWithEltwise final : public IE::SwapConcatWithEltwiseBase<SwapConcatWithEltwise> {
public:
    explicit SwapConcatWithEltwise(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    class FakeQuantOpConverter;

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

//
// FakeQuantOpConverter
//

class SwapConcatWithEltwise::FakeQuantOpConverter final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    FakeQuantOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool isLegalFQ(IE::FakeQuantizeOp op) {
    const auto axis = IE::getFQAxisIndex(op);
    if (axis.hasValue()) {
        return true;
    }

    auto fqInput = op.input();
    auto maybeEltwiseOp = fqInput.getDefiningOp();
    if (maybeEltwiseOp == nullptr || !maybeEltwiseOp->hasTrait<IE::EltwiseOp>()) {
        return true;
    }

    auto eltwiseInput = maybeEltwiseOp->getOperand(0);
    auto maybeConcatOp = eltwiseInput.getDefiningOp<IE::ConcatOp>();
    if (maybeConcatOp == nullptr) {
        return true;
    }

    if (!maybeConcatOp->hasOneUse()) {
        return true;
    }

    const auto concatInputList = maybeConcatOp.inputs();
    for (const auto& concatInput : concatInputList) {
        if (mlir::isa_and_nonnull<IE::ConvolutionOp>(concatInput.getDefiningOp())) {
            return false;
        }
    }

    return true;
}

mlir::LogicalResult SwapConcatWithEltwise::FakeQuantOpConverter::matchAndRewrite(
        IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const {
    if (isLegalFQ(origOp)) {
        return mlir::failure();
    }
    auto origActOp = origOp.input().getDefiningOp();
    auto origConcatOp = origActOp->getOperand(0).getDefiningOp<IE::ConcatOp>();
    const auto concatInputList = origConcatOp.inputs();
    SmallVector<mlir::Value> newConcatInputs;
    for (const auto& concatInput : concatInputList) {
        auto newActOp = rewriter.clone(*origActOp);
        newActOp->setOperand(0, concatInput);
        newActOp->getOpResult(0).setType(concatInput.getType());
        auto newFqOp = rewriter.create<IE::FakeQuantizeOp>(
                origOp->getLoc(), newActOp->getResult(0), origOp.input_low(), origOp.input_high(), origOp.output_low(),
                origOp.output_high(), origOp.levels(), origOp.auto_broadcast());
        newConcatInputs.push_back(newFqOp.output());
    }

    rewriter.replaceOpWithNewOp<IE::ConcatOp>(origOp, newConcatInputs, origConcatOp.per_axisAttr(),
                                              origConcatOp.static_offsetsAttr());

    return mlir::success();
}

void SwapConcatWithEltwise::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<SwapConcatWithEltwise::FakeQuantOpConverter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createSwapConcatWithEltwisePass(Logger log) {
    return std::make_unique<SwapConcatWithEltwise>(log);
}
