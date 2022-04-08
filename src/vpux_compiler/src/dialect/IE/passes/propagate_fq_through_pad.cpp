//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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
// PropagateFqThroughPad
//

class PropagateFqThroughPad final : public IE::PropagateFqThroughPadBase<PropagateFqThroughPad> {
public:
    explicit PropagateFqThroughPad(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    class ConcatOpConverter;

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

//
// ConcatOpConverter
//

class PropagateFqThroughPad::ConcatOpConverter final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    ConcatOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(log) {
        setDebugName("ConcatOpConverter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origConcatOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool isPerTensorFq(const mlir::Value in) {
    const auto maybeFqOp = in.getDefiningOp<IE::FakeQuantizeOp>();
    if (maybeFqOp == nullptr) {
        return false;
    }

    const auto axis = IE::getFQAxisIndex(maybeFqOp);
    return !axis.hasValue();
};

mlir::LogicalResult PropagateFqThroughPad::ConcatOpConverter::matchAndRewrite(IE::ConcatOp origConcatOp,
                                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}]: rewriting {1}", getDebugName(), origConcatOp->getLoc());
    const auto concatInputList = origConcatOp.inputs();
    // Find FQ operation
    auto fqOutputIter = std::find_if(concatInputList.begin(), concatInputList.end(), isPerTensorFq);
    auto fqOp = (*fqOutputIter).getDefiningOp<IE::FakeQuantizeOp>();

    SmallVector<mlir::Value> newConcatInputs;
    for (const auto& concatInput : concatInputList) {
        if (isPerTensorFq(concatInput)) {
            newConcatInputs.push_back(concatInput);
        } else {
            auto newFqOp = rewriter.create<IE::FakeQuantizeOp>(origConcatOp->getLoc(), concatInput, fqOp.input_low(),
                                                               fqOp.input_high(), fqOp.output_low(), fqOp.output_high(),
                                                               fqOp.levels(), fqOp.auto_broadcast());

            _log.nest().trace("Inserted new FQ: {0}", newFqOp);
            newConcatInputs.push_back(newFqOp.output());
        }
    }

    auto newConcatOp = rewriter.create<IE::ConcatOp>(origConcatOp->getLoc(), newConcatInputs,
                                                     origConcatOp.per_axisAttr(), origConcatOp.static_offsetsAttr());

    rewriter.replaceOpWithNewOp<IE::FakeQuantizeOp>(origConcatOp, newConcatOp.output(), fqOp.input_low(),
                                                    fqOp.input_high(), fqOp.output_low(), fqOp.output_high(),
                                                    fqOp.levels(), fqOp.auto_broadcast());

    _log.nest().trace("ConcatOp conversion done");

    return mlir::success();
}

void PropagateFqThroughPad::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto isLegalConcat = [](IE::ConcatOp concatOp) -> bool {
        const auto concatInputList = concatOp.inputs();
        // Either all concat inputs must be quantized, or none of them.
        if (std::all_of(concatInputList.begin(), concatInputList.end(), isPerTensorFq)) {
            return true;
        }

        if (std::none_of(concatInputList.begin(), concatInputList.end(), isPerTensorFq)) {
            return true;
        }

        if (std::count_if(concatInputList.begin(), concatInputList.end(), isPerTensorFq) == 1) {
            return false;
        }

        return true;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::ConcatOp>(isLegalConcat);
    target.addLegalOp<IE::FakeQuantizeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<PropagateFqThroughPad::ConcatOpConverter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateFqThroughPadPass(Logger log) {
    return std::make_unique<PropagateFqThroughPad>(log);
}
