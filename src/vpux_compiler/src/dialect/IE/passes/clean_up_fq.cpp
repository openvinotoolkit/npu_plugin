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

class FakeQuantizeRewriter final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    FakeQuantizeRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp fqOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FakeQuantizeRewriter::matchAndRewrite(IE::FakeQuantizeOp operation,
                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("Got FakeQuantize operation '{0}}'", operation->getLoc());

    auto inputLow = operation.input_low().getDefiningOp<vpux::Const::DeclareOp>();
    VPUX_THROW_UNLESS(inputLow, "Got non-constant input low parameter");

    auto inputHigh = operation.input_high().getDefiningOp<vpux::Const::DeclareOp>();
    VPUX_THROW_UNLESS(inputHigh, "Got non-constant input high parameter");

    auto outputLow = operation.output_low().getDefiningOp<vpux::Const::DeclareOp>();
    VPUX_THROW_UNLESS(outputLow, "Got non-constant output low parameter");

    auto outputHigh = operation.output_high().getDefiningOp<vpux::Const::DeclareOp>();
    VPUX_THROW_UNLESS(outputHigh, "Got non-constant output high parameter");

    const auto inputLowAttribute = inputLow.contentAttr().fold();
    const auto inputHighAttribute = inputHigh.contentAttr().fold();
    const auto outputLowAttribute = outputLow.contentAttr().fold();
    const auto outputHighAttribute = outputHigh.contentAttr().fold();

    const auto isPlanar = [](const auto& shape) {
        return std::count_if(shape.begin(), shape.end(), [](auto dimension) {
                   return dimension != 1;
               }) <= 1;
    };

    VPUX_THROW_UNLESS(isPlanar(inputLowAttribute.getShape()),
                      "Got FakeQuantize {0} non-planar input low range of shape {1}", operation.getLoc(),
                      inputLowAttribute.getShape());

    VPUX_THROW_UNLESS(isPlanar(inputHighAttribute.getShape()),
                      "Got FakeQuantize {0} non-planar input high range of shape {1}", operation.getLoc(),
                      inputHighAttribute.getShape());

    VPUX_THROW_UNLESS(isPlanar(outputLowAttribute.getShape()),
                      "Got FakeQuantize {0} non-planar output low range of shape {1}", operation.getLoc(),
                      outputLowAttribute.getShape());

    VPUX_THROW_UNLESS(isPlanar(outputHighAttribute.getShape()),
                      "Got FakeQuantize {0} non-planar output high range of shape {1}", operation.getLoc(),
                      outputHighAttribute.getShape());

    const auto inputLowType = inputLowAttribute.getElementType();
    const auto inputHighType = inputHighAttribute.getElementType();
    const auto outputLowType = outputLowAttribute.getElementType();
    const auto outputHighType = outputHighAttribute.getElementType();

    VPUX_THROW_UNLESS(inputLowType == inputHighType && inputLowType == outputLowType && inputLowType == outputHighType,
                      "Got FakeQuantize {0} with input low ({1}), input high ({2}), output low ({3}) and output high "
                      "({4}) ranges of different types",
                      operation.getLoc(), inputHighType, outputLowType, outputHighType);

    VPUX_THROW_UNLESS(inputLowType.isF32() || inputLowType.isF16(),
                      "Got FakeQuantize {0} unsupported type for a range {1}: only {2} and {3} are supported",
                      operation.getLoc(), inputLowType, "F32", "F16");

    auto inputLowValues = to_small_vector(inputLowAttribute.getValues<double>());
    auto inputHighValues = to_small_vector(inputHighAttribute.getValues<double>());
    auto outputLowValues = to_small_vector(outputLowAttribute.getValues<double>());
    auto outputHighValues = to_small_vector(outputHighAttribute.getValues<double>());

    if (inputLowValues != outputLowValues || inputHighValues != outputHighValues) {
        return mlir::failure();
    }

    rewriter.replaceOp(operation, operation.input());

    return mlir::success();
}

//
// CleanUpFakeQuantizePass
//

class CleanUpFakeQuantizePass final : public IE::ConvertWeightsToU8Base<CleanUpFakeQuantizePass> {
public:
    explicit CleanUpFakeQuantizePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void CleanUpFakeQuantizePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    mlir::OwningRewritePatternList patterns(&ctx);
    patterns.insert<FakeQuantizeRewriter>(&ctx, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createCleanUpFakeQuantizePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createCleanUpFakeQuantizePass(Logger log) {
    return std::make_unique<CleanUpFakeQuantizePass>(log);
}
