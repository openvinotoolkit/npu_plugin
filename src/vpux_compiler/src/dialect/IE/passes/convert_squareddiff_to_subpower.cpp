//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <ngraph/coordinate_diff.hpp>

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// ConvertSquaredDiffToSubAndPower
//

class ConvertSquaredDiffToSubAndPowerPass final :
        public IE::ConvertSquaredDiffToSubAndPowerBase<ConvertSquaredDiffToSubAndPowerPass> {
public:
    explicit ConvertSquaredDiffToSubAndPowerPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class SquaredDifferenceRewriter;

private:
    void safeRunOnFunc() final;
};

//
// SquaredDifferenceRewriter
//

class ConvertSquaredDiffToSubAndPowerPass::SquaredDifferenceRewriter final :
        public mlir::OpRewritePattern<IE::SquaredDifferenceOp> {
public:
    SquaredDifferenceRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::SquaredDifferenceOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SquaredDifferenceOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertSquaredDiffToSubAndPowerPass::SquaredDifferenceRewriter::matchAndRewrite(
        IE::SquaredDifferenceOp sqdOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got SquaredDifferenceOp for conversion to Subtract and Power - '{0}'", sqdOp->getLoc());

    auto subtract = rewriter.create<IE::SubtractOp>(sqdOp.getLoc(), sqdOp.input1(), sqdOp.input2(),
                                                    sqdOp.auto_broadcast(), nullptr);

    const SmallVector<ngraph::float16> vals = {2.f};
    const SmallVector<int64_t> shape(subtract.getType().getRank(), 1);
    const auto baseType = mlir::RankedTensorType::get(makeArrayRef(shape), mlir::Float16Type::get(getContext()));
    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));
    auto exponent = rewriter.create<Const::DeclareOp>(sqdOp.getLoc(), baseType, Const::ContentAttr::get(baseAttr));

    rewriter.replaceOpWithNewOp<IE::PowerOp>(sqdOp, subtract, exponent, sqdOp.auto_broadcast());

    return mlir::success();
}

void ConvertSquaredDiffToSubAndPowerPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<IE::SquaredDifferenceOp>();
    target.addLegalOp<IE::PowerOp>();
    target.addLegalOp<IE::SubtractOp>();
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<SquaredDifferenceRewriter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// ConvertSquaredDiffToSubAndPowerPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertSquaredDiffToSubAndPowerPass(Logger log) {
    return std::make_unique<ConvertSquaredDiffToSubAndPowerPass>(log);
}
