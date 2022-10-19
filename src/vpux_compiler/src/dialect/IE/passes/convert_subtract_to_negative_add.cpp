//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// ConvertSubtractToNegativeAdd
//

class ConvertSubtractToNegativeAdd final : public mlir::OpRewritePattern<IE::SubtractOp> {
public:
    ConvertSubtractToNegativeAdd(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::SubtractOp>(ctx), _log(log) {
        setDebugName("ConvertSubtractToNegativeAdd");
    }

    mlir::LogicalResult matchAndRewrite(IE::SubtractOp subOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertSubtractToNegativeAdd::matchAndRewrite(IE::SubtractOp subOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    auto input1 = subOp.input1();
    auto input2 = subOp.input2();

    auto negativeOp = rewriter.create<IE::NegativeOp>(subOp.getLoc(), input2.getType(), input2);

    rewriter.replaceOpWithNewOp<IE::AddOp>(subOp, input1, negativeOp.output(), subOp.auto_broadcastAttr(),
                                           /*post_op=*/nullptr);
    return mlir::success();
}

//
// ConvertSubtractToNegativeAddPass
//

class ConvertSubtractToNegativeAddPass final :
        public IE::ConvertSubtractToNegativeAddBase<ConvertSubtractToNegativeAddPass> {
public:
    explicit ConvertSubtractToNegativeAddPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ConvertSubtractToNegativeAddPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto module = getOperation();
    const auto arch = VPU::getArch(module);
    if (arch == VPU::ArchKind::VPUX37XX) {
        _log.trace(
                "Negative is not enabled for VPUX37XX device. ConvertSubtractToNegativeAddPass is disabled. Got: {0}",
                arch);
        return;
    }

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<ConvertSubtractToNegativeAdd>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertSubtractToNegativeAddPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertSubtractToNegativeAddPass(Logger log) {
    return std::make_unique<ConvertSubtractToNegativeAddPass>(log);
}
