//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/TypeSwitch.h>
#include <vpux/compiler/conversion.hpp>

using namespace vpux;

namespace {

//
// SwapTransposeWithConvert
//

class SwapTransposeWithConvert final : public IE::SwapTransposeWithConvertBase<SwapTransposeWithConvert> {
public:
    explicit SwapTransposeWithConvert(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    class TransposeOpConverter;

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

//
// TransposeOpConverter
//

class SwapTransposeWithConvert::TransposeOpConverter final : public mlir::OpRewritePattern<IE::TransposeOp> {
public:
    TransposeOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::TransposeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TransposeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SwapTransposeWithConvert::TransposeOpConverter::matchAndRewrite(
        IE::TransposeOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto transposeIn = origOp.input();
    if (auto origConvertOp = transposeIn.getDefiningOp<IE::ConvertOp>()) {
        auto transposeOp = rewriter.create<IE::TransposeOp>(origOp->getLoc(), origConvertOp.input(), nullptr,
                                                            origOp.order_valueAttr());

        rewriter.replaceOpWithNewOp<IE::ConvertOp>(origOp, transposeOp.output(), origConvertOp.dstElemType());
    }

    return mlir::success();
}

void SwapTransposeWithConvert::safeRunOnFunc() {
    auto func = getFunction();

    auto& ctx = getContext();

    const auto isLegalOp = [](IE::TransposeOp op) -> bool {
        const auto transposeIn = op.input();
        // For OV 2.0 API U8 we can have:
        // NetworkInput (NCHW) -> Convert -> Transpose-> FQ . Because of this lately after propagate quantize
        // dequantize pass and fuse convert with quantize pass, will be needed to propagate the quantizeCast
        // quantParams to Transpose. We want to avoid this. Also in the end this Transpose will be done as PermuteCast.
        if (auto maybeConvertOp = transposeIn.getDefiningOp<IE::ConvertOp>()) {
            return !maybeConvertOp.input().isa<mlir::BlockArgument>();
        }

        return true;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::TransposeOp>(isLegalOp);
    target.addLegalOp<IE::ConvertOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SwapTransposeWithConvert::TransposeOpConverter>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createSwapTransposeWithConvertPass(Logger log) {
    return std::make_unique<SwapTransposeWithConvert>(log);
}
