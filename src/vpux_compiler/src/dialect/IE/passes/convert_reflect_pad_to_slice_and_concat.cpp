//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/adjust_layout_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// MatMulToConvAndPermuteCastPass
//

class ConvertReflectPadToSliceAndConcatPass final :
        public IE::ConvertReflectPadToSliceAndConcatBase<ConvertReflectPadToSliceAndConcatPass> {
public:
    explicit ConvertReflectPadToSliceAndConcatPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class PadOpConverter;

private:
    void safeRunOnFunc() final;
};

//
// ConvertPadOpConverter
//

class ConvertReflectPadToSliceAndConcatPass ::PadOpConverter final : public mlir::OpRewritePattern<IE::PadOp> {
public:
    PadOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::PadOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::PadOp PadOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertReflectPadToSliceAndConcatPass::PadOpConverter::matchAndRewrite(
        IE::PadOp padOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Find Reflect Pad {0} at {1}", padOp, padOp->getLoc());

    const auto loc = padOp.getLoc();
    const auto ctx = padOp.getContext();
    auto const padBegin = parseIntArrayAttr<int64_t>(padOp.pads_begin_attr().getValue());
    auto const padEnd = parseIntArrayAttr<int64_t>(padOp.pads_end_attr().getValue());

    SmallVector<mlir::Value> subSlices;

    auto createPad = [&](mlir::Value input, Dim axis, int64_t offset) {
        auto inputShape = getShape(input);
        auto offsets = SmallVector<int64_t>(inputShape.size(), 0);
        auto sizes = SmallVector<int64_t>(inputShape.begin(), inputShape.end());
        offsets[axis.ind()] = offset;
        sizes[axis.ind()] = 1;
        return rewriter.create<IE::SliceOp>(loc, input, getIntArrayAttr(ctx, offsets), getIntArrayAttr(ctx, sizes))
                .result();
    };

    auto creatAxisPad = [&](mlir::Value input, int64_t padBefore, int64_t padAfter, Dim padAxis) {
        if (padAfter == 0 && padBefore == 0) {
            return input;
        }
        subSlices.clear();
        for (auto i = padBefore; i > 0; --i) {
            auto upSlice = createPad(input, padAxis, i);
            subSlices.push_back(upSlice);
        }

        subSlices.push_back(input);

        for (auto i = 1; i <= padAfter; ++i) {
            auto offset = getShape(input)[padAxis] - 1 - i;
            auto upSlice = createPad(input, padAxis, offset);
            subSlices.push_back(upSlice);
        }

        return rewriter.create<IE::ConcatOp>(loc, subSlices, padAxis).output();
    };

    auto input = padOp.input();

    for (auto i = 0; i < checked_cast<int64_t>(getShape(input).size()); ++i) {
        input = creatAxisPad(input, padBegin[i], padEnd[i], Dim(i));
    }

    rewriter.replaceOp(padOp, input);

    _log.trace("Replease padOp with {0} ", input);

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertReflectPadToSliceAndConcatPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::ConversionTarget target(ctx);

    target.addDynamicallyLegalOp<IE::PadOp>([](IE::PadOp op) -> bool {
        return op.mode() != IE::PadMode::REFLECT;
    });
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::ConcatOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<PadOpConverter>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertReflectPadToSliceAndConcatPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertReflectPadToSliceAndConcatPass(Logger log) {
    return std::make_unique<ConvertReflectPadToSliceAndConcatPass>(log);
}
