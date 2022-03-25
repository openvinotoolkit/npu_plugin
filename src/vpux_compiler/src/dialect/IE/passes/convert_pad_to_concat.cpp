//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/pad_extract.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

#include <legacy/ngraph_ops/convolution_ie.hpp>

using namespace vpux;

namespace {

//
// ReplacePadWithConstAndConcat
//

class ReplacePadWithConstAndConcat final : public mlir::OpRewritePattern<IE::PadOp> {
public:
    ReplacePadWithConstAndConcat(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::PadOp>(ctx), _log(log) {
        setDebugName("ReplacePadWithConstAndConcat");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::PadOp origPadOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReplacePadWithConstAndConcat::matchAndRewrite(IE::PadOp origPadOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("Found IE::PadOp Operation '{0}'", origPadOp->getLoc());

    if (origPadOp.mode() != IE::PadMode::CONSTANT) {
        return mlir::failure();
    }

    auto padsBegin = vpux::IE::extractPads(origPadOp.pads_begin_attrAttr(), _log);
    if (mlir::failed(padsBegin)) {
        return mlir::failure();
    }

    auto padsEnd = vpux::IE::extractPads(origPadOp.pads_end_attrAttr(), _log);
    if (mlir::failed(padsEnd)) {
        return mlir::failure();
    }

    VPUX_THROW_UNLESS(origPadOp.pad_value_attr().hasValue(), "IE::PadOp has pad_value_attr() == nullptr {0}",
                      origPadOp->getLoc());
    const auto padValue = origPadOp.pad_value_attr().getValue().convertToDouble();

    const auto inputShape = origPadOp.input().getType().cast<vpux::NDTypeInterface>().getShape().raw();
    const auto outputShape = origPadOp.output().getType().cast<vpux::NDTypeInterface>().getShape().raw();

    const auto createConstOp = [&](SmallVector<mlir::Value>& values, size_t axis, ArrayRef<int64_t> padSize) {
        if (padSize[axis] == 0) {
            return;
        }

        auto constShape = SmallVector<int64_t>(inputShape.size(), 0);
        for (const auto& ind : irange(inputShape.size())) {
            constShape[ind] = ind < axis ? inputShape[ind] : outputShape[ind];
        }
        constShape[axis] = padSize[axis];

        const auto origElemType = origPadOp.input().getType().cast<vpux::NDTypeInterface>().getElementType();

        const auto padDataStorageType = mlir::RankedTensorType::get(constShape, mlir::Float32Type::get(getContext()));
        const auto padDataStorage = mlir::DenseElementsAttr::get(padDataStorageType, static_cast<float>(padValue));

        const auto padDataType = mlir::RankedTensorType::get(constShape, origElemType);
        const auto padDataAttr = Const::ContentAttr::get(padDataStorage).convertElemType(origElemType);

        auto constant = rewriter.create<Const::DeclareOp>(origPadOp->getLoc(), padDataType, padDataAttr);
        values.push_back(constant.output());
    };

    auto midInput = origPadOp.input();
    const auto padsBeginValue = padsBegin.getValue();
    const auto padsEndValue = padsEnd.getValue();
    VPUX_THROW_UNLESS(padsBeginValue.size() == inputShape.size() && padsEndValue.size() == inputShape.size(),
                      "`IE::PadOp` {0} shape size {1} mismatch with input size {2}", origPadOp.getLoc(),
                      padsBeginValue.size(), inputShape.size());

    for (const auto reversedAxis : irange(inputShape.size()) | reversed) {
        if (padsBeginValue[reversedAxis] == 0 && padsEndValue[reversedAxis] == 0) {
            continue;
        }

        SmallVector<mlir::Value> valueRange;

        _log.nest().trace("Insert ConstOp convert from padsBegin index: {0}", reversedAxis);
        createConstOp(valueRange, reversedAxis, padsBeginValue);

        valueRange.push_back(midInput);

        _log.nest().trace("Insert ConstOp convert from padsEnd index: {0}", reversedAxis);
        createConstOp(valueRange, reversedAxis, padsEndValue);

        auto concat = rewriter.create<IE::ConcatOp>(midInput.getLoc(), valueRange, reversedAxis);
        _log.nest().trace("Insert ConcatOp {0}", concat.getLoc());
        midInput = concat.output();
    }

    rewriter.replaceOp(origPadOp, midInput);

    return mlir::success();
}

//
// ConvertPadToConcat
//

class ConvertPadToConcatPass final : public IE::ConvertPadToConcatBase<ConvertPadToConcatPass> {
public:
    explicit ConvertPadToConcatPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertPadToConcatPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<ReplacePadWithConstAndConcat>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createSupportFusePadOpsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertPadToConcatPass(Logger log) {
    return std::make_unique<ConvertPadToConcatPass>(log);
}
