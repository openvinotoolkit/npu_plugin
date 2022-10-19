//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// ConvertScaleShiftToDepthwisePass
//

class ConvertScaleShiftToDWPass final : public IE::ConvertScaleShiftToDWBase<ConvertScaleShiftToDWPass> {
public:
    explicit ConvertScaleShiftToDWPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class ScaleShiftOpConverter;

private:
    void safeRunOnFunc() final;
};

//
// ScaleShiftOpConverter
//

class ConvertScaleShiftToDWPass::ScaleShiftOpConverter final : public mlir::OpRewritePattern<IE::ScaleShiftOp> {
public:
    ScaleShiftOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ScaleShiftOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ScaleShiftOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertScaleShiftToDWPass::ScaleShiftOpConverter::matchAndRewrite(
        IE::ScaleShiftOp origOp, mlir::PatternRewriter& rewriter) const {
    if (origOp.weights() == nullptr)
        return matchFailed(_log, rewriter, origOp, "Failed to convert ScaleShift to DW, since there are no weights");

    // If ScaleShift after input and first conv can use C-major, It is better not convert to DWConv.
    // More general, if sub-graph like: input -> UPAs -> ScaleShift -> Conv. It is better not convert to DWConv.
    // If layer before ScaleShift is NCE op or with NHWC layout. It should convert to DWConv.
    auto onlySupportNHWCLayout = [&](mlir::Operation* op) -> bool {
        if (auto iface = mlir::dyn_cast<IE::LayoutInfoOpInterface>(op)) {
            auto orderInfo = iface.getLayoutInfo();
            iface.inferLayoutInfo(orderInfo);
            return orderInfo.hasChanges();
        }
        return false;
    };

    auto prevOp = origOp.input().getDefiningOp();
    if (prevOp == nullptr || !onlySupportNHWCLayout(prevOp)) {
        for (auto user : origOp.getResult().getUsers()) {
            auto convOp = mlir::dyn_cast<IE::ConvolutionOp>(user);
            if (convOp != nullptr &&
                VPU::NCEInvariant::isChannelMajorCompatible(VPU::getArch(convOp),
                                                            convOp.input().getType().cast<vpux::NDTypeInterface>())) {
                return mlir::failure();
            }
        }
    }

    const SmallVector<int32_t> strides = {1, 1};
    const SmallVector<int32_t> padBegin = {0, 0};
    const SmallVector<int32_t> padEnd = {0, 0};
    const SmallVector<int32_t> dilations = {1, 1};

    const int64_t kernelSize = 1;

    auto dilationsAttr = getIntArrayAttr(origOp.getContext(), dilations);
    auto stridesAttr = getIntArrayAttr(origOp.getContext(), strides);
    auto padBeginAttr = getIntArrayAttr(origOp.getContext(), padBegin);
    auto padEndAttr = getIntArrayAttr(origOp.getContext(), padEnd);

    auto outShape = getShape(origOp.output()).toValues();
    auto groupAttr = getIntAttr(origOp.getContext(), outShape[Dims4D::Act::C]);
    const SmallVector<int64_t> weightShape = {outShape[Dims4D::Act::C], 1, kernelSize, kernelSize};

    const auto multiply = origOp.weights();
    const auto weightShapeAttr = getIntArrayAttr(origOp.getContext(), weightShape);
    auto dwConvFilter = rewriter.create<IE::ReshapeOp>(origOp->getLoc(), multiply, nullptr, false, weightShapeAttr);

    rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(origOp, origOp.input(), dwConvFilter.output(), origOp.biases(),
                                                        stridesAttr, padBeginAttr, padEndAttr, dilationsAttr, groupAttr,
                                                        nullptr);

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertScaleShiftToDWPass::safeRunOnFunc() {
    auto func = getFunction();

    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<ScaleShiftOpConverter>(&ctx, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertScaleShiftToDWPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertScaleShiftToDWPass(Logger log) {
    return std::make_unique<ConvertScaleShiftToDWPass>(log);
}
