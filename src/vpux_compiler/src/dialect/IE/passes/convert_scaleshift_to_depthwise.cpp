//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
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
    if (origOp.biases() != nullptr) {
        if (mlir::failed(IE::getConstParentOp(origOp.biases()))) {
            return matchFailed(_log, rewriter, origOp,
                               "Failed to convert ScaleShift to DW, since it has non constant biases");
        }
    }

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
    mlir::Value weights;

    auto createConstOp = [&](SmallVector<int64_t> shape, float16 value) -> Const::DeclareOp {
        const auto elemType = origOp.output().getType().cast<vpux::NDTypeInterface>().getElementType();
        auto shape1x1x1x1 = SmallVector<int64_t>(shape.size(), 1);
        const auto dataStorageType = mlir::RankedTensorType::get(shape1x1x1x1, elemType);
        const auto dataType = mlir::RankedTensorType::get(shape, elemType);

        const auto denseElementVal = mlir::DenseElementsAttr::get(dataStorageType, value);
        auto contentAttr = Const::ContentAttr::get(denseElementVal);

        for (auto dim : enumerate(shape)) {
            if (dim.value() > 1) {
                contentAttr = contentAttr.broadcast(Dim(dim.index()), dim.value());
            }
        }
        return rewriter.create<Const::DeclareOp>(origOp.getLoc(), dataType, contentAttr);
    };

    if (origOp.weights() != nullptr) {
        const auto multiply = origOp.weights();
        const auto weightShapeAttr = getIntArrayAttr(origOp.getContext(), weightShape);
        auto dwConvFilter = rewriter.create<IE::ReshapeOp>(origOp->getLoc(), multiply, nullptr, false, weightShapeAttr);
        weights = dwConvFilter.output();
    } else {
        auto dwConvFilter = createConstOp(weightShape, 1.0f);
        weights = dwConvFilter.output();
        auto inputFq = origOp.input().getDefiningOp<IE::FakeQuantizeOp>();
        if (inputFq != nullptr) {
            auto newFqOp = rewriter.create<IE::FakeQuantizeOp>(origOp->getLoc(), weights, weights, weights, weights,
                                                               weights, 255, inputFq.auto_broadcast());
            weights = newFqOp.output();
        }
    }

    rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(origOp, origOp.input(), weights, origOp.biases(), stridesAttr,
                                                        padBeginAttr, padEndAttr, dilationsAttr, groupAttr, nullptr);

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertScaleShiftToDWPass::safeRunOnFunc() {
    auto func = getFunction();

    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ScaleShiftOpConverter>(&ctx, _log);

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
