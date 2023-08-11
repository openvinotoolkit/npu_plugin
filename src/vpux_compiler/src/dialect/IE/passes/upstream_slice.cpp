//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <ngraph/slice_plan.hpp>

using namespace vpux;

namespace {

//
// UpstreamSlicePass
//

class UpstreamSlicePass final : public IE::UpstreamSliceBase<UpstreamSlicePass> {
public:
    explicit UpstreamSlicePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class GenericSliceUpstreaming;

private:
    void safeRunOnFunc() final;
};

//
// GenericSliceUpstreaming
//

class UpstreamSlicePass::GenericSliceUpstreaming final : public mlir::OpInterfaceRewritePattern<IE::LayerOpInterface> {
public:
    GenericSliceUpstreaming(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<IE::LayerOpInterface>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::LayerOpInterface origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool isUpstreamPossible(IE::LayerOpInterface sliceOp, mlir::Value tensor) {
    if (tensor.isa<mlir::BlockArgument>())
        return false;
    mlir::Operation* parentOp = tensor.getDefiningOp();
    // Unary and eltwise ops are primary candidates for upstreaming slice ops.
    // Later on, implementation could handle also Conv, Pool upstreaming
    if (parentOp == nullptr || !parentOp->hasTrait<IE::EltwiseOp>())
        return false;

    if (parentOp->getNumResults() > 1)
        return false;

    // Strided slice is known to be done as UPA task.
    // It does not support datatypes generically
    // so we can't afford changing datatype of this operation.
    if (mlir::isa<IE::StridedSliceOp>(sliceOp)) {
        auto sliceOpElementType = tensor.getType().cast<vpux::NDTypeInterface>().getElementType();
        auto parentOpElementType = parentOp->getOperand(0).getType().cast<vpux::NDTypeInterface>().getElementType();
        if (sliceOpElementType != parentOpElementType)
            return false;
    }

    // Another restriction due to limited implementation.
    // Upstreaming through the graph using op interfaces it's hard
    // enough to reason about which operands are path of the activation
    // path and which are parameters.
    // An interface that makes this dinstinction easy would be of help.
    const auto operands = parentOp->getOperands();
    if (operands.size() > 1 &&
        std::adjacent_find(operands.begin(), operands.end(), [](mlir::Value val1, mlir::Value val2) {
            return getShape(val1) != getShape(val2);
        }) != operands.end())
        return false;

    const auto inputShape = getShape(sliceOp.getInputs()[0]);
    const auto outputShape = getShape(sliceOp.getOutputs()[0]);

    // Can't reason yet with generic shape dimensions
    if (inputShape.size() != 4 || outputShape.size() != 4)
        return false;

    // Can't handle yet upstreaming and adapting channelwise parameters
    if (const auto quantAxis = IE::getQuantAxisIndex(parentOp)) {
        if (inputShape[Dim(quantAxis.getValue())] != outputShape[Dim(quantAxis.getValue())]) {
            return false;
        }
    }

    return true;
}

mlir::LogicalResult UpstreamSlicePass::GenericSliceUpstreaming::matchAndRewrite(IE::LayerOpInterface origOp,
                                                                                mlir::PatternRewriter& rewriter) const {
    if (!mlir::isa<IE::SliceOp, IE::StridedSliceOp>(origOp)) {
        return mlir::failure();
    }

    auto origInput = origOp.getInputs()[0];
    if (!origInput.hasOneUse()) {
        return mlir::failure();
    }

    if (!isUpstreamPossible(origOp, origInput)) {
        return mlir::failure();
    }

    mlir::InferTypeOpInterface parentOp = origInput.getDefiningOp();
    rewriter.setInsertionPoint(parentOp);
    auto opOperands = parentOp->getOpOperands();
    if (std::adjacent_find(opOperands.begin(), opOperands.end(), [](mlir::OpOperand& val1, mlir::OpOperand& val2) {
            return val1.get() != val2.get();
        }) == opOperands.end()) {
        auto newSlice = mlir::cast<mlir::InferTypeOpInterface>(rewriter.clone(*origOp));
        newSlice->setOperand(0, opOperands[0].get());
        inferReturnTypes(newSlice, InferShapedTypeMode::ALL);
        for (auto& operand : opOperands) {
            operand.set(newSlice->getResult(0));
        }
    } else {
        for (auto& operand : opOperands) {
            auto newSlice = mlir::cast<mlir::InferTypeOpInterface>(rewriter.clone(*origOp));
            newSlice->setOperand(0, operand.get());
            inferReturnTypes(newSlice, InferShapedTypeMode::ALL);
            operand.set(newSlice->getResult(0));
        }
    }

    VPUX_THROW_UNLESS(parentOp->getResults().size() == 1, "Don't support backprop for multiple outputs yet '{0}'",
                      parentOp);
    inferReturnTypes(parentOp, InferShapedTypeMode::SHAPE);

    rewriter.replaceOp(origOp, parentOp->getResults());
    return mlir::success();
}

//
// safeRunOnFunc
//

void UpstreamSlicePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GenericSliceUpstreaming>(&ctx, _log);

    IE::SliceOp::getCanonicalizationPatterns(patterns, &ctx);
    IE::StridedSliceOp::getCanonicalizationPatterns(patterns, &ctx);

    auto func = getOperation();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUpstreamSlicePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createUpstreamSlicePass(Logger log) {
    return std::make_unique<UpstreamSlicePass>(log);
}
