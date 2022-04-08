//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Pass/PassManager.h>

using namespace vpux;

namespace {

const int64_t UNEXPANDED_CHANNELS = 3;
const int64_t EXPANDED_CHANNELS = 4;

//
// EltwiseShapeCastRewriter
//

class EltwiseShapeCastRewriter final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    EltwiseShapeCastRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log) {
        setDebugName("EltwiseShapeCastRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ExpandOp mulOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool checkInput(mlir::Value tensor, int64_t origChannels) const;
    mlir::Value getExpandedInput(mlir::PatternRewriter& rewriter, IE::ShapeCastOp shapeCast, IE::ExpandOp expandOp,
                                 Shape expandedShape) const;

    Logger _log;
};

mlir::Value EltwiseShapeCastRewriter::getExpandedInput(mlir::PatternRewriter& rewriter, IE::ShapeCastOp shapeCast,
                                                       IE::ExpandOp expandOp, Shape expandedShape) const {
    auto tensor = shapeCast.source();
    if (auto permuteQuantize = mlir::dyn_cast_or_null<IE::PermuteQuantizeOp>(tensor.getDefiningOp())) {
        rewriter.setInsertionPointAfter(permuteQuantize);

        // TODO enable one permute quantize solution when S2D descriptor E#67286 is implemented
        auto permuteQuantizeClone =
                !permuteQuantize.output().hasOneUse() ? rewriter.clone(*permuteQuantize) : permuteQuantize;

        auto expand = rewriter.create<IE::ExpandOp>(expandOp.getLoc(), permuteQuantizeClone->getResult(0),
                                                    expandOp.pads_begin(), expandOp.pads_end());

        auto newShapeCast = rewriter.create<IE::ShapeCastOp>(
                expandOp.getLoc(), expand.output(), getIntArrayAttr(expandOp.getContext(), expandedShape.raw()));

        return newShapeCast;
    } else if (auto depthToSpace = mlir::dyn_cast_or_null<IE::DepthToSpaceOp>(tensor.getDefiningOp())) {
        rewriter.setInsertionPoint(expandOp);
        auto expand = rewriter.create<IE::ExpandOp>(expandOp.getLoc(), depthToSpace.output(), expandOp.pads_begin(),
                                                    expandOp.pads_end());

        return rewriter.create<IE::ShapeCastOp>(expandOp.getLoc(), expand.output(),
                                                getIntArrayAttr(expandOp.getContext(), expandedShape.raw()));
    }

    VPUX_THROW("Unsupported input {0}", *tensor.getDefiningOp());
}

// Check here that there are layers before ShapeCast
// which might be fused with Expand
// For now following cases are supported:
// 1.  PermuteQuantize -> Expand -> ShapeCase
//  in this case PermuteQuantize is fused with Expand later on
// 2. Slice -> DepthToSpace -> Expand -> ShapeCast
//  here Expand might be fused with DepthToSpace which has padded descriptor
// if we fuse these layers with Expand, ShapeCast uses already expanded tensor
// and reshape it back after Expand

bool EltwiseShapeCastRewriter::checkInput(mlir::Value tensor, int64_t origChannels) const {
    if (auto permuteQuantize = mlir::dyn_cast_or_null<IE::PermuteQuantizeOp>(tensor.getDefiningOp())) {
        // fusing with PermuteQuantize is availiable for VPUX37XX only
        const auto arch = VPU::getArch(permuteQuantize->getParentOfType<mlir::ModuleOp>());

        if (arch != VPU::ArchKind::VPUX37XX) {
            return false;
        }

        // PermuteQuantize is going to be fused with Expand only in that case
        if (getShape(permuteQuantize.output())[Dims4D::Act::C] != UNEXPANDED_CHANNELS &&
            origChannels != EXPANDED_CHANNELS) {
            return false;
        }

        return true;
    } else if (auto depthToSpace = mlir::dyn_cast_or_null<IE::DepthToSpaceOp>(tensor.getDefiningOp())) {
        if (!depthToSpace.output().hasOneUse()) {
            return false;
        }

        auto slice = depthToSpace.input().getDefiningOp<IE::SliceOp>();

        if (slice == nullptr || !slice.result().hasOneUse()) {
            return false;
        }

        if (getShape(slice.source())[Dims4D::Act::C] / (depthToSpace.block_size() * depthToSpace.block_size()) !=
            origChannels) {
            return false;
        }

        return true;
    }

    return false;
}

mlir::LogicalResult EltwiseShapeCastRewriter::matchAndRewrite(IE::ExpandOp expandOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("Got op {0} at {1}", expandOp->getName(), expandOp->getLoc());

    // F16 case would be beneficial to be executed as Concat + Const.
    auto inputTensor = expandOp.input();
    if (inputTensor.getType()
                .cast<vpux::NDTypeInterface>()
                .getElementType()
                .dyn_cast_or_null<mlir::quant::QuantizedType>() == nullptr) {
        return mlir::failure();
    }

    auto lastShapeCastOp = inputTensor.getDefiningOp<IE::ShapeCastOp>();

    if (lastShapeCastOp == nullptr) {
        return mlir::failure();
    }

    auto addOp = lastShapeCastOp.source().getDefiningOp<IE::AddOp>();

    if (addOp == nullptr) {
        return mlir::failure();
    }

    auto firstInput = addOp.input1().getDefiningOp<IE::ShapeCastOp>();
    auto secondInput = addOp.input2().getDefiningOp<IE::ShapeCastOp>();

    if (firstInput == nullptr || secondInput == nullptr) {
        return mlir::failure();
    }

    auto origChannels = getShape(expandOp)[Dims4D::Act::C];
    bool equalInputs = firstInput == secondInput;

    if (!checkInput(firstInput.source(), origChannels) || !checkInput(secondInput.source(), origChannels)) {
        return mlir::failure();
    }

    Shape newUnexpandedShape = getShape(firstInput.source()).toValues();
    newUnexpandedShape[Dims4D::Act::C] = origChannels;
    Shape newExpandedShape = getShape(firstInput.source()).toValues();
    newExpandedShape[Dims4D::Act::C] = getShape(addOp)[Dims4D::Act::C];

    auto expandedShape = vpux::IE::getShapeCastExpandedShape(addOp.getOperation(), newExpandedShape, newUnexpandedShape,
                                                             _log.nest());

    if (mlir::failed(expandedShape)) {
        return mlir::failure();
    }

    auto expandedInput1 = getExpandedInput(rewriter, firstInput, expandOp, expandedShape.getValue());
    auto expandedInput2 =
            !equalInputs ? getExpandedInput(rewriter, secondInput, expandOp, expandedShape.getValue()) : expandedInput1;
    rewriter.setInsertionPoint(expandOp);

    auto outputType = expandedInput1.getType().cast<vpux::NDTypeInterface>().changeElemType(
            addOp.output().getType().cast<vpux::NDTypeInterface>().getElementType());
    auto newEltwise = rewriter.create<IE::AddOp>(addOp.getLoc(), outputType, expandedInput1, expandedInput2,
                                                 addOp.auto_broadcast(), nullptr);

    rewriter.replaceOpWithNewOp<IE::ShapeCastOp>(expandOp, newEltwise.output(),
                                                 getIntArrayAttr(expandOp.getContext(), getShape(expandOp).raw()));

    return mlir::success();
}

//
// DepthToSpaceSliceRewriter
//

class DepthToSpaceSliceRewriter final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    DepthToSpaceSliceRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log) {
        setDebugName("DepthToSpaceSliceRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ExpandOp mulOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DepthToSpaceSliceRewriter::matchAndRewrite(IE::ExpandOp expandOp,
                                                               mlir::PatternRewriter& rewriter) const {
    _log.trace("Got op {0} at {1}", expandOp->getName(), expandOp->getLoc());

    auto depthToSpace = expandOp.input().getDefiningOp<IE::DepthToSpaceOp>();

    if (depthToSpace == nullptr) {
        return mlir::failure();
    }

    auto slice = depthToSpace.input().getDefiningOp<IE::SliceOp>();

    if (slice == nullptr) {
        return mlir::failure();
    }

    if (!mlir::isa_and_nonnull<IE::AlignedChannelsOpInterface>(slice.source().getDefiningOp())) {
        return mlir::failure();
    }

    auto blockSizeSquare = depthToSpace.block_size() * depthToSpace.block_size();
    if (getShape(slice.source())[Dims4D::Act::C] / blockSizeSquare != getShape(expandOp)[Dims4D::Act::C]) {
        return mlir::failure();
    }

    auto paddedIC = getShape(slice.source())[Dims4D::Act::C] - getShape(slice.result())[Dims4D::Act::C];
    auto paddedOC = paddedIC / blockSizeSquare;

    auto paddedChannels = IE::ChannelPadding::get(getIntAttr(expandOp.getContext(), paddedIC),
                                                  getIntAttr(expandOp.getContext(), paddedOC), expandOp.getContext());

    rewriter.replaceOpWithNewOp<IE::DepthToSpaceOp>(expandOp, slice.source(), depthToSpace.block_size(),
                                                    depthToSpace.mode(), paddedChannels);

    return mlir::success();
}

//
// PropagateExpandPass
//

class PropagateExpandPass final : public IE::PropagateExpandBase<PropagateExpandPass> {
public:
    explicit PropagateExpandPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void PropagateExpandPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<EltwiseShapeCastRewriter>(&ctx, _log);
    patterns.add<DepthToSpaceSliceRewriter>(&ctx, _log);
    // TODO E#67286 enable solution when S2D descriptor is ready

    auto func = getFunction();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createPropagateExpandPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateExpandPass(Logger log) {
    return std::make_unique<PropagateExpandPass>(log);
}
