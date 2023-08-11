//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/IE/loop.hpp"
#include "vpux/utils/core/enums.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

// ======================================================================================
// FusePermuteQuantizeExpandTogetherRewrite
//   FusePermuteQuantizeExpandTogetherRewrite -> [Expand -> Reorder -> Add -> QuantizeCastOp] -> [PermuteQuantizeExpand
//   -> QuantizeCastOp]

class FusePermuteQuantizeExpandTogetherRewrite final : public mlir::OpRewritePattern<IE::QuantizeCastOp> {
public:
    FusePermuteQuantizeExpandTogetherRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::QuantizeCastOp>(ctx, benefitHigh), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeCastOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FusePermuteQuantizeExpandTogetherRewrite::matchAndRewrite(IE::QuantizeCastOp origOp,
                                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    // check patern
    auto opAdd = origOp.input().getDefiningOp<IE::AddOp>();
    if (opAdd == nullptr) {
        return mlir::failure();
    }
    auto opReorder = opAdd.input1().getDefiningOp<IE::ReorderOp>();
    if (opReorder == nullptr) {
        return mlir::failure();
    }

    auto opExpand = opReorder.input().getDefiningOp<IE::ExpandOp>();
    if (opExpand == nullptr) {
        return mlir::failure();
    }

    // check just 1 child for linked patern
    for (auto user : llvm::make_early_inc_range(opReorder.getResult().getUsers())) {
        if (user != opAdd) {
            return mlir::failure();
        }
    }
    if (!opAdd.getResult().hasOneUse()) {
        return mlir::failure();
    }

    // check if Add is used for quantize
    if (opAdd.input1() != opAdd.input2()) {
        return mlir::failure();
    }

    if (!opExpand.getResult().hasOneUse()) {
        return mlir::failure();
    }

    const auto inType = opAdd.input1().getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto outType = opAdd.output().getType().cast<vpux::NDTypeInterface>().getElementType();
    if (!(inType.isF16() && outType.isa<mlir::quant::QuantizedType>())) {
        return mlir::failure();
    }

    // check uniform quantize
    const auto qType = outType.cast<mlir::quant::QuantizedType>();
    if (!qType.isa<mlir::quant::UniformQuantizedType>()) {
        return mlir::failure();
    }

    // check if reorder will not be removed
    auto inOrder = DimsOrder::fromValue(opReorder.input());
    auto outOrder = DimsOrder::fromValue(opReorder.output());
    if (inOrder == outOrder) {
        return mlir::failure();
    }
    // check and add pass for verified orders and scenarios
    if (!((inOrder == DimsOrder::NCHW) && (outOrder == DimsOrder::NHWC))) {
        return mlir::failure();
    }
    // allow expand just on C dim, that will be last after reorder.
    const auto iExpType = opExpand.input().getType().cast<vpux::NDTypeInterface>();
    const auto oExpType = opExpand.output().getType().cast<vpux::NDTypeInterface>();
    if (iExpType.getShape()[Dims4D::Act::N] != oExpType.getShape()[Dims4D::Act::N]) {
        return mlir::failure();
    }
    if (iExpType.getShape()[Dims4D::Act::W] != oExpType.getShape()[Dims4D::Act::W]) {
        return mlir::failure();
    }
    if (iExpType.getShape()[Dims4D::Act::H] != oExpType.getShape()[Dims4D::Act::H]) {
        return mlir::failure();
    }

    // experiments show that shave is far more performant when C == 1, C == 3 or C == 4 than DMA-MemPermute
    if ((iExpType.getShape()[Dims4D::Act::C] != 3) && (iExpType.getShape()[Dims4D::Act::C] != 1) &&
        (iExpType.getShape()[Dims4D::Act::C] != 4)) {
        return mlir::failure();
    }
    if (iExpType.getShape()[Dims4D::Act::N] != 1) {
        return mlir::failure();
    }

    // input can be fp32, so fuse and convertOp if it is possible.
    auto paternInput = opExpand.input();
    auto opReshape = opExpand.input().getDefiningOp<IE::AffineReshapeOp>();
    auto opConvert = opExpand.input().getDefiningOp<IE::ConvertOp>();
    // first patern when no reshape involve, just fuse ConvertOp
    if (opConvert != nullptr) {
        if (opConvert.getResult().hasOneUse() &&
            opConvert.input().getType().cast<vpux::NDTypeInterface>().getElementType().isF32()) {
            paternInput = opConvert.input();
        }
    }
    // pattern 2 when we have Convert->Reshape>PermuteQuantizePattern
    // in this case Reshape will be move before ConvertOp
    if (opReshape != nullptr) {
        opConvert = opReshape.input().getDefiningOp<IE::ConvertOp>();
        if (opConvert != nullptr) {
            if (opReshape.getResult().hasOneUse() && opConvert.getResult().hasOneUse() &&
                opConvert.input().getType().cast<vpux::NDTypeInterface>().getElementType().isF32()) {
                const auto newReshapeOpLoc = appendLoc(origOp->getLoc(), "AffineReshape");
                auto newReshapeOp = rewriter.create<IE::AffineReshapeOp>(
                        newReshapeOpLoc, opConvert.input(), opReshape.dim_mappingAttr(), opReshape.shape_valueAttr());
                paternInput = newReshapeOp.output();
            }
        }
    }

    auto memPermAttr = mlir::AffineMapAttr::get(getPermutationFromOrders(inOrder, outOrder, origOp->getContext()));
    // Get target quant type from IE.Add, not from IE.QuantizeCast.
    // QuantizeToAddRewriter multiplies output scale by 2. It is necessary to cancel out this factor.
    const auto permQuantOutType = rescaleUniformQuantizedType(opAdd.output().getType(), 0.5);
    const auto permQuantElemType = permQuantOutType.cast<vpux::NDTypeInterface>().getElementType();
    const auto dstElemTypeAttr = mlir::TypeAttr::get(permQuantElemType);
    const auto permQuantLoc = appendLoc(origOp->getLoc(), "PermuteQuantizeExpand");
    auto permuteQuantizeOp = rewriter.create<IE::PermuteQuantizeOp>(
            permQuantLoc, permQuantOutType, paternInput, opReorder.dstOrderAttr(), memPermAttr, dstElemTypeAttr,
            opExpand.pads_beginAttr(), opExpand.pads_endAttr());

    // Preserve IE.QuantizeCast in the graph.
    auto quantCast =
            rewriter.create<IE::QuantizeCastOp>(origOp->getLoc(), permuteQuantizeOp.output(), origOp.dstElemTypeAttr());

    rewriter.replaceOp(origOp, quantCast.output());

    return mlir::success();
}

// ======================================================================================
// FuseExpandIntoPermuteQuantizeRewrite
//   FuseExpandIntoPermuteQuantizeRewrite -> [PermuteQuantize -> Expand ] -> PermuteQuantizeExpand

class FuseExpandIntoPermuteQuantizeRewrite final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    FuseExpandIntoPermuteQuantizeRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ExpandOp>(ctx, benefitHigh), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseExpandIntoPermuteQuantizeRewrite::matchAndRewrite(IE::ExpandOp origOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    // check patern
    auto opPermuteQuantize = origOp.input().getDefiningOp<IE::PermuteQuantizeOp>();
    if (opPermuteQuantize == nullptr) {
        return mlir::failure();
    }
    if (!opPermuteQuantize.getResult().hasOneUse()) {
        return mlir::failure();
    }

    // alow expand just on C dim, that will be last after reoreder.
    const auto iExpType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto oExpType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    if (!((4 == iExpType.getRank()) && (4 == oExpType.getRank()))) {
        return mlir::failure();
    }
    if (iExpType.getShape()[Dims4D::Act::N] != oExpType.getShape()[Dims4D::Act::N]) {
        return mlir::failure();
    }
    if (iExpType.getShape()[Dims4D::Act::W] != oExpType.getShape()[Dims4D::Act::W]) {
        return mlir::failure();
    }
    if (iExpType.getShape()[Dims4D::Act::H] != oExpType.getShape()[Dims4D::Act::H]) {
        return mlir::failure();
    }

    auto permuteQuantizeOp = rewriter.create<IE::PermuteQuantizeOp>(
            origOp->getLoc(), opPermuteQuantize.input(), opPermuteQuantize.dst_orderAttr(),
            opPermuteQuantize.mem_permAttr(), opPermuteQuantize.dstElemTypeAttr(), origOp.pads_beginAttr(),
            origOp.pads_endAttr());
    rewriter.replaceOp(origOp, permuteQuantizeOp.output());

    return mlir::success();
}

// ======================================================================================
// FuseQuantizeCastExpandIntoPermuteQuantizeQuantizeCastRewrite
//   FuseQuantizeCastExpandIntoPermuteQuantizeQuantizeCastRewrite -> [PermuteQuantize-> QuantizeCast -> Expand] ->
//   PermuteQuantizeExpand->QuantizeCast

class FuseQuantizeCastExpandIntoPermuteQuantizeQuantizeCastRewrite final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    FuseQuantizeCastExpandIntoPermuteQuantizeQuantizeCastRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ExpandOp>(ctx, benefitHigh), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseQuantizeCastExpandIntoPermuteQuantizeQuantizeCastRewrite::matchAndRewrite(
        IE::ExpandOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    // check patern
    auto opQuantizeCast = origOp.input().getDefiningOp<IE::QuantizeCastOp>();
    if (opQuantizeCast == nullptr) {
        return mlir::failure();
    }
    auto opPermuteQuantize = opQuantizeCast.input().getDefiningOp<IE::PermuteQuantizeOp>();
    if (opPermuteQuantize == nullptr) {
        return mlir::failure();
    }
    if (!opQuantizeCast.getResult().hasOneUse()) {
        return mlir::failure();
    }
    if (!opPermuteQuantize.getResult().hasOneUse()) {
        return mlir::failure();
    }

    // alow expand just on C dim, that will be last after reoreder.
    const auto iExpType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto oExpType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    if (!((4 == iExpType.getRank()) && (4 == oExpType.getRank()))) {
        return mlir::failure();
    }
    if (iExpType.getShape()[Dims4D::Act::N] != oExpType.getShape()[Dims4D::Act::N]) {
        return mlir::failure();
    }
    if (iExpType.getShape()[Dims4D::Act::W] != oExpType.getShape()[Dims4D::Act::W]) {
        return mlir::failure();
    }
    if (iExpType.getShape()[Dims4D::Act::H] != oExpType.getShape()[Dims4D::Act::H]) {
        return mlir::failure();
    }

    auto permuteQuantizeOp = rewriter.create<IE::PermuteQuantizeOp>(
            origOp->getLoc(), opPermuteQuantize.input(), opPermuteQuantize.dst_orderAttr(),
            opPermuteQuantize.mem_permAttr(), opPermuteQuantize.dstElemTypeAttr(), origOp.pads_beginAttr(),
            origOp.pads_endAttr());
    auto quantizeCastOp = rewriter.create<IE::QuantizeCastOp>(origOp.getLoc(), permuteQuantizeOp.getResult(),
                                                              opQuantizeCast.dstElemTypeAttr());

    rewriter.replaceOp(origOp, quantizeCastOp.output());

    return mlir::success();
}

//
// FusePermuteQuantizeExpandPass
//

class FusePermuteQuantizeExpandPass final : public IE::FusePermuteQuantizeExpandBase<FusePermuteQuantizeExpandPass> {
public:
    explicit FusePermuteQuantizeExpandPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void FusePermuteQuantizeExpandPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FusePermuteQuantizeExpandTogetherRewrite>(&ctx, _log);
    patterns.add<FuseExpandIntoPermuteQuantizeRewrite>(&ctx, _log);
    patterns.add<FuseQuantizeCastExpandIntoPermuteQuantizeQuantizeCastRewrite>(&ctx, _log);

    mlir::ConversionTarget target(ctx);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createFusePermuteQuantizeExpandPass
//
std::unique_ptr<mlir::Pass> vpux::IE::createFusePermuteQuantizeExpandPass(Logger log) {
    return std::make_unique<FusePermuteQuantizeExpandPass>(log);
}
