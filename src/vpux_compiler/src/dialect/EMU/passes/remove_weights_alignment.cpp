//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/EMU/ops.hpp"
#include "vpux/compiler/dialect/EMU/passes.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"

#include "vpux/utils/core/enums.hpp"

using namespace vpux;

namespace {

/*
   DWCONV weights are aligned following the rule: {OC, IC * KY * KX + padding, 1, 1}
   Therefore, we first strip the padding using a subview transformation -> {OC, IC * KY * KX, 1, 1}
   After that we reshape the weights to the original shape -> {OC, IC, KY, KX}
*/
mlir::Value handleDWConstWeights(mlir::PatternRewriter& rewriter, Const::DeclareOp weightsConst,
                                 mlir::ArrayAttr rawFilterShape) {
    const auto weightsContentAttr = weightsConst.contentAttr();

    const auto finalWeightsShape = parseIntArrayAttr<int64_t>(rawFilterShape);

    const auto KX = finalWeightsShape[3];
    const auto KY = finalWeightsShape[2];
    const auto IC = finalWeightsShape[1];
    const auto OC = finalWeightsShape[0];

    const auto reorderedWeightsContentAttr = weightsContentAttr.reorder(DimsOrder::NCHW);
    const auto noPadWeightsContentAttr = reorderedWeightsContentAttr.subview({0, 0, 0, 0}, {OC, IC * KY * KX, 1, 1});
    const auto finalWeightsContentAttr = noPadWeightsContentAttr.reshape(Shape(finalWeightsShape));

    const auto weightsType = weightsConst.output().getType().cast<NDTypeInterface>();
    const auto outAllocType =
            mlir::RankedTensorType::get(finalWeightsShape, weightsType.getElementType()).cast<vpux::NDTypeInterface>();

    auto finalWeightsOp =
            rewriter.create<Const::DeclareOp>(weightsConst->getLoc(), outAllocType, finalWeightsContentAttr);
    return finalWeightsOp.output();
}

/*
   NonConst DWCONV weights are aligned following the rule: {OC, IC * KY * KX + padding, 1, 1}
   But since the weights input is not const, the alignment is done through a chain of ops:
        Reshape/AffineReshape -> PermuteCast -> Concat (adds the padding) -> PermuteCast
   Therefore, we check that this sequence of ops exist and, if it does, we replace the weights input
   with the input of the alignment subgraph.
*/
mlir::FailureOr<mlir::Value> handleDWNonConstWeights(mlir::Value weights) {
    auto permuteCast0 = weights.getDefiningOp<IE::PermuteCastOp>();
    if (permuteCast0 == nullptr) {
        return mlir::failure();
    }

    auto concat = permuteCast0.input().getDefiningOp<IE::ConcatOp>();
    if (concat == nullptr || concat.inputs().size() != 2 ||
        concat.inputs()[1].getDefiningOp<Const::DeclareOp>() == nullptr) {
        return mlir::failure();
    }

    auto permuteCast1 = concat.inputs()[0].getDefiningOp<IE::PermuteCastOp>();
    if (permuteCast1 == nullptr) {
        return mlir::failure();
    }

    auto reshape = permuteCast1.input().getDefiningOp<IE::ReshapeOp>();
    if (reshape == nullptr) {
        auto affineReshape = permuteCast1.input().getDefiningOp<IE::AffineReshapeOp>();
        if (affineReshape == nullptr) {
            return mlir::failure();
        }

        return affineReshape.input();
    }

    return reshape.input();
}

mlir::FailureOr<mlir::Value> adjustDWConvWeights(mlir::PatternRewriter& rewriter, EMU::NCEClusterTaskOp origOp) {
    const auto origWeights = origOp.weights();
    if (auto weightsConst = origWeights.getDefiningOp<Const::DeclareOp>()) {
        return handleDWConstWeights(rewriter, weightsConst, origOp.rawFilterShapeAttr());
    }

    return handleDWNonConstWeights(origWeights);
}

/*
   CMCONV weights are aligned following the rule: {OC, 1, 1, IC * KY * KX + padding}
   Therefore, we first strip the padding using a subview transformation -> {OC, 1, 1, IC * KY * KX}
   After that we reshape the weights to the original shape -> {OC, IC, KY, KX}
*/
mlir::FailureOr<mlir::Value> adjustConvWeights(mlir::PatternRewriter& rewriter, EMU::NCEClusterTaskOp origOp) {
    auto origWeightsConst = origOp.weights().getDefiningOp<Const::DeclareOp>();
    const auto weightsType = origWeightsConst.output().getType().cast<NDTypeInterface>();
    const auto weightsContentAttr = origWeightsConst.contentAttr();

    const auto finalWeightsShape = parseIntArrayAttr<int64_t>(origOp.rawFilterShapeAttr());

    const auto KX = finalWeightsShape[Dims4D::Filter::KX.ind()];
    const auto KY = finalWeightsShape[Dims4D::Filter::KY.ind()];
    const auto IC = finalWeightsShape[Dims4D::Filter::IC.ind()];
    const auto OC = finalWeightsShape[Dims4D::Filter::OC.ind()];

    const auto reorderedWeightsContentAttr = weightsContentAttr.reorder(DimsOrder::NCHW);
    const auto noPadWeightsContentAttr = reorderedWeightsContentAttr.subview({0, 0, 0, 0}, {OC, 1, 1, IC * KY * KX});
    const auto finalWeightsContentAttr = noPadWeightsContentAttr.reshape(Shape(finalWeightsShape));

    const auto outAllocType =
            mlir::RankedTensorType::get(finalWeightsShape, weightsType.getElementType()).cast<vpux::NDTypeInterface>();

    auto finalWeightsOp =
            rewriter.create<Const::DeclareOp>(origWeightsConst->getLoc(), outAllocType, finalWeightsContentAttr);
    return finalWeightsOp.output();
}

//
// RemoveWeightsAlignmentRewriter
//

class RemoveWeightsAlignmentRewriter final : public mlir::OpRewritePattern<EMU::NCEClusterTaskOp> {
public:
    RemoveWeightsAlignmentRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<EMU::NCEClusterTaskOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(EMU::NCEClusterTaskOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult RemoveWeightsAlignmentRewriter::matchAndRewrite(EMU::NCEClusterTaskOp origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto taskType = origOp.task_type();
    const auto weights = (taskType == VPUIP::NCETaskType::CMCONV || taskType == VPUIP::NCETaskType::CONV)
                                 ? adjustConvWeights(rewriter, origOp)
                                 : adjustDWConvWeights(rewriter, origOp);

    if (mlir::failed(weights)) {
        return mlir::failure();
    }

    auto nceOp = rewriter.create<EMU::NCEClusterTaskOp>(origOp->getLoc(), origOp.getType(), origOp.input(),
                                                        weights.getValue(), origOp.weight_table(), taskType,
                                                        origOp.kernel_sizeAttr(), origOp.kernel_stridesAttr(),
                                                        origOp.kernel_paddingAttr(), origOp.rawFilterShapeAttr());

    nceOp.addPPETask(rewriter, origOp);

    rewriter.replaceOp(origOp, nceOp.output());
    return mlir::success();
}

//
// RemoveWeightsAlignmentPass
//

class RemoveWeightsAlignmentPass final : public EMU::RemoveWeightsAlignmentBase<RemoveWeightsAlignmentPass> {
public:
    explicit RemoveWeightsAlignmentPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    };

private:
    void safeRunOnFunc() final;
};

/*
   For CONV (on 37XX and 40XX), CMCONV and DWCONV, weights alignment is done as part of IE2VPU lowering. Emulator is not
   equiped to work with padded weights that squash IC, KY and KX into one dimension. Therefore, this pass is reverting
   the weights for these 2 ops to the original shape, stored in rasFilterShape attr.
*/
void RemoveWeightsAlignmentPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<EMU::NCEClusterTaskOp>([&](EMU::NCEClusterTaskOp op) {
        auto taskType = op.task_type();
        if (taskType != VPUIP::NCETaskType::CMCONV && taskType != VPUIP::NCETaskType::DWCONV &&
            taskType != VPUIP::NCETaskType::CONV) {
            return true;
        }

        const auto weightsType = op.weights().getType().cast<NDTypeInterface>();
        const auto weightsShape = weightsType.getShape();

        const Shape origWeightsShape = Shape(parseIntArrayAttr<int64_t>(op.rawFilterShapeAttr()));
        return weightsShape == origWeightsShape;
    });

    target.addLegalOp<EMU::PPETaskOp>();
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<RemoveWeightsAlignmentRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::EMU::createRemoveWeightsAlignmentPass(Logger log) {
    return std::make_unique<RemoveWeightsAlignmentPass>(log);
}
