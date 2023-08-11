//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

//
// TilingConverter
//

class TilingConverter final : public mlir::OpRewritePattern<VPU::LayoutCastOp> {
public:
    TilingConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::LayoutCastOp>(ctx), _log(log) {
        this->setDebugName("TilingConverter");
    }

private:
    mlir::LogicalResult matchAndRewrite(VPU::LayoutCastOp outLayoutCastOp, mlir::PatternRewriter& rewriter) const final;
    int64_t findSplitAxis(VPU::SliceOp sliceOp) const;

private:
    Logger _log;
};

mlir::LogicalResult TilingConverter::matchAndRewrite(VPU::LayoutCastOp outLayoutCastOp,
                                                     mlir::PatternRewriter& rewriter) const {
    auto* ctx = outLayoutCastOp->getContext();
    const auto loc = outLayoutCastOp.getLoc();
    auto concatOp = outLayoutCastOp.input().getDefiningOp<VPU::ConcatOp>();
    // ReshapeOp may be folded, that's why we explicitly check that it exists as a trailing
    // operation to layout cast and don't try to match the whole chain starting from affine reshape.
    mlir::Operation* reshapeOp = nullptr;
    if (outLayoutCastOp->hasOneUse()) {
        reshapeOp = mlir::dyn_cast_or_null<VPU::AffineReshapeOp>(*outLayoutCastOp->getUsers().begin());
    }

    mlir::Value originalInput = nullptr;
    SmallVector<mlir::ArrayAttr> sliceOffsets;
    SmallVector<mlir::ArrayAttr> sliceSizes;
    SmallVector<Shape> permuteQuantizeShape;
    SmallVector<VPU::NCEPermuteQuantizeOp> permuteQuantizes;
    SmallVector<mlir::Operation*> opsToDelete;
    for (const auto input : concatOp.inputs()) {
        auto permuteQuantize = input.getDefiningOp<VPU::NCEPermuteQuantizeOp>();
        const Shape sliceShape = getShape(permuteQuantize.input()).toValues();
        permuteQuantizeShape.push_back(sliceShape);

        auto slice = permuteQuantize.input().getDefiningOp<VPU::SliceOp>();
        sliceOffsets.push_back(slice.static_offsets());
        sliceSizes.push_back(slice.static_sizes());
        auto inLayoutCast = slice.source().getDefiningOp<VPU::LayoutCastOp>();
        auto inReshape = inLayoutCast.input().getDefiningOp<VPU::ReshapeOp>();
        originalInput = inReshape.input();

        permuteQuantizes.push_back(permuteQuantize);

        opsToDelete.push_back(permuteQuantize);
        opsToDelete.push_back(slice);
    }

    VPUX_THROW_UNLESS(originalInput != nullptr, "Unable to fint the input for the sequence");
    int64_t splitAxis = -1;

    // Slice the original input.
    SmallVector<mlir::Value> concatInputs;
    for (const auto sliceIdx : irange(concatOp.inputs().size())) {
        const auto srcStaticOffsets = parseIntArrayAttr<int64_t>(sliceOffsets[sliceIdx]);
        const SmallVector<int64_t> dstStaticOffsets = {
                srcStaticOffsets[Dims4D::Act::N.ind()],
                srcStaticOffsets[Dims4D::Act::H.ind()],
                srcStaticOffsets[Dims4D::Act::W.ind()],
                srcStaticOffsets[Dims4D::Act::C.ind()],
        };
        const auto srcStaticSizes = parseIntArrayAttr<int64_t>(sliceSizes[sliceIdx]);
        const SmallVector<int64_t> dstStaticSizes = {
                srcStaticSizes[Dims4D::Act::N.ind()],
                srcStaticSizes[Dims4D::Act::H.ind()],
                srcStaticSizes[Dims4D::Act::W.ind()],
                srcStaticSizes[Dims4D::Act::C.ind()],
        };
        const auto inputSliceLoc = appendLoc(loc, "_slice_{0}_{1}", dstStaticOffsets, dstStaticSizes);
        auto inputSlice =
                rewriter.create<VPU::SliceOp>(inputSliceLoc, originalInput, getIntArrayAttr(ctx, dstStaticOffsets),
                                              getIntArrayAttr(ctx, dstStaticSizes));

        if (splitAxis == -1) {
            splitAxis = findSplitAxis(inputSlice);
            VPUX_THROW_WHEN(splitAxis < 0, "TilingConverter::findSplitAxis failed to find split axis");
        }

        const auto newInputShape = permuteQuantizeShape[sliceIdx];
        const auto newInputShapeAttr = getIntArrayAttr(ctx, newInputShape);
        const auto inputReshapeLoc = appendLoc(loc, "input reshape {0}", dstStaticOffsets);
        auto inputReshape = rewriter.create<VPU::ReshapeOp>(inputReshapeLoc, inputSlice.result(), nullptr, false,
                                                            newInputShapeAttr);

        const auto targetInOrder = DimsOrder::NHWC;
        const auto orderInAttr = mlir::AffineMapAttr::get(targetInOrder.toAffineMap(ctx));
        const auto inputLayoutCastLoc = appendLoc(loc, "input layout cast {0}", dstStaticOffsets);
        auto inputLayoutCast = rewriter.create<VPU::LayoutCastOp>(inputLayoutCastLoc, inputReshape, orderInAttr);

        const auto permuteQuantizesLoc = appendLoc(loc, "permute quantize {0}", dstStaticOffsets);
        auto permuteQuantizeTile = rewriter.create<VPU::NCEPermuteQuantizeOp>(
                permuteQuantizesLoc, permuteQuantizes[sliceIdx].output().getType(), inputLayoutCast.output(),
                permuteQuantizes[sliceIdx].padAttr(), permuteQuantizes[sliceIdx].dstElemTypeAttr(),
                permuteQuantizes[sliceIdx].dstOrderAttr(), permuteQuantizes[sliceIdx].ppeAttr(),
                permuteQuantizes[sliceIdx].multiClusterStrategyAttr());

        const auto targetOutOrder = DimsOrder::NHWC;
        const auto orderOutAttr = mlir::AffineMapAttr::get(targetOutOrder.toAffineMap(ctx));
        const auto outputLayoutCastLoc = appendLoc(loc, "output layout cast {0}", dstStaticOffsets);
        auto outputLayoutCast = rewriter.create<VPU::LayoutCastOp>(outputLayoutCastLoc,
                                                                   permuteQuantizeTile->getResult(0), orderOutAttr);

        const Shape srcShape = getShape(outputLayoutCast.output()).toValues();
        const SmallVector<int64_t> dstShape = {
                srcShape[Dims4D::Act::N],
                srcShape[Dims4D::Act::H],
                srcShape[Dims4D::Act::W],
                srcShape[Dims4D::Act::C],
        };

        const auto outputLayoutCastType = outputLayoutCast.output().getType().cast<vpux::NDTypeInterface>();
        SmallVector<SmallVector<int64_t>> reassociationMap(dstShape.size());
        for (size_t dimIdx = 0; dimIdx < reassociationMap.size(); dimIdx++) {
            reassociationMap[dimIdx].push_back(dimIdx);
        }
        const auto outReshapeOpLoc = appendLoc(loc, "output reshape {0}", dstStaticOffsets);
        auto outReshapeOp = rewriter.create<VPU::AffineReshapeOp>(
                outReshapeOpLoc, outputLayoutCastType.changeShape(ShapeRef(dstShape)), outputLayoutCast.output(),
                getIntArrayOfArray(ctx, reassociationMap), getIntArrayAttr(ctx, dstShape));
        concatInputs.push_back(outReshapeOp.output());
    }

    const auto outputConcatLoc = appendLoc(loc, "concat permute quantize");
    auto outputConcat = rewriter.create<VPU::ConcatOp>(outputConcatLoc, concatInputs, splitAxis);
    if (reshapeOp != nullptr) {
        rewriter.replaceOp(reshapeOp, outputConcat->getResult(0));
        rewriter.eraseOp(outLayoutCastOp);
    } else {
        rewriter.replaceOp(outLayoutCastOp, outputConcat->getResult(0));
    }

    rewriter.eraseOp(concatOp);

    for (const auto& op : opsToDelete) {
        rewriter.eraseOp(op);
    }

    return mlir::success();
}

int64_t TilingConverter::findSplitAxis(VPU::SliceOp sliceOp) const {
    const auto inputShape = getShape(sliceOp.source());
    const auto outputShape = getShape(sliceOp.result());
    int64_t splitAxis = -1;
    for (size_t axis = 0; axis < inputShape.size() && axis < outputShape.size(); axis++) {
        if (inputShape[Dim(axis)] != outputShape[Dim(axis)]) {
            VPUX_THROW_UNLESS(splitAxis == -1, "Multiple axis slice is not supported");
            splitAxis = axis;
        }
    }

    return splitAxis;
}

bool isPermuteQuantizePattern(VPU::LayoutCastOp outLayoutCast, Logger log) {
    auto maybeConcatOp = outLayoutCast.input().getDefiningOp();
    if (!mlir::isa<VPU::ConcatOp>(maybeConcatOp)) {
        log.trace("isPermuteQuantizePattern: cannot find VPU.Concat. PermuteQuantize is not tiled.");
        return false;
    }
    auto concatOp = mlir::dyn_cast<VPU::ConcatOp>(maybeConcatOp);

    for (const auto input : concatOp.inputs()) {
        auto permuteQuantize = input.getDefiningOp<VPU::NCEPermuteQuantizeOp>();
        if (permuteQuantize == nullptr) {
            log.trace("isPermuteQuantizePattern: VPU.Concat input is not a VPU.NCE.PermuteQuantize operation");
            return false;
        }
        auto slice = permuteQuantize.input().getDefiningOp<VPU::SliceOp>();
        if (slice == nullptr) {
            log.trace("isPermuteQuantizePattern: VPU.NCE.PermuteQuantize input is not sliced");
            return false;
        }
        auto inLayoutCast = slice.source().getDefiningOp<VPU::LayoutCastOp>();
        if (inLayoutCast == nullptr) {
            log.trace("isPermuteQuantizePattern: VPU.Slice input is not a VPU.LayoutCast operation");
            return false;
        }
        auto inReshape = inLayoutCast.input().getDefiningOp<VPU::ReshapeOp>();
        if (inReshape == nullptr) {
            log.trace("isPermuteQuantizePattern: VPU.LayoutCast input is not a VPU.Reshape operation");
            return false;
        }
    }

    return true;
}

//
// AdjustTilingForPermuteQuantizePass
//

class AdjustTilingForPermuteQuantizePass final :
        public VPU::AdjustTilingForPermuteQuantizePassBase<AdjustTilingForPermuteQuantizePass> {
public:
    explicit AdjustTilingForPermuteQuantizePass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void AdjustTilingForPermuteQuantizePass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();
    mlir::ConversionTarget target(ctx);
    const auto isLegalLayoutCast = [&](const VPU::LayoutCastOp op) {
        return !isPermuteQuantizePattern(op, _log);
    };
    target.addDynamicallyLegalOp<VPU::LayoutCastOp>(isLegalLayoutCast);
    target.addLegalOp<VPU::SliceOp>();
    target.addLegalOp<VPU::ReshapeOp>();
    target.addLegalOp<VPU::AffineReshapeOp>();
    target.addLegalOp<VPU::NCEPermuteQuantizeOp>();
    target.addLegalOp<VPU::ConcatOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<TilingConverter>(&ctx, _log);
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createAdjustTilingForPermuteQuantizePass(Logger log) {
    return std::make_unique<AdjustTilingForPermuteQuantizePass>(log);
}
