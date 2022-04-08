//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/numeric.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// SplitDepthToSpace
//

class SplitDepthToSpacePass final : public IE::SplitDepthToSpaceBase<SplitDepthToSpacePass> {
public:
    explicit SplitDepthToSpacePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class DepthToSpaceConverter;

private:
    void safeRunOnFunc() final;
};

class SplitDepthToSpacePass::DepthToSpaceConverter final : public mlir::OpRewritePattern<IE::DepthToSpaceOp> {
public:
    DepthToSpaceConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::DepthToSpaceOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DepthToSpaceOp origOp, mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
};

inline Optional<Dim> getDimToSplit(const NDTypeInterface& inputType) {
    // pick the highest dimension other than N to split to avoid redundant stride DMAs
    // N!=1 is not supported
    // e.g., NHWC, return H; NCHW return C
    if (inputType.getShape()[inputType.getDimsOrder().dimAt(0)] != 1) {
        return None;
    }
    return inputType.getDimsOrder().dimAt(1);
}

Optional<SmallVector<Shape>> getDepthToSpaceSubShape(NDTypeInterface inputType, Byte cmxMemSize, int64_t blockSize) {
    const auto inputShape = inputType.getShape();
    const auto elemTypeSize = Byte(inputType.getElemTypeSize());
    const auto IC = inputShape[Dims4D::Act::C];
    const auto IH = inputShape[Dims4D::Act::H];
    const auto IW = inputShape[Dims4D::Act::W];

    auto dimToSplitResult = getDimToSplit(inputType);
    if (!dimToSplitResult.hasValue()) {
        return None;
    }
    auto dimToSplit = dimToSplitResult.getValue();
    auto splitDimOrigSize = inputType.getDimsOrder() == vpux::DimsOrder::NHWC ? IH : IC;
    // "2" represents two times of the data size, including input and output
    auto unchangedDimsSize = IC * IH * IW * 2 * elemTypeSize.count() / splitDimOrigSize;
    const int64_t subDimSize = cmxMemSize.count() / unchangedDimsSize;
    if (subDimSize < 1) {
        // couldn't make the op fit into CMX with one dimension splitting
        return None;
    }
    auto splitNum = divUp(splitDimOrigSize, subDimSize);
    auto subShape = Shape(to_small_vector(inputShape));
    if (splitNum > 1) {
        subShape[dimToSplit] = subDimSize;
        SmallVector<Shape> outputShapes(splitNum - 1, subShape);
        subShape[dimToSplit] = splitDimOrigSize - subDimSize * (splitNum - 1);
        outputShapes.push_back(subShape);
        if (dimToSplit == Dims4D::Act::C &&
            std::any_of(outputShapes.begin(), outputShapes.end(), [&](Shape outputShape) {
                return outputShape[Dims4D::Act::C] % blockSize != 0;
            })) {
            // the input channel size must be aligned to the blockSize
            return None;
        }
        return outputShapes;
    }

    return SmallVector<Shape>{subShape};
}

mlir::LogicalResult SplitDepthToSpacePass::DepthToSpaceConverter::matchAndRewrite(
        IE::DepthToSpaceOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto ctx = rewriter.getContext();

    const auto inputType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto outputType = origOp.output().getType().cast<vpux::NDTypeInterface>();

    Byte requiredCMX(0);
    requiredCMX += inputType.getTotalAllocSize();
    requiredCMX += outputType.getTotalAllocSize();
    const auto cmxMemSize = VPU::getTotalCMXSize(origOp.getOperation());
    // The purpose to split DepthToSpace is:
    //   1. The full tensor can't fit into CMX.
    //   2. Convert DepthToSpace to CMX stride DMA is more efficient.
    if (requiredCMX <= cmxMemSize) {
        _log.trace("DepthToSpaceOp Op {0} can fit into CMX, doesn't need to split.", origOp.getLoc());
        return mlir::failure();
    }

    auto subShapes = getDepthToSpaceSubShape(inputType, cmxMemSize, origOp.block_size());
    if (!subShapes.hasValue()) {
        return mlir::failure();
    }

    SmallVector<mlir::Value> subOutputs;
    int64_t offset = 0;
    auto offsets = SmallVector<int64_t>(inputType.getShape().size(), 0);
    auto dimToSplit = getDimToSplit(inputType).getValue();
    for (int index = 0; index < checked_cast<int64_t>(subShapes.getValue().size()); index++) {
        const auto loc = appendLoc(origOp->getLoc(), "slice {0}", index);
        offsets[dimToSplit.ind()] = offset;
        auto sliceOp = rewriter.create<IE::SliceOp>(loc, origOp.input(), getIntArrayAttr(ctx, offsets),
                                                    getIntArrayAttr(ctx, subShapes.getValue()[index]));
        offset += subShapes.getValue()[index][dimToSplit];
        auto depthToSpaceOp = rewriter.create<IE::DepthToSpaceOp>(origOp.getLoc(), sliceOp.result(),
                                                                  origOp.block_sizeAttr(), origOp.modeAttr());
        auto subDepthToSpaceOutputType = depthToSpaceOp.output().getType().cast<NDTypeInterface>();
        depthToSpaceOp.output().setType(subDepthToSpaceOutputType.changeDimsOrder(outputType.getDimsOrder()));
        subOutputs.push_back(depthToSpaceOp.output());
    }

    _log.trace("Split DepthToSpaceOp Op {0} by {1}", origOp.getLoc(), subShapes);
    rewriter.replaceOpWithNewOp<IE::ConcatOp>(origOp, subOutputs, dimToSplit);

    return mlir::success();
}

//
// safeRunOnFunc
//

void SplitDepthToSpacePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<DepthToSpaceConverter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createSplitDepthToSpacePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createSplitDepthToSpacePass(Logger log) {
    return std::make_unique<SplitDepthToSpacePass>(log);
}
