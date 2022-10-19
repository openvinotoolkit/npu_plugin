//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include "vpux/compiler/core/aliases_info.hpp"

#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/PatternMatch.h>

#include <mlir/Pass/PassManager.h>

using namespace vpux;

namespace {

int64_t getNumberOfPlanes(VPUIP::CopyOp copyOp) {
    const auto inputShape = getShape(copyOp.input());
    const auto inOrder = DimsOrder::fromValue(copyOp.input());
    const auto inMemShape = inOrder.toMemoryOrder(inputShape);

    return checked_cast<int64_t>(inMemShape[MemDim(Dims4D::Act::C.ind())]);
}

Byte getDmaSize(VPUIP::CopyOp copyOp) {
    const auto inputShape = getShape(copyOp.input());
    const auto outputShape = getShape(copyOp.output());
    VPUX_THROW_UNLESS(inputShape == outputShape,
                      "CopyOpTiling: Copy node's input and output have different shapes: {0} vs {1}", inputShape,
                      outputShape);

    return static_cast<Byte>(getCompactSize(copyOp.input()));
}

// The concept of striding levels means that tensor is not contiguous in some number of dimensions.
// For a contiguous tensor that number equals to 0.
// A tensor with the following properties has striding level 1:
// sizes: [1, 360, 1280, 18]
// strides: [235929600 Bit, 655360 Bit, 512 Bit, 16 Bit]
// Since 18 * 16 bit = 288 bit which is less than 512 bit (previous stride)
// A tensor with striding level 2 would look like that:
// sizes: [1, 360, 1280, 18]
// strides: [471859200 Bit, 1310720 Bit, 512 Bit, 16 Bit]
// 18 * 16 bit = 288 bit < 512 bit
// 1280 * 512 bit = 655360 bit < 1310720 bit

int64_t getStridingLevel(const mlir::Value val) {
    const auto dims = getShape(val);
    const auto strides = getStrides(val);
    const auto order = DimsOrder::fromValue(val);
    const auto dimsMemOrder = to_small_vector(order.toMemoryOrder(dims));
    const auto stridesMemOrder = to_small_vector(order.toMemoryOrder(strides));

    int64_t stridingLevel = 0;
    for (size_t ind = 1; ind < dimsMemOrder.size() && ind < stridesMemOrder.size(); ind++) {
        if (dimsMemOrder[ind] * stridesMemOrder[ind] != stridesMemOrder[ind - 1]) {
            stridingLevel++;
        }
    }
    return stridingLevel;
}

//
// CopyOpTilingPass
//

class CopyOpTilingPass final : public VPUIP::CopyOpTilingBase<CopyOpTilingPass> {
public:
    explicit CopyOpTilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// CopyOpTiling
//

// Splits large CopyOps into a bunch of smaller ones to fit DMA capabilities
class CopyOpTiling final : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    CopyOpTiling(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUIP::CopyOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    SmallVector<mlir::Value> createTiles(VPUIP::CopyOp origOp, mlir::PatternRewriter& rewriter) const;

    Logger _log;
};

SmallVector<mlir::Value> CopyOpTiling::createTiles(VPUIP::CopyOp origOp, mlir::PatternRewriter& rewriter) const {
    // Currently, tiling is implemented only for 4D shapes.
    const auto origInputShape = getShape(origOp.input());
    VPUX_THROW_UNLESS(origInputShape.size() == 4,
                      "CopyOpTiling: found shape {0} which is not supported yet (only 4D tensors are)", origInputShape);

    const auto fullCopySize = getDmaSize(origOp);
    // A workaround to always split by the first non-batch dimension, regardless the layout
    // NCHW - C, NHWC - H, NWHC - W
    const auto inOrder = DimsOrder::fromValue(origOp.input());
    const auto tileDim = inOrder.toDim(MemDim(Dims4D::Act::N.ind() + 1));

    // We cannot _just_ divide the fullCopySize by sizeLimit to get the number of tiles required
    // Example: let fullCopySize=48MB, sizeLimit=16MB and IFM.C=4, then it would be 48/16=3 tiles, but it's obviously
    //          impossible to split 4 channels into 3 tiles each of those would fit the limits
    const auto numPlanesOfFullShape = origInputShape[tileDim];
    const auto singlePlaneSize = fullCopySize / numPlanesOfFullShape;
    //  The number of planes DMA could process within one tile. In case of small spatial dimensions of tensor (e.g.
    // 1x2048x8x8) it can exceed CMX_DMA_MAX_NUM_PLANES, so it's necessary to limit this value
    const auto desiredPlanesPerTileAmount = (VPUIP::DMA_LIMIT.count() / singlePlaneSize.count());
    VPUX_THROW_UNLESS(desiredPlanesPerTileAmount != 0,
                      "Couldn't split a CopyOp with single plane size greater than DMA_LIMIT");

    const auto numPlanesPerTile = std::min(desiredPlanesPerTileAmount, VPUIP::CMX_DMA_MAX_NUM_PLANES);

    SmallVector<mlir::Value> concatInputs;
    auto currentOffset = SmallVector<int64_t>{0, 0, 0, 0};
    auto currentTileShapeVector = to_small_vector(origInputShape);
    auto planesLeftToCopy = numPlanesOfFullShape;
    for (int64_t tileIdx = 0; planesLeftToCopy > 0; ++tileIdx) {
        // Get the proper shape and a new location for the tile
        const auto tileLoc = appendLoc(origOp->getLoc(), "tile {0}", tileIdx);
        currentTileShapeVector[tileDim.ind()] = std::min(numPlanesPerTile, planesLeftToCopy);

        // Create the operations for it
        auto inputSubView =
                rewriter.create<VPUIP::SubViewOp>(tileLoc, origOp.input(), currentOffset, currentTileShapeVector);
        auto outputSubView =
                rewriter.create<VPUIP::SubViewOp>(tileLoc, origOp.output_buff(), currentOffset, currentTileShapeVector);
        auto copyTile = rewriter.create<VPUIP::CopyOp>(tileLoc, inputSubView.result(), outputSubView.result());

        concatInputs.push_back(copyTile.output());
        _log.nest().trace("Created tile #{0} for {1} planes that requires {2}", tileIdx,
                          currentTileShapeVector[tileDim.ind()], static_cast<Byte>(getCompactSize(copyTile.input())));

        // Take into account the part of the original tensor covered with the newly created tile
        planesLeftToCopy -= currentTileShapeVector[tileDim.ind()];
        currentOffset[tileDim.ind()] += currentTileShapeVector[tileDim.ind()];
    }

    VPUX_THROW_UNLESS(planesLeftToCopy == 0 && currentOffset[tileDim.ind()] == numPlanesOfFullShape,
                      "CopyOpTiling: a part of the original shape was not covered by Copy tiles");

    return concatInputs;
}

mlir::LogicalResult CopyOpTiling::matchAndRewrite(VPUIP::CopyOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Copy Operation '{0}'", origOp->getLoc());

    const auto concatInputs = createTiles(origOp, rewriter);

    rewriter.replaceOpWithNewOp<VPUIP::ConcatViewOp>(origOp, concatInputs, origOp.output_buff());

    return mlir::success();
}

//
// safeRunOnFunc
//

void CopyOpTilingPass::safeRunOnFunc() {
    auto& ctx = getContext();

    auto isLegalOp = [](VPUIP::CopyOp copyOp) {
        // If tensor size is greater than DMA_LIMIT its no longer legal operation
        if (getDmaSize(copyOp) > VPUIP::DMA_LIMIT) {
            return false;
        }

        const auto inputShape = getShape(copyOp.input());
        if (inputShape.size() < 4) {
            return true;
        }

        const auto inputStridingLevel = getStridingLevel(copyOp.input());
        const auto outputStridingLevel = getStridingLevel(copyOp.output());
        constexpr int64_t maxStridingLevel = 2;
        if (inputStridingLevel < maxStridingLevel && outputStridingLevel < maxStridingLevel) {
            // DMA transaction is able to handle such striding
            return true;
        }

        // If striding level is greater than 1, try splitting the tensor by plane dimension.
        return getNumberOfPlanes(copyOp) <= VPUIP::CMX_DMA_MAX_NUM_PLANES;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<VPUIP::CopyOp>(isLegalOp);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<CopyOpTiling>(&ctx, _log);

    // The new operations added by CopyOpTiling pattern:
    target.addLegalOp<VPUIP::SubViewOp>();
    target.addLegalOp<VPUIP::ConcatViewOp>();

    if (mlir::failed(mlir::applyPartialConversion(getFunction(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createCopyOpTilingPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createCopyOpTilingPass(Logger log) {
    return std::make_unique<CopyOpTilingPass>(log);
}
