//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"

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

Byte getDmaSize(VPUIP::CopyOp copyOp) {
    const auto inputShape = getShape(copyOp.input());
    const auto outputShape = getShape(copyOp.output());
    VPUX_THROW_UNLESS(inputShape == outputShape,
                      "CopyOpTiling: Copy node's input and output have different shapes: {0} vs {1}", inputShape,
                      outputShape);

    // Sparse data is composed of multiple buffers which will later get ungrouped into individual Copy operations
    // Therefore, the maximum buffer size is selected for tiling
    if (auto sparseInput = copyOp.input().getType().dyn_cast<VPUIP::SparseBufferType>()) {
        auto dataSize = sparseInput.getData().cast<vpux::NDTypeInterface>().getCompactAllocSize();
        auto sparsityMapSize =
                (sparseInput.getSparsityMap() != nullptr)
                        ? sparseInput.getSparsityMap().cast<vpux::NDTypeInterface>().getCompactAllocSize()
                        : Byte(0);
        auto seTableSize =
                (sparseInput.getStorageElementTable() != nullptr)
                        ? sparseInput.getStorageElementTable().cast<vpux::NDTypeInterface>().getCompactAllocSize()
                        : Byte(0);
        return std::max({dataSize, sparsityMapSize, seTableSize});
    }

    return static_cast<Byte>(getCompactSize(copyOp.input()));
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
    CopyOpTiling(mlir::MLIRContext* ctx, Logger log, VPU::ArchKind arch)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx), _log(log), _arch(arch) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    SmallVector<mlir::Value> createTiles(VPUIP::CopyOp origOp, mlir::PatternRewriter& rewriter) const;

    Logger _log;
    VPU::ArchKind _arch;
};

SmallVector<mlir::Value> CopyOpTiling::createTiles(VPUIP::CopyOp origOp, mlir::PatternRewriter& rewriter) const {
    // Currently, tiling is implemented only for 4D shapes.
    const auto origInputShape = getShape(origOp.input());

    const auto fullCopySize = getDmaSize(origOp);

    auto tileDim = VPUIP::getCopyDMATilingDimForLargeSize(origOp);
    if (VPUIP::isSplitNeededForLargePlanesNum(origOp)) {
        tileDim = VPUIP::getCopyDMATilingDimForLargePlaneNum(origOp);
    }

    // We cannot _just_ divide the fullCopySize by sizeLimit to get the number of tiles required
    // Example: let fullCopySize=48MB, sizeLimit=16MB and IFM.C=4, then it would be 48/16=3 tiles, but it's obviously
    //          impossible to split 4 channels into 3 tiles each of those would fit the limits
    const auto numPlanesOfFullShape = origInputShape[tileDim];
    const auto singlePlaneSize = fullCopySize / numPlanesOfFullShape;
    //  The number of planes DMA could process within one tile. In case of small spatial dimensions of tensor (e.g.
    // 1x2048x8x8) it can exceed CMX_DMA_MAX_NUM_PLANES, so it's necessary to limit this value
    const auto maxNumPlanes = VPUIP::getMaxNumberPlanes(_arch);
    const auto desiredPlanesPerTileAmount = (VPUIP::DMA_LIMIT.count() / singlePlaneSize.count());
    VPUX_THROW_UNLESS(desiredPlanesPerTileAmount != 0,
                      "Couldn't split a CopyOp with single plane size greater than DMA_LIMIT");

    const auto numPlanesPerTile = std::min(desiredPlanesPerTileAmount, maxNumPlanes);

    SmallVector<mlir::Value> concatInputs;
    auto currentOffset = SmallVector<int64_t>(origInputShape.size(), 0);
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
                          currentTileShapeVector[tileDim.ind()], getDmaSize(copyTile));

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

/*
For two strides DMA in VPU, it will be implemented through plane.
If a two strides DMA do this date movement:
123 456 789
  ||
  \/                 | plane |
 1XX2XX3XX XXXXXXXXX 4XX5XX6XX XXXXXXXXX 7XX8XX9XX XXXXXXXXX
 |  |                |                   |
 stride              |                   |
                     |<-  plane stride ->|
The higher dim stride is implemented through plane stride.

So if the higher dim with stride size large than CMX_DMA_MAX_NUM_PLANES, we need tile the copy on this dim
*/

void CopyOpTilingPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    auto isLegalOp = [](VPUIP::CopyOp copyOp) {
        // Distributed CopyOp is not handled
        if (mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(copyOp->getParentOp()) != nullptr) {
            return true;
        }

        // If tensor size is greater than DMA_LIMIT its no longer legal operation
        if (getDmaSize(copyOp) > VPUIP::DMA_LIMIT) {
            return false;
        }

        return !VPUIP::isSplitNeededForLargePlanesNum(copyOp);
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<VPUIP::CopyOp>(isLegalOp);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<CopyOpTiling>(&ctx, _log, arch);

    // The new operations added by CopyOpTiling pattern:
    target.addLegalOp<VPUIP::SubViewOp>();
    target.addLegalOp<VPUIP::ConcatViewOp>();

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
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
