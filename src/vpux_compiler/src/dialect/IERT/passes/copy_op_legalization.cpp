//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/IERT/passes.hpp"

#include "vpux/compiler/dialect/IERT/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"

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

Byte getDmaSize(IERT::CopyOp copyOp) {
    const auto inputShape = getShape(copyOp.input());
    const auto outputShape = getShape(copyOp.output());
    VPUX_THROW_UNLESS(inputShape == outputShape,
                      "CopyOpTiling: Copy node's input and output have different shapes: {0} vs {1}", inputShape,
                      outputShape);

    return static_cast<Byte>(getCompactSize(copyOp.input()));
}

//
// CopyOpLegalizationPass
//

class CopyOpLegalizationPass final : public IERT::CopyOpLegalizationBase<CopyOpLegalizationPass> {
public:
    explicit CopyOpLegalizationPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// CopyOpTiling
//

// Splits large CopyOps into a bunch of smaller ones to fit DMA capabilities
class CopyOpTiling final : public mlir::OpRewritePattern<IERT::CopyOp> {
public:
    CopyOpTiling(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IERT::CopyOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::CopyOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    SmallVector<mlir::Value> createChannelWiseTiles(IERT::CopyOp origOp, mlir::PatternRewriter& rewriter) const;

    Logger _log;
};

SmallVector<mlir::Value> CopyOpTiling::createChannelWiseTiles(IERT::CopyOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    // Currently, only C-wise tiling is implemented for 4D shapes. TODO: reuse the generic tiling utilities
    const auto origInputShape = getShape(origOp.input());
    VPUX_THROW_UNLESS(origInputShape.size() == 4,
                      "CopyOpTiling: found shape {0} which is not supported yet (only 4D tensors are)", origInputShape);

    const auto fullCopySize = getDmaSize(origOp);

    // We cannot _just_ divide the fullCopySize by sizeLimit to get the number of tiles required
    // Example: let fullCopySize=48MB, sizeLimit=16MB and IFM.C=4, then it would be 48/16=3 tiles, but it's obviously
    //          impossible to split 4 channels into 3 tiles each of those would fit the limits
    const auto numChannelsOfFullShape = origInputShape[Dims4D::Act::C];
    const auto singleChannelSize = fullCopySize / numChannelsOfFullShape;
    const auto numChannelsPerTile = (VPUIP::DMA_LIMIT.count() / singleChannelSize.count());
    const auto numTilesRequired = divUp(numChannelsOfFullShape, numChannelsPerTile);
    VPUX_THROW_UNLESS(numTilesRequired != 0,
                      "Couldn't split a CopyOp into tiles C-wise, consider extending the implementation to other axes");

    SmallVector<mlir::Value> concatInputs;
    auto currentOffset = SmallVector<int64_t>{0, 0, 0, 0};
    auto currentTileShapeVector = to_small_vector(origInputShape);
    auto channelsLeftToCopy = numChannelsOfFullShape;
    for (int64_t tileIdx = 0; channelsLeftToCopy > 0; ++tileIdx) {
        // Get the proper shape and a new location for the tile
        const auto tileLoc = appendLoc(origOp->getLoc(), llvm::formatv("tile {0}", tileIdx).str());
        currentTileShapeVector[Dims4D::Act::C.ind()] = std::min(numChannelsPerTile, channelsLeftToCopy);

        // Create the operations for it
        auto inputSubView =
                rewriter.create<IERT::SubViewOp>(tileLoc, origOp.input(), currentOffset, currentTileShapeVector);
        auto outputSubView =
                rewriter.create<IERT::SubViewOp>(tileLoc, origOp.output_buff(), currentOffset, currentTileShapeVector);
        auto copyTile = rewriter.create<IERT::CopyOp>(tileLoc, inputSubView.result(), outputSubView.result());

        concatInputs.push_back(copyTile.output());
        _log.nest().trace("Created tile #{0} for {1} channels that requires {2}", tileIdx,
                          currentTileShapeVector[Dims4D::Act::C.ind()],
                          static_cast<Byte>(getCompactSize(copyTile.input())));

        // Take into account the part of the original tensor covered with the newly created tile
        channelsLeftToCopy -= currentTileShapeVector[Dims4D::Act::C.ind()];
        currentOffset[Dims4D::Act::C.ind()] += currentTileShapeVector[Dims4D::Act::C.ind()];
    }

    VPUX_THROW_UNLESS(channelsLeftToCopy == 0 && currentOffset[Dims4D::Act::C.ind()] == numChannelsOfFullShape,
                      "CopyOpTiling: a part of the original shape was not covered by Copy tiles");

    return concatInputs;
}

mlir::LogicalResult CopyOpTiling::matchAndRewrite(IERT::CopyOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Copy Operation '{0}'", origOp->getLoc());

    // Current implementation only tries C-wise tiling. TODO: reuse the generic tiling utilities instead
    const auto concatInputs = createChannelWiseTiles(origOp, rewriter);

    rewriter.replaceOpWithNewOp<IERT::ConcatViewOp>(origOp, concatInputs, origOp.output_buff());

    return mlir::success();
}

//
// safeRunOnFunc
//

void CopyOpLegalizationPass::safeRunOnFunc() {
    auto& ctx = getContext();

    auto isLegalOp = [](IERT::CopyOp copyOp) {
        return getDmaSize(copyOp) <= VPUIP::DMA_LIMIT;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IERT::CopyOp>(isLegalOp);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<CopyOpTiling>(&ctx, _log);

    // The new operations added by CopyOpTiling pattern:
    target.addLegalOp<IERT::SubViewOp>();
    target.addLegalOp<IERT::ConcatViewOp>();

    if (mlir::failed(mlir::applyPartialConversion(getFunction(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createCopyOpLegalizationPass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createCopyOpLegalizationPass(Logger log) {
    return std::make_unique<CopyOpLegalizationPass>(log);
}
