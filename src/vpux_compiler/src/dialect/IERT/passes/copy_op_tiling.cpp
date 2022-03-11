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

int64_t getNumberOfPlanes(IERT::CopyOp copyOp) {
    const auto inputShape = getShape(copyOp.input());
    const auto inOrder = DimsOrder::fromValue(copyOp.input());
    const auto inMemShape = inOrder.toMemoryOrder(inputShape);

    return checked_cast<int64_t>(inMemShape[MemDim(Dims4D::Act::C.ind())]);
}

Byte getDmaSize(IERT::CopyOp copyOp) {
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

class CopyOpTilingPass final : public IERT::CopyOpTilingBase<CopyOpTilingPass> {
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
    // Work around to split always by first dim, not depending on layout
    // NCHW - C, NHWC - H, NWHC - W
    const auto inOrder = DimsOrder::fromValue(origOp.input());
    const auto tileDim = inOrder.toDim(MemDim(Dims4D::Act::C.ind()));

    // We cannot _just_ divide the fullCopySize by sizeLimit to get the number of tiles required
    // Example: let fullCopySize=48MB, sizeLimit=16MB and IFM.C=4, then it would be 48/16=3 tiles, but it's obviously
    //          impossible to split 4 channels into 3 tiles each of those would fit the limits
    const auto numPlanesOfFullShape = origInputShape[tileDim];
    const auto singlePlaneSize = fullCopySize / numPlanesOfFullShape;
    // How many channels DMA could process within one tile. In case of small spatial dimensions of tensor (f.e.
    // 1x2048x8x8) it can exceed CMX_DMA_MAX_NUM_PLANES, so neccesary to limit this value
    const auto desiredPlanesPerTileAmount = (VPUIP::DMA_LIMIT.count() / singlePlaneSize.count());
    VPUX_THROW_UNLESS(desiredPlanesPerTileAmount != 0,
                      "Couldn't split a CopyOp with single plane size greater then DMA_LIMIT");

    const auto numPlanesPerTile = std::min(desiredPlanesPerTileAmount, VPUIP::CMX_DMA_MAX_NUM_PLANES);

    SmallVector<mlir::Value> concatInputs;
    auto currentOffset = SmallVector<int64_t>{0, 0, 0, 0};
    auto currentTileShapeVector = to_small_vector(origInputShape);
    auto planesLeftToCopy = numPlanesOfFullShape;
    for (int64_t tileIdx = 0; planesLeftToCopy > 0; ++tileIdx) {
        // Get the proper shape and a new location for the tile
        const auto tileLoc = appendLoc(origOp->getLoc(), llvm::formatv("tile {0}", tileIdx).str());
        currentTileShapeVector[tileDim.ind()] = std::min(numPlanesPerTile, planesLeftToCopy);

        // Create the operations for it
        auto inputSubView =
                rewriter.create<IERT::SubViewOp>(tileLoc, origOp.input(), currentOffset, currentTileShapeVector);
        auto outputSubView =
                rewriter.create<IERT::SubViewOp>(tileLoc, origOp.output_buff(), currentOffset, currentTileShapeVector);
        auto copyTile = rewriter.create<IERT::CopyOp>(tileLoc, inputSubView.result(), outputSubView.result());

        concatInputs.push_back(copyTile.output());
        _log.nest().trace("Created tile #{0} for {1} channels that requires {2}", tileIdx,
                          currentTileShapeVector[tileDim.ind()], static_cast<Byte>(getCompactSize(copyTile.input())));

        // Take into account the part of the original tensor covered with the newly created tile
        planesLeftToCopy -= currentTileShapeVector[tileDim.ind()];
        currentOffset[tileDim.ind()] += currentTileShapeVector[tileDim.ind()];
    }

    VPUX_THROW_UNLESS(planesLeftToCopy == 0 && currentOffset[tileDim.ind()] == numPlanesOfFullShape,
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

void CopyOpTilingPass::safeRunOnFunc() {
    auto& ctx = getContext();

    auto isLegalOp = [](IERT::CopyOp copyOp) {
        const auto isDmaLegal = getDmaSize(copyOp) <= VPUIP::DMA_LIMIT;

        const auto inputShape = getShape(copyOp.input());
        if (inputShape.size() < 4) {
            return true;
        }

        const auto inputStridingLevel = getStridingLevel(copyOp.input());
        const auto outputStridingLevel = getStridingLevel(copyOp.output());
        constexpr int64_t maxStridingLevel = 2;
        if (inputStridingLevel < maxStridingLevel && outputStridingLevel < maxStridingLevel) {
            // DMA transaction is able to handle such striding
            return isDmaLegal;
        }

        // If striding level is greater than 1, try splitting the tensor by plane dimension.
        return getNumberOfPlanes(copyOp) <= VPUIP::CMX_DMA_MAX_NUM_PLANES && isDmaLegal;
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
// createCopyOpTilingPass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createCopyOpTilingPass(Logger log) {
    return std::make_unique<CopyOpTilingPass>(log);
}
