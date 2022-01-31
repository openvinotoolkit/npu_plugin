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
// CopyOpSplitByPlanesPass
//

class CopyOpSplitByPlanesPass final : public IERT::CopyOpSplitByPlanesBase<CopyOpSplitByPlanesPass> {
public:
    explicit CopyOpSplitByPlanesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// CopyOpPlaneTiling
//

// Splits CopyOps which exceed maximal number of planes into a bunch of smaller ones to fit DMA capabilities
class CopyOpPlaneTiling final : public mlir::OpRewritePattern<IERT::CopyOp> {
public:
    CopyOpPlaneTiling(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IERT::CopyOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::CopyOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    SmallVector<mlir::Value> tileByPlanes(IERT::CopyOp origOp, mlir::PatternRewriter& rewriter) const;

    Logger _log;
};

SmallVector<mlir::Value> CopyOpPlaneTiling::tileByPlanes(IERT::CopyOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto origInputShape = getShape(origOp.input());
    VPUX_THROW_UNLESS(origInputShape.size() == 4,
                      "CopyOpPlaneTiling: found shape {0} which is not supported yet (only 4D tensors are)",
                      origInputShape);
    const int64_t srcPlanes = getNumberOfPlanes(origOp);

    const auto numberOfSplits = srcPlanes / VPUIP::CMX_DMA_MAX_NUM_PLANES;
    const auto splitRemainder = srcPlanes % VPUIP::CMX_DMA_MAX_NUM_PLANES;
    auto splitSizes = SmallVector<int64_t>(numberOfSplits, VPUIP::CMX_DMA_MAX_NUM_PLANES);
    if (splitRemainder > 0) {
        splitSizes.push_back(splitRemainder);
    }

    const auto inOrder = DimsOrder::fromValue(origOp.input());
    const auto planeDim = inOrder.toDim(MemDim(Dims4D::Act::C.ind()));

    SmallVector<mlir::Value> concatInputs;
    auto currentOffset = SmallVector<int64_t>{0, 0, 0, 0};
    auto currentTileShapeVector = to_small_vector(origInputShape);
    for (size_t tileIdx = 0; tileIdx < splitSizes.size(); tileIdx++) {
        const auto tileLoc = appendLoc(origOp->getLoc(), llvm::formatv("tile {0}", tileIdx).str());
        currentTileShapeVector[planeDim.ind()] = splitSizes[tileIdx];

        auto inputSubView =
                rewriter.create<IERT::SubViewOp>(tileLoc, origOp.input(), currentOffset, currentTileShapeVector);
        auto outputSubView =
                rewriter.create<IERT::SubViewOp>(tileLoc, origOp.output_buff(), currentOffset, currentTileShapeVector);
        auto copyTile = rewriter.create<IERT::CopyOp>(tileLoc, inputSubView.result(), outputSubView.result());

        concatInputs.push_back(copyTile.output());

        // Take into account the part of the original tensor covered with the newly created tile
        currentOffset[planeDim.ind()] += currentTileShapeVector[planeDim.ind()];
    }

    return concatInputs;
}

mlir::LogicalResult CopyOpPlaneTiling::matchAndRewrite(IERT::CopyOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Copy Operation '{0}'", origOp->getLoc());

    const auto concatInputs = tileByPlanes(origOp, rewriter);
    rewriter.replaceOpWithNewOp<IERT::ConcatViewOp>(origOp, concatInputs, origOp.output_buff());

    return mlir::success();
}

//
// safeRunOnFunc
//

void CopyOpSplitByPlanesPass::safeRunOnFunc() {
    auto& ctx = getContext();

    auto isLegalOp = [](IERT::CopyOp copyOp) {
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
    target.addDynamicallyLegalOp<IERT::CopyOp>(isLegalOp);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<CopyOpPlaneTiling>(&ctx, _log);

    target.addLegalOp<IERT::SubViewOp>();
    target.addLegalOp<IERT::ConcatViewOp>();

    if (mlir::failed(mlir::applyPartialConversion(getFunction(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createCopyOpSplitByPlanesPass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createCopyOpSplitByPlanesPass(Logger log) {
    return std::make_unique<CopyOpSplitByPlanesPass>(log);
}
