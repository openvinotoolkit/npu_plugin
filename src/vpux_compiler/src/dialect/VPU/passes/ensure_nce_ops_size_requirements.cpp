//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <mlir/IR/BlockAndValueMapping.h>
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

SmallVector<Dim> getDimsOverHWLimit(ShapeRef shape) {
    SmallVector<Dim> wrongDims = {};
    for (size_t i = 0; i < shape.size(); i++) {
        const auto dim = Dim(i);
        if (shape[dim] > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
            wrongDims.push_back(dim);
        }
    }
    return wrongDims;
}

//
//  EnsureSizeRequirements
//

class EnsureSizeRequirements final : public mlir::OpInterfaceRewritePattern<VPU::TilingBuilderOpInterface> {
public:
    EnsureSizeRequirements(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<VPU::TilingBuilderOpInterface>(ctx), _log(log) {
        this->setDebugName("EnsureSizeRequirements");
    }
    mlir::LogicalResult matchAndRewrite(VPU::TilingBuilderOpInterface origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult EnsureSizeRequirements::matchAndRewrite(VPU::TilingBuilderOpInterface origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    auto op = origOp.getOperation();
    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface", op->getName());

    const auto outputType = op->getResult(0).getType().cast<NDTypeInterface>();
    const auto outputShape = outputType.getShape();
    Shape nTilesOnDim(outputShape.size(), 1);
    const auto log = _log.nest();
    const auto tilingMode = TilingMode::ISOLATED;
    const auto tileDimOrder = getTileDimOrder(op, tilingMode, log);
    _log.nest(4).trace("Tile Dim order is {0}", tileDimOrder);

    const auto isSupportedTileSize = [&](ShapeRef nTilesOnDim, int32_t dimToTile) -> bool {
        const auto tiles = fillDividedTiles(op, nTilesOnDim, outputShape);
        for (auto tile : tiles) {
            if (tile.shape.raw()[dimToTile] > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
                return false;
            }
            auto inputTiling = origOp.backInferTileInfo(tile, log);
            auto& inTiles = inputTiling.tiles;
            if ((dimToTile != Dims4D::Act::C.ind()) &&
                (inTiles.begin()->shape.raw()[dimToTile] > VPU::NCEInvariant::VPU_DIMENSION_LIMIT)) {
                return false;
            }
        }
        return true;
    };

    for (auto tileDimIter = tileDimOrder.begin(); tileDimIter < tileDimOrder.end(); ++tileDimIter) {
        auto dimToTile = *tileDimIter;
        while (!isSupportedTileSize(nTilesOnDim, dimToTile.ind())) {
            ++nTilesOnDim[dimToTile];
        }
    }

    const auto tilesNew = fillDividedTiles(op, nTilesOnDim, outputShape);
    return VPU::applyTileStrategy(origOp, tilesNew, rewriter, log.nest());
}

//
// EnsureNCEOpsSizeRequirementsPass
//

class EnsureNCEOpsSizeRequirementsPass final :
        public VPU::EnsureNCEOpsSizeRequirementsBase<EnsureNCEOpsSizeRequirementsPass> {
public:
    explicit EnsureNCEOpsSizeRequirementsPass(bool allowLargeInputChannels, Logger log)
            : _allowLargeInputChannels(allowLargeInputChannels) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    bool _allowLargeInputChannels;
};

//
// safeRunOnFunc
//

void EnsureNCEOpsSizeRequirementsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    mlir::RewritePatternSet patterns(&ctx);
    target.addLegalOp<VPU::SliceOp, VPU::ConcatOp>();

    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (!mlir::isa<VPU::NCEOpInterface>(op)) {
            return true;
        }

        if (mlir::dyn_cast<VPU::TilingInfoOpInterface>(op)) {
            const auto inputShape = getShape(op->getOperand(0));
            const auto outputShape = getShape(op->getResult(0));

            auto inSizeWrongDims = getDimsOverHWLimit(inputShape);

            // TODO E64000 Support tiling over InputChannels
            const auto channelWrongDim = std::find(inSizeWrongDims.begin(), inSizeWrongDims.end(), Dims4D::Act::C);
            if (channelWrongDim != inSizeWrongDims.end()) {
                inSizeWrongDims.erase(channelWrongDim);
                if (!_allowLargeInputChannels) {
                    _log.warning("Tiling over input channels NOT SUPPORTED: {0} is bigger than {1}",
                                 inputShape[Dim(Dims4D::Act::C)], VPU::NCEInvariant::VPU_DIMENSION_LIMIT);
                }
            }

            if (!inSizeWrongDims.empty()) {
                _log.nest(2).info("Input size has dims greater than HW requirements: {0}", inSizeWrongDims);
            }
            const auto outSizeWrongDims = getDimsOverHWLimit(outputShape);
            if (!outSizeWrongDims.empty()) {
                _log.nest(2).info("Output size has dims greater than HW requirements: {0}", outSizeWrongDims);
            }
            return !(!inSizeWrongDims.empty() || !outSizeWrongDims.empty());
        }

        return true;
    });
    patterns.add<EnsureSizeRequirements>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(getFunction(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createEnsureNCEOpsSizeRequirementsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createEnsureNCEOpsSizeRequirementsPass(bool allowLargeInputChannels,
                                                                              Logger log) {
    return std::make_unique<EnsureNCEOpsSizeRequirementsPass>(allowLargeInputChannels, log);
}
