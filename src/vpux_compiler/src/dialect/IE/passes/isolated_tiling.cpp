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

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/FunctionExtras.h>

#include <numeric>
#include <vpux/compiler/conversion.hpp>

using namespace vpux;

namespace {

void applyWAForGroupConv(mlir::Operation* op, Shape& nTilesOnDim) {
    auto groupCovn = mlir::dyn_cast<IE::GroupConvolutionOp>(op);
    if (groupCovn == nullptr) {
        return;
    }

    const auto module = op->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);
    if (arch == VPU::ArchKind::MTL) {
        return;
    }

    if (auto channelsInfo = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
        const auto chanAlignment = channelsInfo.getChannelAlignment();
        const auto outputShape = getShape(groupCovn.output());

        VPUX_THROW_UNLESS(outputShape[Dims4D::Act::C] % chanAlignment == 0,
                          "Depthwise convolution output channels must be a multiple of {0}, got {1}", chanAlignment,
                          outputShape[Dims4D::Act::C]);

        nTilesOnDim[Dims4D::Act::C] = outputShape[Dims4D::Act::C] / chanAlignment;
    }
}

OutputTiling generateTiles(IE::TilingBuilderOpInterface origOp, Logger log) {
    auto* op = origOp.getOperation();

    auto tilingInfo = mlir::dyn_cast<IE::TilingInfoOpInterface>(op);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface",
                    origOp->getName());

    const auto outputType = origOp->getResult(0).getType().cast<mlir::ShapedType>();
    const auto outputShape = getShape(outputType);

    VPUX_THROW_UNLESS(outputShape.size() == 4, "Unsupported operation '{0}' at '{1}', it has non 4D result",
                      origOp->getName(), origOp->getLoc());

    int64_t minChannelSize = 1;
    if (auto channelsInfo = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
        minChannelSize = channelsInfo.getChannelAlignment();
    }

    Shape nTilesOnDim(outputShape.size(), 1);
    applyWAForGroupConv(op, nTilesOnDim);
    const auto isSupportedChannelDivision = [&]() {
        if ((outputShape[Dims4D::Act::C] % nTilesOnDim[Dims4D::Act::C]) != 0) {
            return false;
        }

        const auto tileChannels = outputShape[Dims4D::Act::C] / nTilesOnDim[Dims4D::Act::C];
        return (tileChannels % minChannelSize) == 0;
    };

    // Try to tile the largest dim (C or H) first, then proceed with other dims
    SmallVector<Dim> tileDimOrder = {Dims4D::Act::C, Dims4D::Act::H, Dims4D::Act::W};
    if (outputShape[Dims4D::Act::C] < outputShape[Dims4D::Act::H]) {
        tileDimOrder = {Dims4D::Act::H, Dims4D::Act::C, Dims4D::Act::W};
    }

    auto tileDimIter = tileDimOrder.begin();
    auto dimToTile = *tileDimIter;

    const auto& maxNumTiles = origOp.getMaxNumTiles();
    const auto isDimLeftToTile = [&]() {
        return nTilesOnDim[dimToTile] < maxNumTiles[dimToTile.ind()];
    };

    const auto isSupportedTileSize = [&tilingInfo, outputShape, log](ShapeRef nTilesOnDim) -> bool {
        const auto tiles = vpux::fillDividedTiles(nTilesOnDim, outputShape);
        return tilingInfo.isSupportedTiling(tiles, log);
    };

    while (!isSupportedTileSize(nTilesOnDim)) {
        VPUX_THROW_WHEN(tileDimIter == tileDimOrder.end(), "Failed to tile {0} at '{1}'", origOp->getName(),
                        origOp->getLoc());

        if (!isDimLeftToTile()) {
            dimToTile = *(++tileDimIter);
        }

        if (dimToTile == Dims4D::Act::C) {
            do {
                ++nTilesOnDim[Dims4D::Act::C];
            } while (!isSupportedChannelDivision());
        } else if (dimToTile == Dims4D::Act::H || dimToTile == Dims4D::Act::W) {
            nTilesOnDim[dimToTile]++;
        } else {
            // Trying to tile in unsupported dimension, tiling in supported dimensions not sufficient
            VPUX_THROW("Failed to tile {0} at '{1}'", origOp->getName(), origOp->getLoc());
        }
    }

    return vpux::fillDividedTiles(nTilesOnDim, outputShape);
}

mlir::Value reifyTile(IE::TilingBuilderOpInterface origOp, const TileInfo& outputTile, mlir::OpBuilder& builder,
                      Logger log) {
    log.nest(2).trace("{0}", outputTile);

    const auto inputTiling = origOp.backInferTileInfo(outputTile);
    const auto& inTiles = inputTiling.tiles;
    VPUX_THROW_UNLESS(!inTiles.empty(), "Got empty tile information");

    mlir::BlockAndValueMapping mapper;
    for (auto& p : origOp->getOperands() | indexed) {
        auto origInput = p.value();
        auto inputIdx = p.index();

        const auto valName = llvm::formatv("input {0}", inputIdx).str();
        const auto tiledInput = vpux::IE::makeTile(builder, origOp->getLoc(), origInput, inTiles[inputIdx], valName);

        mapper.map(origInput, tiledInput);
    }

    const auto tileName = llvm::formatv("output tile {0}", outputTile.offsets).str();
    const auto tileLoc = appendLoc(origOp->getLoc(), tileName);

    auto* tiledOp = builder.clone(*origOp, mapper);
    tiledOp->setLoc(tileLoc);

    auto tiledBuilderOp = mlir::dyn_cast<IE::TilingBuilderOpInterface>(tiledOp);
    VPUX_THROW_WHEN(tiledBuilderOp == nullptr, "Operation '{0}' doesn't implement TilingBuilderOpInterface",
                    tiledBuilderOp->getName());

    tiledBuilderOp.adjustAttrs(inputTiling);

    VPUX_THROW_UNLESS(origOp->getNumResults() == 1,
                      "Unsupported operation '{0}' at '{1}', it must have one and only one result", origOp->getName(),
                      origOp->getLoc());

    const auto baseResType = origOp->getResult(0).getType().cast<mlir::ShapedType>();
    const auto tiledResType = getDenseTileType(baseResType, outputTile.offsets, outputTile.shape);

    auto tiledRes = tiledOp->getResult(0);
    tiledRes.setType(tiledResType);

    return tiledRes;
}

//
// GenericTiling
//

class GenericTiling final : public mlir::OpInterfaceRewritePattern<IE::TilingBuilderOpInterface> {
public:
    GenericTiling(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<IE::TilingBuilderOpInterface>(ctx), _log(log) {
        this->setDebugName("GenericTiling");
    }

    mlir::LogicalResult matchAndRewrite(IE::TilingBuilderOpInterface origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult GenericTiling::matchAndRewrite(IE::TilingBuilderOpInterface origOp,
                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    VPUX_THROW_UNLESS(origOp->getNumResults() == 1,
                      "Unsupported operation '{0}' at '{1}', it must have one and only one result", origOp->getName(),
                      origOp->getLoc());

    const auto tiles = generateTiles(origOp, _log);
    _log.nest(1).trace("Create {0} tiles:", tiles.size());

    SmallVector<mlir::Value> resultTileVals;
    SmallVector<ShapeRef> resultTileOffsets;

    resultTileVals.reserve(tiles.size());
    resultTileOffsets.reserve(tiles.size());

    for (const auto& outputTile : tiles) {
        const auto tiledRes = reifyTile(origOp, outputTile, rewriter, _log);

        const auto tiledShape = getShape(tiledRes);
        VPUX_THROW_UNLESS(tiledShape == outputTile.shape,
                          "Inferred tiled output shape '{0}' doesn't match with generated '{1}'", tiledShape,
                          outputTile.shape);

        resultTileVals.push_back(tiledRes);
        resultTileOffsets.push_back(outputTile.offsets);
    }

    rewriter.replaceOpWithNewOp<IE::ConcatOp>(origOp, origOp->getResult(0).getType(), mlir::ValueRange(resultTileVals),
                                              makeArrayRef(resultTileOffsets));
    return mlir::success();
}

//
// IsolatedTilingPass
//

class IsolatedTilingPass final : public IE::IsolatedTilingBase<IsolatedTilingPass> {
public:
    explicit IsolatedTilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void IsolatedTilingPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addLegalOp<IE::SliceOp, IE::ConcatOp>();
    target.markUnknownOpDynamicallyLegal([this](mlir::Operation* op) {
        if (auto iface = mlir::dyn_cast<IE::TilingInfoOpInterface>(op)) {
            return !iface.needTiling(_log.nest());
        }

        return true;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GenericTiling>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(getFunction(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createIsolatedTilingPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createIsolatedTilingPass(Logger log) {
    return std::make_unique<IsolatedTilingPass>(log);
}
