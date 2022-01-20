//
// Copyright 2021 Intel Corporation.
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

#include "vpux/compiler/dialect/IE/utils/generate_tiling.hpp"

namespace vpux {
namespace IE {

Shape computeGeneralTileStrategy(mlir::Operation* op, Logger log) {
    auto tilingInfo = mlir::dyn_cast<IE::TilingInfoOpInterface>(op);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface", op->getName());
    auto tilingBuilder = mlir::dyn_cast<IE::TilingBuilderOpInterface>(op);
    VPUX_THROW_WHEN(tilingBuilder == nullptr, "Operation '{0}' doesn't implement TilingBuilderOpInterface",
                    op->getName());

    const auto outputType = op->getResult(0).getType().cast<mlir::ShapedType>();
    const auto outputShape = getShape(outputType);

    VPUX_THROW_UNLESS(outputShape.size() == 4, "Unsupported operation '{0}' at '{1}', it has non 4D result",
                      op->getName(), op->getLoc());

    int64_t minChannelSize = 1;
    if (auto channelsInfo = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
        minChannelSize = channelsInfo.getOutputChannelAlignment();
    }

    Shape nTilesOnDim(outputShape.size(), 1);

    // Try to tile the largest dim (C or H) first, then proceed with other dims
    SmallVector<Dim> tileDimOrder = {Dims4D::Act::C, Dims4D::Act::H, Dims4D::Act::W};
    if (outputShape[Dims4D::Act::C] < outputShape[Dims4D::Act::H]) {
        tileDimOrder = {Dims4D::Act::H, Dims4D::Act::C, Dims4D::Act::W};
    }

    auto tileDimIter = tileDimOrder.begin();
    auto dimToTile = *tileDimIter;

    const auto isSupportedTileSize = [&tilingInfo, outputShape, log](ShapeRef nTilesOnDim) -> bool {
        const auto tiles = fillDividedTiles(nTilesOnDim, outputShape);
        return tilingInfo.isSupportedTiling(tiles, log);
    };

    const auto isSupportedChannelDivision = [&]() {
        if ((outputShape[Dims4D::Act::C] % nTilesOnDim[Dims4D::Act::C]) != 0) {
            return false;
        }
        const auto tileChannels = outputShape[Dims4D::Act::C] / nTilesOnDim[Dims4D::Act::C];
        return (tileChannels % minChannelSize) == 0;
    };

    const auto& maxNumTiles = tilingBuilder.getMaxNumTiles();
    const auto isDimLeftToTile = [&]() {
        return nTilesOnDim[dimToTile] < maxNumTiles[dimToTile.ind()];
    };

    while (!isSupportedTileSize(nTilesOnDim)) {
        VPUX_THROW_WHEN(tileDimIter == tileDimOrder.end(), "Failed to tile {0} at '{1}'", op->getName(), op->getLoc());

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
            VPUX_THROW("Failed to tile {0} at '{1}'", op->getName(), op->getLoc());
        }
    }
    return nTilesOnDim;
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

    const auto baseResType = origOp->getResult(0).getType().cast<mlir::ShapedType>();
    const auto tiledResType = getDenseTileType(baseResType, outputTile.offsets, outputTile.shape);

    auto tiledRes = tiledOp->getResult(0);
    tiledRes.setType(tiledResType);

    return tiledRes;
}

mlir::LogicalResult applyTileStrategy(IE::TilingBuilderOpInterface origOp, OutputTiling tiles,
                                      mlir::PatternRewriter& rewriter, Logger log) {
    SmallVector<mlir::Value> resultTileVals;
    SmallVector<ShapeRef> resultTileOffsets;

    resultTileVals.reserve(tiles.size());
    resultTileOffsets.reserve(tiles.size());

    for (const auto& outputTile : tiles) {
        const auto tiledRes = reifyTile(origOp, outputTile, rewriter, log);

        const auto tiledShape = getShape(tiledRes);
        VPUX_THROW_UNLESS(tiledShape == outputTile.shape,
                          "Inferred tiled output shape '{0}' doesn't match with generated '{1}'", tiledShape,
                          outputTile.shape);

        resultTileVals.push_back(tiledRes);
        resultTileOffsets.push_back(outputTile.offsets);
    }

    auto newConcat = rewriter.replaceOpWithNewOp<IE::ConcatOp>(origOp, origOp->getResult(0).getType(), mlir::ValueRange(resultTileVals),
                                              makeArrayRef(resultTileOffsets));

    // TODO: remove with EISW-25333
    newConcat->setAttr("CMXConcat", rewriter.getBoolAttr(false));
    return mlir::success();
    
    for (auto concatOp : resultTileVals[0].getUsers()) {
        if (!mlir::isa<IE::ConcatOp>(*concatOp)) {
            continue;
        }
        origOp->replaceAllUsesWith(concatOp);
        break;
    }
    return mlir::success();
}
}  // namespace IE
}  // namespace vpux
