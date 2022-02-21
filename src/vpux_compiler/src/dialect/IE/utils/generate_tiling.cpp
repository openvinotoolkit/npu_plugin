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
#include "vpux/compiler/core/tiling.hpp"

namespace vpux {
namespace IE {

OutputTiling getTilingStrategy(mlir::Operation* op, Logger log, TilingMode tilingMode) {
    auto tilingInfo = mlir::dyn_cast<IE::TilingInfoOpInterface>(op);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface", op->getName());
    auto tilingBuilder = mlir::dyn_cast<IE::TilingBuilderOpInterface>(op);
    VPUX_THROW_WHEN(tilingBuilder == nullptr, "Operation '{0}' doesn't implement TilingBuilderOpInterface",
                    op->getName());

    const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputShape = outputType.getShape();

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
    if (tilingMode == TilingMode::PATTERN_PREFETCH_TILING) {
        tileDimOrder = (outputShape[Dims4D::Act::C] < outputShape[Dims4D::Act::H]) ? SmallVector<Dim>({Dims4D::Act::H})
                                                                                   : SmallVector<Dim>({Dims4D::Act::C});
    }

    auto tileDimIter = tileDimOrder.begin();
    auto dimToTile = *tileDimIter;

    const auto isSupportedTileSize = [&tilingInfo, outputShape, log](ShapeRef nTilesOnDim,
                                                                     TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(nTilesOnDim, outputShape);
        return tilingInfo.isSupportedTiling(tiles, log, tilingMode);
    };

    const auto isSupportedChannelDivision = [&](ShapeRef tileDim) {
        if ((outputShape[Dims4D::Act::C] % tileDim[Dims4D::Act::C]) != 0) {
            return false;
        }
        const auto tileChannels = outputShape[Dims4D::Act::C] / nTilesOnDim[Dims4D::Act::C];
        return (tileChannels % minChannelSize) == 0;
    };

    const auto& maxNumTiles = tilingBuilder.getMaxNumTiles();
    const auto isDimLeftToTile = [&](ShapeRef tileShape) -> bool {
        return tileShape[dimToTile] < maxNumTiles[dimToTile.ind()];
    };

    const auto tilingModeToCheck = tilingMode == TilingMode::PREFETCH_TILING ? TilingMode::ISOLATED_TILING : tilingMode;
    while (!isSupportedTileSize(nTilesOnDim, tilingModeToCheck)) {
        if (!isDimLeftToTile(nTilesOnDim)) {
            dimToTile = *(++tileDimIter);
        }
        if (tileDimIter == tileDimOrder.end()) {
            VPUX_THROW_WHEN(tilingModeToCheck == TilingMode::ISOLATED_TILING, "Failed to tile {0} at '{1}'",
                            op->getName(), op->getLoc());
            // If still not find the tiling strategy in PATTERN_PREFETCH_TILING, fall back to neutral tiling
            return fillDividedTiles(Shape(outputShape.size(), 1), outputShape);
        }

        if (dimToTile == Dims4D::Act::C) {
            do {
                ++nTilesOnDim[Dims4D::Act::C];
            } while (!isSupportedChannelDivision(nTilesOnDim));
        } else if (dimToTile == Dims4D::Act::H || dimToTile == Dims4D::Act::W) {
            nTilesOnDim[dimToTile]++;
        } else {
            // Trying to tile in unsupported dimension, tiling in supported dimensions not sufficient
            VPUX_THROW("Failed to tile {0} at '{1}'", op->getName(), op->getLoc());
        }
    }
    auto origTiles = fillDividedTiles(nTilesOnDim, outputShape);

    if (tilingMode != TilingMode::PREFETCH_TILING) {
        return origTiles;
    }

    auto getDimsToTile = [](const Shape& nTilesOnDim) -> SmallVector<Dim> {
        SmallVector<Dim> res;
        for (unsigned i = 0; i < nTilesOnDim.size(); i++) {
            if (nTilesOnDim[Dim(i)] > 1)
                res.emplace_back(Dim(i));
        }
        return res;
    };
    auto dimsToTile = getDimsToTile(nTilesOnDim);
    if (dimsToTile.size() > 1) {
        // return isolated tiling when getting nested tiles.
        return origTiles;
    }

    // Prefetch Tiling

    const auto targetDim = dimsToTile[0];
    Shape prefetchableTilesOnDim = nTilesOnDim;
    while (!isSupportedTileSize(prefetchableTilesOnDim, TilingMode::PREFETCH_TILING)) {
        if (prefetchableTilesOnDim[targetDim] >= IE::MAX_PREFETCH_TILING_TIME * nTilesOnDim[targetDim] ||
            !isDimLeftToTile(prefetchableTilesOnDim)) {
            return origTiles;
        }
        if (targetDim == Dims4D::Act::C) {
            do {
                ++prefetchableTilesOnDim[Dims4D::Act::C];
            } while (!isSupportedChannelDivision(prefetchableTilesOnDim));
        } else {
            prefetchableTilesOnDim[dimToTile]++;
        }
    }
    return fillDividedTiles(prefetchableTilesOnDim, outputShape);
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

    const auto baseResType = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto tiledResType = baseResType.extractDenseTile(outputTile.offsets, outputTile.shape);

    auto tiledRes = tiledOp->getResult(0);
    tiledRes.setType(tiledResType);

    return tiledRes;
}

mlir::LogicalResult applyTileStrategy(IE::TilingBuilderOpInterface origOp, OutputTiling tiles,
                                      mlir::PatternRewriter& rewriter, Logger log) {
    // apply the generated tiling strategy and create tiled operations
    // insert the tiled pattern with a concat to the IR
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

    rewriter.replaceOpWithNewOp<IE::ConcatOp>(origOp, origOp->getResult(0).getType(), mlir::ValueRange(resultTileVals),
                                              makeArrayRef(resultTileOffsets));

    // update concat users and also place correctly in the IR
    for (auto* concatOp : resultTileVals[0].getUsers()) {
        if (!mlir::isa<IE::ConcatOp>(concatOp)) {
            continue;
        }
        origOp->replaceAllUsesWith(concatOp);
        for (auto concatConsumer : concatOp->getResult(0).getUsers()) {
            if (concatOp->isBeforeInBlock(concatConsumer)) {
                continue;
            }
            concatOp->moveBefore(concatConsumer);
            // also move the Slice+Conv pattern, first conv, then slice
            for (auto concatOperand : concatOp->getOperands()) {
                auto concatProducer = concatOperand.getDefiningOp();
                if (concatProducer->isBeforeInBlock(concatOp)) {
                    continue;
                }
                concatProducer->moveBefore(concatOp);
                auto sliceOp = concatProducer->getOperand(0).getDefiningOp();
                for (auto sliceOperand : concatProducer->getOperands()) {
                    if (mlir::isa<IE::SliceOp>(sliceOperand.getDefiningOp())) {
                        sliceOp = sliceOperand.getDefiningOp();
                        break;
                    }
                }
                if (!mlir::isa<IE::SliceOp>(sliceOp)) {
                    continue;
                }
                sliceOp->moveBefore(concatProducer);
            }
        }
        break;
    }

    return mlir::success();
}

mlir::Operation* getParentTargetOp(mlir::Operation* op) {
    // For const prefetch ignore cases where activation is handled by
    // intermediate operations and causes a stall
    // Prefetch is wanted from current op to previous op which is not ignored.
    mlir::Operation* parentOp = op->getOperand(0).getDefiningOp();
    auto isOpIgnorable = [](mlir::Operation* op) -> bool {
        // These operations has little memory/DPU occupation
        // The current operation can be prefetched ignoring these operations
        return mlir::isa<IE::AndOp>(op) || mlir::isa<IE::PermuteCastOp>(op) || mlir::isa<IE::ReshapeOp>(op);
    };
    while (parentOp && isOpIgnorable(parentOp)) {
        // skip the Permute, Reshape and And
        if (parentOp->getOperands().size() < 1) {
            return nullptr;
        }
        parentOp = parentOp->getOperand(0).getDefiningOp();
    }
    return parentOp;
}

bool prefetchTilingConditionSatisfied(mlir::Operation* op, Logger log) {
    auto parentOp = getParentTargetOp(op);
    if (parentOp == nullptr) {
        return false;
    }
    auto opTilingInter = mlir::dyn_cast<IE::TilingInfoOpInterface>(op);
    auto parentTilingInter = mlir::dyn_cast<IE::TilingInfoOpInterface>(parentOp);
    if (!opTilingInter || !parentTilingInter) {
        return false;
    }
    // For parallel sub-graphs, the order is undecided yet
    // Abandon prefetching these cases
    if (!parentOp->getResult(0).hasOneUse()) {
        auto user1 = *parentOp->getResult(0).getUsers().begin();
        for (auto remainUser : parentOp->getResult(0).getUsers()) {
            if (remainUser != user1) {
                return false;
            }
        }
    }

    // Check if tile pattern is supported
    const auto resShape = getShape(op->getResult(0));
    const Shape neutralTile(resShape.size(), 1);
    if (opTilingInter.isSupportedTiling(fillDividedTiles(neutralTile, resShape), log,
                                        TilingMode::PATTERN_PREFETCH_TILING)) {
        return false;
    }
    // Try to tile to satisfy prefetching
    auto tiles = getTilingStrategy(op, log.nest(), TilingMode::PATTERN_PREFETCH_TILING);
    return tiles.begin()->axis != neutralTile;
}
}  // namespace IE
}  // namespace vpux
