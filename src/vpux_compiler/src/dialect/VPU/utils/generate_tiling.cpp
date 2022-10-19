//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"

namespace vpux {
namespace VPU {

void storeTilingStrategyForOp(mlir::Operation* op, ShapeRef nTilesOnDim) {
    const auto tilesAttr = getIntArrayAttr(op->getContext(), nTilesOnDim);
    op->setAttr(tilingStrategy, tilesAttr);
}

OutputTiling getTilingStrategy(mlir::Operation* op, Logger log, TilingMode tilingMode) {
    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface", op->getName());
    auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
    VPUX_THROW_WHEN(tilingBuilder == nullptr, "Operation '{0}' doesn't implement TilingBuilderOpInterface",
                    op->getName());

    const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputShape = outputType.getShape();

    VPUX_THROW_UNLESS(outputShape.size() == 4, "Unsupported operation '{0}' at '{1}', it has non 4D result",
                      op->getName(), op->getLoc());

    int64_t outputChannelAlignment = 1;
    if (auto channelsInfo = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
        outputChannelAlignment = channelsInfo.getOutputChannelAlignment();
    }

    Shape nTilesOnDim(outputShape.size(), 1);

    // Try to tile the largest dim (C or H) first, then proceed with other dims
    SmallVector<Dim> tileDimOrder = {Dims4D::Act::C, Dims4D::Act::H, Dims4D::Act::W};
    if (outputShape[Dims4D::Act::C] < outputShape[Dims4D::Act::H]) {
        tileDimOrder = {Dims4D::Act::H, Dims4D::Act::C, Dims4D::Act::W};
    }
    if (tilingMode == TilingMode::PREFETCHING) {
        tileDimOrder = (outputShape[Dims4D::Act::C] < outputShape[Dims4D::Act::H]) ? SmallVector<Dim>({Dims4D::Act::H})
                                                                                   : SmallVector<Dim>({Dims4D::Act::C});
    }

    auto tileDimIter = tileDimOrder.begin();
    auto dimToTile = *tileDimIter;

    const auto isSupportedTileSize = [op, &tilingInfo, outputShape, log](ShapeRef nTilesOnDim,
                                                                         TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(op, nTilesOnDim, outputShape);
        return tilingInfo.isSupportedTiling(tiles, tilingMode, log);
    };

    // Allow uneven tiling over OC, such as OC = 80 can be tiled as three tiles [32, 32, 16]
    const auto isSupportedChannelDivision = [&](ShapeRef tileDim) {
        auto base = vpux::divUp(outputShape[Dims4D::Act::C], tileDim[Dims4D::Act::C]);
        auto alignedBase = vpux::alignShape({base}, {outputChannelAlignment})[0];
        auto remainder = outputShape[Dims4D::Act::C] - alignedBase * (tileDim[Dims4D::Act::C] - 1);
        return remainder > 0;
    };

    const auto& maxNumTiles = tilingBuilder.getMaxNumTiles();
    const auto isDimLeftToTile = [&](ShapeRef tileShape) -> bool {
        return tileShape[dimToTile] < maxNumTiles[dimToTile.ind()];
    };

    // In case of pipelining, an isolated tiling strategy is first created
    // Then the tiling number would be increased to get a pipelining tiling strategy
    // If no feasible pipelining tiling could be found, fallback to isolated tiling strategy
    const auto tilingModeToCheck = tilingMode == TilingMode::PIPELINING ? TilingMode::ISOLATED : tilingMode;
    // Step1. get an feasible isolated tiling strategy or prefetching strategy
    while (!isSupportedTileSize(nTilesOnDim, tilingModeToCheck)) {
        while ((tileDimIter < tileDimOrder.end()) && (!isDimLeftToTile(nTilesOnDim))) {
            dimToTile = *(++tileDimIter);
        }
        if (tileDimIter == tileDimOrder.end()) {
            VPUX_THROW_WHEN(tilingModeToCheck == TilingMode::ISOLATED, "Failed to tile {0} at '{1}'", op->getName(),
                            op->getLoc());
            // If still not find the tiling strategy in PREFETCHING, fall back to neutral tiling
            return fillDividedTiles(op, Shape(outputShape.size(), 1), outputShape);
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
    auto origTiles = fillDividedTiles(op, nTilesOnDim, outputShape);

    if (tilingMode != TilingMode::PIPELINING) {
        storeTilingStrategyForOp(op, nTilesOnDim);
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
        storeTilingStrategyForOp(op, nTilesOnDim);
        return origTiles;
    }

    // Step2. For pipelining, continue to increase on the dimension of isolated tiling
    //        or on the channel dimension in case of neutral tiling to cover cases with large constants
    const auto targetDim = dimsToTile.size() == 0 ? Dims4D::Act::C : dimsToTile[0];
    Shape prefetchableTilesOnDim = nTilesOnDim;
    log.nest(2).trace("pipelining tiling strategy generating...");
    while (!isSupportedTileSize(prefetchableTilesOnDim, TilingMode::PIPELINING)) {
        if (prefetchableTilesOnDim[targetDim] >= VPU::MAX_PREFETCH_TILING_TIME * nTilesOnDim[targetDim] ||
            !isDimLeftToTile(prefetchableTilesOnDim)) {
            storeTilingStrategyForOp(op, nTilesOnDim);
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

    log.nest(2).trace("pipelining tiling strategy get: {0}", prefetchableTilesOnDim);
    storeTilingStrategyForOp(op, prefetchableTilesOnDim);
    return fillDividedTiles(op, prefetchableTilesOnDim, outputShape);
}

mlir::Value reifyTile(VPU::TilingBuilderOpInterface origOp, const TileInfo& outputTile, mlir::OpBuilder& builder,
                      Logger log) {
    log.nest(2).trace("{0}", outputTile);

    const auto inputTiling = origOp.backInferTileInfo(outputTile, log);
    const auto& inTiles = inputTiling.tiles;
    VPUX_THROW_UNLESS(!inTiles.empty(), "Got empty tile information");

    mlir::BlockAndValueMapping mapper;
    for (auto& p : origOp->getOperands() | indexed) {
        auto origInput = p.value();
        auto inputIdx = p.index();

        const auto valName = printToString("input {0}", inputIdx);
        const auto tiledInput = vpux::VPU::makeTile(builder, origOp->getLoc(), origInput, inTiles[inputIdx], valName);

        mapper.map(origInput, tiledInput);
    }

    const auto tileLoc = appendLoc(origOp->getLoc(), "output tile {0}", outputTile.offsets);

    auto* tiledOp = builder.clone(*origOp, mapper);
    tiledOp->setLoc(tileLoc);

    auto tiledBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(tiledOp);
    VPUX_THROW_WHEN(tiledBuilderOp == nullptr, "Operation '{0}' doesn't implement TilingBuilderOpInterface",
                    tiledBuilderOp->getName());

    tiledBuilderOp.adjustAttrs(inputTiling, outputTile);

    const auto baseResType = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto tiledResType = baseResType.extractDenseTile(outputTile.offsets, outputTile.shape);

    auto tiledRes = tiledOp->getResult(0);
    tiledRes.setType(tiledResType);

    return tiledRes;
}

mlir::LogicalResult applyTileStrategy(VPU::TilingBuilderOpInterface origOp, OutputTiling tiles,
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

    rewriter.replaceOpWithNewOp<VPU::ConcatOp>(origOp, origOp->getResult(0).getType(), mlir::ValueRange(resultTileVals),
                                               makeArrayRef(resultTileOffsets));

    // update concat users and also place correctly in the IR
    for (auto* concatOp : resultTileVals[0].getUsers()) {
        if (!mlir::isa<VPU::ConcatOp>(concatOp)) {
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
                    if (mlir::isa<VPU::SliceOp>(sliceOperand.getDefiningOp())) {
                        sliceOp = sliceOperand.getDefiningOp();
                        break;
                    }
                }
                if (!mlir::isa<VPU::SliceOp>(sliceOp)) {
                    continue;
                }
                sliceOp->moveBefore(concatProducer);
            }
        }
        break;
    }

    return mlir::success();
}

bool hasMultiBranches(mlir::Operation* op) {
    // not the only result
    if (op->getResults().size() != 1) {
        return true;
    }
    // only one result but multiple users
    if (!op->getResult(0).hasOneUse()) {
        auto user1 = op->getResult(0).user_begin();
        for (auto remainUser : op->getResult(0).getUsers()) {
            if (remainUser != *user1) {
                return true;
            }
        }
    }
    return false;
}

mlir::Operation* getParentTargetOp(mlir::Operation* op) {
    // for const prefetch ignore cases where activation is handled by
    // intermediate operations and causes a stall
    // prefetch is wanted from current op to parent op
    const auto isOpIgnorable = [](mlir::Operation* op) -> bool {
        if (auto nceEltwiseAnd = mlir::dyn_cast<VPU::NCEEltwiseOp>(op)) {
            return nceEltwiseAnd.op_type() == VPU::EltwiseType::AND;
        }
        return mlir::isa<IE::AndOp>(op) || mlir::isa<VPU::AndOp>(op) || mlir::isa<VPU::PermuteCastOp>(op) ||
               mlir::isa<VPU::ReshapeOp>(op);
    };

    mlir::Operation* parentOp = op->getOperand(0).getDefiningOp();
    while (parentOp && isOpIgnorable(parentOp)) {
        // skip the Permute, Reshape and And
        if (parentOp->getOperands().size() < 1) {
            break;
        }
        if (hasMultiBranches(parentOp)) {
            // for parallel sub-graphs, the order is undecided yet
            // abandon prefetching these cases
            return nullptr;
        }
        parentOp = parentOp->getOperand(0).getDefiningOp();
    }
    // check the last op
    return (parentOp == nullptr || hasMultiBranches(parentOp)) ? nullptr : parentOp;
}

bool prefetchTilingConditionSatisfied(mlir::Operation* op, Logger log) {
    auto parentOp = getParentTargetOp(op);
    if (parentOp == nullptr) {
        return false;
    }
    const auto arch = VPU::getArch(op);
    if ((arch == VPU::ArchKind::VPUX30XX) || (arch == VPU::ArchKind::VPUX311X)) {
        if (!mlir::isa<VPU::NCEOpInterface>(parentOp)) {
            return false;
        }
    }
    auto opTilingInter = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
    auto parentTilingInter = mlir::dyn_cast<VPU::TilingInfoOpInterface>(parentOp);
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
    if (opTilingInter.isSupportedTiling(fillDividedTiles(op, neutralTile, resShape), TilingMode::PREFETCHING, log)) {
        return false;
    }
    // Try to tile to satisfy prefetching
    auto tiles = getTilingStrategy(op, log.nest(), TilingMode::PREFETCHING);
    return tiles.begin()->axis != neutralTile;
}

bool isLargeConstOp(mlir::Operation* op, Logger log) {
    // The operation should have constant filter
    if (!mlir::isa<VPU::NCEConvolutionOp>(op) && !mlir::isa<VPU::NCEDepthConvolutionOp>(op)) {
        return false;
    }
    auto filter = op->getOperand(1).getDefiningOp<Const::DeclareOp>();
    if (filter == nullptr) {
        return false;
    }

    Byte filterSize(0);
    auto filterType = filter.output().getType().cast<vpux::NDTypeInterface>();
    if (op->hasAttr(multiClusterStrategy)) {
        auto nceOp = mlir::cast<VPU::NCEOpInterface>(op);
        auto numClusters = VPU::getOptimalNumClusters(
                nceOp, filterType.getShape()[Dims4D::Filter::OC],
                nceOp->getAttr(VPU::multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue());
        auto filterDistributedType = VPU::getDistributedFilterTypeFromOp(nceOp, filterType, numClusters);
        filterSize = filterDistributedType.getTotalAllocSize();
    } else {
        filterSize = filterType.getTotalAllocSize();
    }

    auto cmxThreshold = Byte(static_cast<int64_t>(
            std::ceil(static_cast<double>(VPU::getTotalCMXSize(op).count()) * VPU::LARGE_CONST_THRESHOLD_RATIO)));
    if (filterSize > cmxThreshold) {
        log.nest(1).trace("filter size {0} is larger than cmxThreshold {1}", filterSize, cmxThreshold);
        return true;
    }
    return false;
}

bool largeConstPipelineConditionSatisfied(mlir::Operation* op, Logger log) {
    // Check if the operation has large constant filter
    if (!isLargeConstOp(op, log)) {
        return false;
    }

    // Find the available tiling size over C
    // The pipelining should be doable with this tiling size
    auto tiles = getTilingStrategy(op, log, TilingMode::PIPELINING);
    if (tiles.begin()->axis != Shape(getShape(op->getResult(0)).size(), 1)) {
        log.nest(1).trace("Found pipelining tiling strategy {0}", tiles.begin()->axis);
        return true;
    }

    return false;
}

bool archSupportsSwLayerTiling(VPU::ArchKind arch) {
    return arch == VPU::ArchKind::VPUX37XX || arch == VPU::ArchKind::VPUX40XX;
}

}  // namespace VPU
}  // namespace vpux
