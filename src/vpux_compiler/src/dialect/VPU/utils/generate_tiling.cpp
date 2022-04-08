//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
#include <llvm/ADT/TypeSwitch.h>

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/strategy_manager.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"

namespace vpux {
namespace VPU {

void storeTilingStrategyForOp(mlir::Operation* op, ShapeRef nTilesOnDim) {
    const auto tilesAttr = getIntArrayAttr(op->getContext(), nTilesOnDim);
    op->setAttr(tilingStrategy, tilesAttr);
}

mlir::LogicalResult checkAndAlignActInputTiling(vpux::VPU::NCEOpInterface nceOp, InputTiling& inputTiling,
                                                vpux::Logger log) {
    auto origInputType = nceOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    auto tiledInputType = origInputType.extractDenseTile(inputTiling.tiles[0].offsets, inputTiling.tiles[0].shape);
    if (mlir::succeeded(nceOp.verifyInputType(tiledInputType))) {
        return mlir::success();
    }
    log.trace("Inferred activation input tiling {0} is invalid for {1}", inputTiling.tiles[0], nceOp->getName());
    auto stride = nceOp.getStrides()[Dims4D::Strides::X.ind()];  // get W side strides
    int64_t bias = 0;
    auto newInputActTiling = inputTiling.tiles[0];
    while (++bias < stride) {
        auto alignedShape =
                Shape({inputTiling.tiles[0].shape[Dims4D::Act::N], inputTiling.tiles[0].shape[Dims4D::Act::C],
                       inputTiling.tiles[0].shape[Dims4D::Act::H], inputTiling.tiles[0].shape[Dims4D::Act::W] + bias});
        newInputActTiling = TileInfo(alignedShape, inputTiling.tiles[0].offsets, inputTiling.tiles[0].axis);
        auto newInputActType = origInputType.extractDenseTile(newInputActTiling.offsets, newInputActTiling.shape);
        if (mlir::succeeded(nceOp.verifyInputType(newInputActType))) {
            inputTiling.tiles[0] = newInputActTiling;
            log.trace("Input tiling is corrected to {0}", inputTiling.tiles[0]);
            return mlir::success();
        }
    }
    VPUX_THROW("Cannot find aligned act input tiling for op {0} at {1}", nceOp->getName(), nceOp->getLoc());
}

// Temporari solution until E#59988 will be implemented.
// Function can be deleted when E#59993 related interface will be added.
SmallVector<mlir::Value> reifyTileTopK(VPU::TilingBuilderOpInterface origOp, const TileInfo& outputTile,
                                       mlir::OpBuilder& builder, Logger log) {
    log = log.nest(2);
    log.trace("{0}", outputTile);

    auto inputTiling = origOp.backInferTileInfo(outputTile, log);
    auto& inTiles = inputTiling.tiles;
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

    SmallVector<mlir::Value> ret;
    for (auto& p : origOp->getResults() | indexed) {
        auto idx = p.index();
        const auto baseResType = origOp->getResult(idx).getType().cast<vpux::NDTypeInterface>();
        const auto tiledResType = baseResType.extractDenseTile(outputTile.offsets, outputTile.shape);
        auto tiledRes = tiledOp->getResult(idx);
        tiledRes.setType(tiledResType);
        ret.push_back(tiledRes);
    }
    return ret;
}

// Temporari solution until E#59988 will be implemented.
// Function can be deleted when E#59993 related interface will be added.
mlir::LogicalResult applyTileStrategyTopK(VPU::TilingBuilderOpInterface origOp, OutputTiling tiles,
                                          mlir::PatternRewriter& rewriter, Logger log) {
    // apply the generated tiling strategy and create tiled operations
    // insert the tiled pattern with a concat to the IR
    SmallVector<SmallVector<mlir::Value>> resultTileVals(origOp->getNumResults());
    SmallVector<SmallVector<ShapeRef>> resultTileOffsets(origOp->getNumResults());
    for (const auto& outputTile : tiles) {
        const auto tiledRes = reifyTileTopK(origOp, outputTile, rewriter, log);
        for (auto& p : origOp->getResults() | indexed) {
            auto idx = p.index();
            const auto tiledShape = getShape(tiledRes[idx]);
            VPUX_THROW_UNLESS(tiledShape == outputTile.shape,
                              "Inferred tiled output shape '{0}' doesn't match with generated '{1}'", tiledShape,
                              outputTile.shape);
            resultTileOffsets[idx].push_back(outputTile.offsets);
            resultTileVals[idx].push_back(tiledRes[idx]);
        }
    }

    SmallVector<mlir::Value> opsConcat;
    for (auto& p : origOp->getResults() | indexed) {
        auto idx = p.index();
        auto opConcat = rewriter.create<VPU::ConcatOp>(origOp->getLoc(), origOp->getResult(idx).getType(),
                                                       mlir::ValueRange(resultTileVals[idx]),
                                                       makeArrayRef(resultTileOffsets[idx]));
        opsConcat.push_back(opConcat.output());
    }
    rewriter.replaceOp(origOp, opsConcat);

    return mlir::success();
}

mlir::Value reifyTile(VPU::TilingBuilderOpInterface origOp, const TileInfo& outputTile, mlir::OpBuilder& builder,
                      Logger log) {
    log = log.nest(2);
    log.trace("{0}", outputTile);

    auto inputTiling = origOp.backInferTileInfo(outputTile, log);
    auto& inTiles = inputTiling.tiles;

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
    storeTilingStrategyForOp(origOp.getOperation(), tiles.front().axis);
    // TODO: delete when function will allow rewrite for multiple output ops (E#59988)
    if (mlir::isa<VPU::TopKOp>(origOp)) {
        return applyTileStrategyTopK(origOp, tiles, rewriter, log);
    }

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
    // only one user
    if (op->getResult(0).hasOneUse()) {
        return false;
    }
    // only one result but multiple users
    auto user1 = op->getResult(0).user_begin();
    for (auto remainUser : llvm::drop_begin(op->getResult(0).getUsers())) {
        if (remainUser != *user1) {
            return true;
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
    auto opTilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
    if (!opTilingBuilder) {
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
    log.nest(1).trace("Attempting to satisfy PREFETCHING tiling.");
    auto tiles = opTilingBuilder.getTilingStrategy(TilingMode::PREFETCHING, log.nest());
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
        auto clusterOp = mlir::cast<VPU::ClusteredOpInterface>(op);
        auto numClusters = VPU::getOptimalNumClusters(
                clusterOp, filterType.getShape()[Dims4D::Filter::OC],
                clusterOp->getAttr(VPU::multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue());
        auto filterDistributedType = VPU::getDistributedFilterTypeFromOp(nceOp, filterType, numClusters);
        for (auto filterType : filterDistributedType.getDistributedTypes()) {
            filterSize += filterType.cast<VPU::DistributedTensorType>().getTotalAllocSize();
        }
    } else {
        filterSize = filterType.getTotalAllocSize();
    }

    auto cmxThreshold = Byte(static_cast<int64_t>(
            std::ceil(static_cast<double>(VPU::getTotalCMXSize(op).count()) * LARGE_CONST_THRESHOLD_RATIO)));
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
    auto opTilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
    if (!opTilingBuilder) {
        return false;
    }

    // Find the available tiling size over C
    // The pipelining should be doable with this tiling size
    log.nest(1).trace("Checking large const pipeline tiling.");
    auto tiles = opTilingBuilder.getTilingStrategy(TilingMode::PIPELINING, log.nest());
    if (tiles.begin()->axis != Shape(getShape(op->getResult(0)).size(), 1)) {
        log.nest(1).trace("Found pipelining tiling strategy {0}", tiles.begin()->axis);
        return true;
    }

    return false;
}

bool archSupportsSwLayerTiling(VPU::ArchKind arch) {
    return arch != VPU::ArchKind::VPUX30XX && arch != VPU::ArchKind::VPUX311X;
}

}  // namespace VPU
}  // namespace vpux
