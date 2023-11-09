//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"

using namespace vpux;
using namespace VPU;

TilingStorage vpux::VPU::restoreTilingRegions(VPU::VerticalFusionOp vfOp, Logger log,
                                              const TilingOperationStorage::UPtr& opStorage) {
    auto storage = calculateTilingRegions(
            vfOp, makeArrayRef(parseIntArrayAttr<int64_t>(vfOp.tilingStrategy().cast<mlir::ArrayAttr>())), log,
            opStorage);

    VPUX_THROW_WHEN(mlir::failed(storage), "Restored tiling {0} of operation {1} is incorrect", vfOp.tilingStrategy(),
                    vfOp);

    return storage.value();
}

mlir::FailureOr<TilingStorage> vpux::VPU::calculateTilingRegions(VPU::VerticalFusionOp vfOp, const OutputTiling& tiles,
                                                                 Logger log,
                                                                 const TilingOperationStorage::UPtr& opStorage) {
    auto termination = vfOp.getBody()->getTerminator();

    if (termination == nullptr) {
        return mlir::failure();
    }

    auto lastOp = llvm::dyn_cast<VPU::TilingBuilderOpInterface>(termination->getOperands().back().getDefiningOp());

    if (lastOp == nullptr) {
        return mlir::failure();
    }

    return calculateTilingRegions(lastOp, tiles, log, opStorage);
}

mlir::FailureOr<TilingStorage> vpux::VPU::calculateTilingRegions(VPU::TilingBuilderOpInterface tilingBuilderOp,
                                                                 const OutputTiling& tiles, Logger log,
                                                                 const TilingOperationStorage::UPtr& opStorage,
                                                                 mlir::Optional<size_t> numTile) {
    TilingStorage storage;

    auto tilingInfoInterface = mlir::dyn_cast<VPU::TilingInfoOpInterface>(tilingBuilderOp.getOperation());

    if (tilingInfoInterface == nullptr) {
        return mlir::failure();
    }

    if (!tilingInfoInterface.isSupportedTiling(tiles, TilingMode::ISOLATED, log)) {
        return mlir::failure();
    }

    for (const auto& item : tiles | indexed) {
        auto tile = item.value();
        const auto inputTiling = tilingBuilderOp.backInferTileInfo(tile, log);
        const auto tileNumber = numTile.value_or(item.index());

        if (opStorage != nullptr) {
            opStorage->insert(tilingBuilderOp.getOperation(), tileNumber, std::make_pair(inputTiling, tile));
            log.trace("TileInfo inserted for operation {0} tile {1}, {2}", *tilingBuilderOp.getOperation(), tileNumber,
                      tile);
        }

        for (const auto& op : tilingBuilderOp.getOperation()->getOperands() | indexed) {
            const auto operand = op.value();
            const auto indexOp = op.index();

            if (auto arg = operand.dyn_cast<mlir::BlockArgument>()) {
                storage.insert(arg.getArgNumber(), tileNumber, inputTiling.tiles[indexOp]);
                log.trace("TileInfo inserted for argument {0} tile {1}, {2}", arg.getArgNumber(), tileNumber,
                          inputTiling.tiles[indexOp]);
                continue;
            }
            const auto oneTile = {inputTiling.tiles[indexOp]};
            auto innerStorage = calculateTilingRegions(operand.getDefiningOp(), oneTile, log, opStorage,
                                                       numTile.value_or(item.index()));

            if (mlir::failed(innerStorage)) {
                return mlir::failure();
            }

            storage.merge(innerStorage.value());
        }
    }

    return storage;
}

mlir::FailureOr<TilingStorage> vpux::VPU::calculateTilingRegions(VPU::VerticalFusionOp vfOp,
                                                                 ArrayRef<int64_t> tilingStrategy, Logger log,
                                                                 const TilingOperationStorage::UPtr& opStorage) {
    const auto outputShape = getShape(vfOp->getResult(0));
    const auto strategy = Shape(tilingStrategy);

    const auto tiles = fillDividedTiles(vfOp, strategy, outputShape);
    if (mlir::failed(tiles)) {
        return mlir::failure();
    }

    return calculateTilingRegions(vfOp, tiles.value(), log, opStorage);
}

int64_t vpux::VPU::getTilingLimit(Dim axis, ArrayRef<mlir::Operation*> operations) {
    const auto axisLengths = to_small_vector(operations | transformed([&](auto* op) {
                                                 return getShape(op->getResult(0))[axis];
                                             }));
    auto minAxisLength = std::min_element(axisLengths.begin(), axisLengths.end());

    VPUX_THROW_WHEN(minAxisLength == axisLengths.end(), "Unable to get minimum of axis length");

    const auto minTilingLength =
            MINIMUM_LENGTH_TILING *
            IE::getAvailableExecutor(operations.front()->getParentOfType<mlir::ModuleOp>(), ExecutorKind::NCE).count();

    const auto minLength = *minAxisLength;
    VPUX_THROW_WHEN(minLength <= minTilingLength || minTilingLength == 0,
                    "Minimum size by axis {0} and tiling length {1} are incorrect", axis, minTilingLength);

    return minLength / minTilingLength;
}
