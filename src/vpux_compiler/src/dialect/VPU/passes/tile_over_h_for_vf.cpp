//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;
using namespace VPU;

namespace {

//
// TileOverHForVFPass
//

class TileOverHForVFPass final : public TileOverHForVFBase<TileOverHForVFPass> {
public:
    explicit TileOverHForVFPass(bool enablePrefetchTiling, Logger log): _enablePrefetchTiling(enablePrefetchTiling) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
    bool _enablePrefetchTiling = true;
};

mlir::LogicalResult TileOverHForVFPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (tilingMode.hasValue()) {
        _log.trace("Overloading C++ createTilingStrategyAssignmentPass argument by MLIR variable");
        _enablePrefetchTiling = tilingMode.getValue() == "PREFETCH";
    }
    return mlir::success();
}

DimArr increaseDimPrioritization(DimArrRef tileDimOrder, Dim dim) {
    DimArr resultTileDimOrder(to_small_vector(tileDimOrder));
    auto dimIt = std::find(resultTileDimOrder.begin(), resultTileDimOrder.end(), dim);
    if (dimIt == resultTileDimOrder.begin() || dimIt == resultTileDimOrder.end()) {
        return resultTileDimOrder;
    }
    resultTileDimOrder.erase(dimIt);
    resultTileDimOrder.insert(resultTileDimOrder.begin(), dim);
    return resultTileDimOrder;
}

//
// safeRunOnFunc
//

void TileOverHForVFPass::safeRunOnFunc() {
    auto func = getOperation();

    func->walk([&](VPU::VerticalFusionOpInterface op) {
        if (!mlir::isa<VPU::TilingBuilderOpInterface>(op.getOperation())) {
            return;
        }

        if (!op->hasAttr(tilingStrategy)) {
            return;
        }

        auto isHOnlyTilingStrategy = [&](Shape tiling) {
            auto tileDimNum = llvm::count_if(tiling, [](auto i) {
                return i != 1;
            });
            return tileDimNum == 1 && tiling[Dims4D::Act::H] != 1;
        };
        auto origTilingStrategy =
                Shape(parseIntArrayAttr<int64_t>(op->getAttr(tilingStrategy).cast<mlir::ArrayAttr>()));
        if (isHOnlyTilingStrategy(origTilingStrategy)) {
            return;
        }

        _log.trace("Got op {0} at {1}, original tiling {2}", op->getName(), op->getLoc(), origTilingStrategy);
        auto tilingMode = TilingMode::ISOLATED;
        auto tileDimOrder = getTileDimOrder(op, tilingMode, _log);
        auto newTileDimOrder = increaseDimPrioritization(tileDimOrder, Dims4D::Act::H);
        if (newTileDimOrder.empty() || newTileDimOrder[0] != Dims4D::Act::H) {
            _log.trace("Tiling on H is not supported");
            return;
        }
        mlir::FailureOr<OutputTiling> tileHStrategy;
        // To keep aligned with prefetch-tiling pass
        // By default, SW uses isolated tiling mode
        // HW uses Pipelining mode to prioritize the activation tiling
        if (mlir::isa<VPU::SWOpInterface>(op.getOperation())) {
            tileHStrategy = getSWLayerTilingStrategyWithTileDimOrder(op, tilingMode, newTileDimOrder, _log);
        } else {
            if (_enablePrefetchTiling) {
                tilingMode = TilingMode::PIPELINING;
            }
            tileHStrategy = getHWLayerTilingStrategyWithTileDimOrder(op, tilingMode, newTileDimOrder, _log);
        }
        if (mlir::failed(tileHStrategy)) {
            _log.trace("H-prioritized tiling strategy is not found");
            return;
        }
        auto tileHStrategyArray = tileHStrategy.getValue()[0].axis;
        if (!isHOnlyTilingStrategy(tileHStrategyArray)) {
            _log.trace("Tiling strategy {0} is not only tiled by H", tileHStrategyArray);
            return;
        }
        _log.trace("Tile over H for op `{0}` at `{1}`, {2}", op->getName(), op->getLoc(), tileHStrategyArray);
        op->setAttr(tilingStrategy, getIntArrayAttr(op->getContext(), tileHStrategyArray));
    });
}

}  // namespace

//
// createTileOverHForVFPass
//

std::unique_ptr<mlir::Pass> VPU::createTileOverHForVFPass(bool enablePrefetchTiling, Logger log) {
    return std::make_unique<TileOverHForVFPass>(enablePrefetchTiling, log);
}
