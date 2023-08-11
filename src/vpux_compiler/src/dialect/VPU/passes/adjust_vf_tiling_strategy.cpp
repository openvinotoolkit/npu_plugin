//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include <mlir/Pass/PassManager.h>

using namespace vpux;
using namespace VPU;

namespace {

bool validateCMXSize(mlir::Operation* largest, mlir::ArrayRef<mlir::Operation*> inputOps,
                     mlir::ArrayRef<mlir::Operation*> outputOps, int64_t strategySize,
                     const TilingOperationStorage::UPtr& opStorage, Logger log) {
    for (auto index : irange(strategySize)) {
        auto inputSize = Byte(0);

        for (auto op : inputOps) {
            auto tileInfo = opStorage->get(op, index);
            VPUX_THROW_WHEN(!tileInfo.has_value(), "There is no information about tile {0} of operation {1}", index,
                            *op);

            auto tileTypes = getTileTypes(op, tileInfo.value().second, tileInfo.value().first);
            VPUX_THROW_WHEN(tileTypes.empty(), "There are not enough types for tile of operation {0}", *op);
            // exclude output type information
            tileTypes.pop_back();
            for (auto type : tileTypes) {
                inputSize += type.getTotalAllocSize();
            }
        }

        auto outputSize = Byte(0);

        for (auto op : outputOps) {
            auto tileInfo = opStorage->get(op, index);
            VPUX_THROW_WHEN(!tileInfo.has_value(), "There is no information about tile {0} of operation {1}", index,
                            *op);

            auto tileTypes = getTileTypes(op, tileInfo.value().second, tileInfo.value().first);
            VPUX_THROW_WHEN(tileTypes.empty(), "There is no output type for tile of operation {0}", *op);

            auto type = tileTypes.back();
            outputSize += type.getTotalAllocSize();
        }

        auto opTiling = opStorage->get(largest, index);
        VPUX_THROW_WHEN(!opTiling.has_value(), "There is no information about tile {0} of operation {1}", index,
                        *largest);
        log.trace("Check for tile number {0}: inputs' size {1} outputs's size {2}", index, inputSize, outputSize);
        if (inputSize + outputSize +
                    VPU::getRequiredCMX(largest, opTiling.value().second, log, opTiling.value().first) >=
            getTotalCMXFragmentationAwareSize(largest)) {
            return false;
        }
    }

    return true;
}

mlir::Operation* getLargestOp(VPU::VerticalFusionOp op) {
    auto operations = op.getBody()->without_terminator();

    const auto sumTypes = [&](const Byte& sum, mlir::Value value) {
        return sum + value.getType().cast<vpux::NDTypeInterface>().getTotalAllocSize();
    };

    const auto getValuesSize = [&](auto valueList) -> Byte {
        return std::accumulate(valueList.begin(), valueList.end(), Byte(0), sumTypes);
    };

    auto largestOperation = std::max_element(operations.begin(), operations.end(), [&](auto& op1, auto& op2) {
        return getValuesSize(op1.getOperands()) + getValuesSize(op1.getResults()) <
               getValuesSize(op2.getOperands()) + getValuesSize(op2.getResults());
    });

    if (largestOperation == operations.end()) {
        return nullptr;
    }

    return &(*largestOperation);
}

SmallVector<mlir::Operation*> getOutputOps(VPU::VerticalFusionOp op) {
    return to_small_vector(op.getBody()->getTerminator()->getOperands() |
                           transformed([](auto operand) -> mlir::Operation* {
                               return operand.getDefiningOp();
                           }));
}

SmallVector<mlir::Operation*> getInputOps(VPU::VerticalFusionOp op) {
    return to_small_vector(op.getBody()->without_terminator() | filtered([](auto& current) -> bool {
                               return llvm::all_of(current.getOperands(), [](mlir::Value operand) {
                                   return operand.dyn_cast<mlir::BlockArgument>() != nullptr;
                               });
                           }) |
                           transformed([](auto& current) -> mlir::Operation* {
                               return &current;
                           }));
}

bool hasSpillInSubgraph(VPU::VerticalFusionOp op, mlir::ArrayRef<mlir::Operation*> inputOps,
                        mlir::ArrayRef<mlir::Operation*> outputOps, mlir::ArrayRef<int64_t> tilingStrategy,
                        int64_t tilingLimit, mlir::Operation* largestOp,
                        const std::unique_ptr<TilingOperationStorage>& opStorage, Logger log) {
    // same for all tiles
    size_t numTile = 0;

    const auto hasDifferentTileOp = llvm::any_of(inputOps, [&](auto* operation) {
        auto operTilingPair = opStorage->get(operation, numTile);

        if (!operTilingPair.has_value()) {
            return true;
        }

        auto outputOriginTiling = operTilingPair.value().second;

        const auto usersHasDiffTiling = llvm::any_of(operation->getUsers(), [&](auto* user) {
            auto vfTilingInfo = opStorage->get(user, numTile);

            VPUX_THROW_WHEN(!vfTilingInfo.has_value(),
                            "There is no information about VF tiling for operation {0} and tile {1}", user, numTile);

            const auto vfTiles = vfTilingInfo.value().first.tiles;

            return llvm::find(vfTiles, outputOriginTiling) == vfTiles.end();
        });
        return usersHasDiffTiling;
    });

    if (hasDifferentTileOp) {
        return true;
    }

    auto opMaxStorage = std::make_unique<TilingOperationStorage>();
    auto tilingRegions = calculateTilingRegions(op, tilingStrategy, log, opMaxStorage);
    if (mlir::failed(tilingRegions)) {
        return true;
    }

    return !validateCMXSize(largestOp, inputOps, outputOps, tilingLimit, opMaxStorage, log);
}

//
// AdjustVFTilingStrategyPass
//

class AdjustVFTilingStrategyPass final : public AdjustVFTilingStrategyBase<AdjustVFTilingStrategyPass> {
public:
    explicit AdjustVFTilingStrategyPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnModule
//

void AdjustVFTilingStrategyPass::safeRunOnFunc() {
    auto func = getOperation();

    func.walk([&](VPU::VerticalFusionOp op) {
        _log.trace("Recalculating tiling strategy for '{0}'", op.getLoc());

        auto tilingStrategy = parseIntArrayAttr<int64_t>(op.tilingStrategy().cast<mlir::ArrayAttr>());

        auto maxTiledLen = std::max_element(tilingStrategy.begin(), tilingStrategy.end());

        if (maxTiledLen == tilingStrategy.end()) {
            return;
        }

        auto opStorage = std::make_unique<TilingOperationStorage>();
        auto tilingRegions = calculateTilingRegions(op, tilingStrategy, _log, opStorage);
        if (mlir::failed(tilingRegions)) {
            VPUX_THROW("Incorrect tiling {0} for VF {1}", tilingStrategy, op);
        }

        const auto tilingAxis = Dim(std::distance(tilingStrategy.begin(), maxTiledLen));
        const auto getPoint = [](auto& op) {
            return &op;
        };
        const auto tilingLimit =
                getTilingLimit(tilingAxis, to_small_vector(op.getBody()->without_terminator() | transformed(getPoint)));

        auto largestOp = getLargestOp(op);

        if (largestOp == nullptr) {
            return;
        }

        // get all operations in VF block which inputs are arguments of the block only
        auto inputOps = getInputOps(op);

        // get all outputs of the block
        auto outputOps = getOutputOps(op);

        // if there is bigger tile than expected for any operation,
        // there definitely will be spill
        // or there is spill even with maximum number of tiles
        auto tilingMaxStrategy = tilingStrategy;
        tilingMaxStrategy[tilingAxis.ind()] = tilingLimit;
        if (hasSpillInSubgraph(op, inputOps, outputOps, tilingMaxStrategy, tilingLimit, largestOp, opStorage, _log)) {
            return;
        }

        do {
            _log.trace("Analysis of VF block {0} for tiling strategy {1}", op, tilingStrategy);

            if (validateCMXSize(largestOp, inputOps, outputOps, tilingStrategy[tilingAxis.ind()], opStorage, _log)) {
                break;
            }

            ++tilingStrategy[tilingAxis.ind()];

            opStorage = std::make_unique<TilingOperationStorage>();
            tilingRegions = calculateTilingRegions(op, tilingStrategy, _log, opStorage);

            if (mlir::failed(tilingRegions)) {
                VPUX_THROW("Incorrect tiling {0} for VF {1}", tilingStrategy, op);
            }

        } while (tilingStrategy[tilingAxis.ind()] < tilingLimit);

        op.tilingStrategyAttr(getIntArrayAttr(op.getContext(), makeArrayRef(tilingStrategy)));
    });
}

}  // namespace

//
// createAdjustVFTilingStrategyPass
//

std::unique_ptr<mlir::Pass> VPU::createAdjustVFTilingStrategyPass(Logger log) {
    return std::make_unique<AdjustVFTilingStrategyPass>(log);
}
