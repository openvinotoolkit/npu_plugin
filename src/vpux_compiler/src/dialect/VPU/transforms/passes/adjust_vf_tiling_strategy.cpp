//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion_utils.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"

using namespace vpux;
using namespace VPU;

namespace {

bool validateCMXSize(mlir::Operation* largest, mlir::ArrayRef<mlir::Operation*> inputOps,
                     mlir::ArrayRef<mlir::Operation*> outputOps, int64_t strategySize,
                     const TilingOperationStorage::UPtr& opStorage, bool isVerticalFusionPipeliningCandidate,
                     Logger log) {
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
        const auto thresholdCMXSize = isVerticalFusionPipeliningCandidate
                                              ? getTotalCMXVFPipelineFragmentationAwareSize(largest)
                                              : getTotalCMXFragmentationAwareSize(largest);
        if (inputSize + outputSize +
                    VPU::getRequiredCMX(largest, opTiling.value().second, log, opTiling.value().first) >=
            thresholdCMXSize) {
            return false;
        }
    }

    return true;
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

    return !validateCMXSize(largestOp, inputOps, outputOps, tilingLimit, opMaxStorage, false, log);
}

//
// AdjustVFTilingStrategyPass
//

class AdjustVFTilingStrategyPass final : public AdjustVFTilingStrategyBase<AdjustVFTilingStrategyPass> {
public:
    explicit AdjustVFTilingStrategyPass(bool enableVerticalFusionPipelining, Logger log)
            : _enableVerticalFusionPipelining(enableVerticalFusionPipelining) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
    bool _enableVerticalFusionPipelining = false;
};

mlir::LogicalResult AdjustVFTilingStrategyPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (enableVerticalFusionPipelining.hasValue()) {
        _log.trace("Overloading AdjustVFTilingStrategyPass argument by MLIR variable");
        _enableVerticalFusionPipelining = enableVerticalFusionPipelining;
    }
    return mlir::success();
}

//
// safeRunOnModule
//

void AdjustVFTilingStrategyPass::safeRunOnFunc() {
    auto func = getOperation();

    func.walk([&](VPU::VerticalFusionOp op) {
        auto vfOps = op.getOps().front().getOps<VPU::VerticalFusionOpInterface>();
        if (std::distance(vfOps.begin(), vfOps.end()) == 1) {
            return;
        }
        _log.trace("Recalculating tiling strategy for '{0}'", op.getLoc());

        auto tilingStrategy = parseIntArrayAttr<int64_t>(op.getTilingStrategy().cast<mlir::ArrayAttr>());
        const auto axis = getVFTilingDim(tilingStrategy, getVFOperations(op));

        if (mlir::failed(axis)) {
            return;
        }

        const auto tilingAxis = axis.value();

        // if it's not spatial dimension, do not increase
        if (tilingAxis.ind() < Dims4D::Act::getSpatialDim(0).ind()) {
            return;
        }

        // Candidates to enable VF pipelining
        // E.g.
        //      DPU_0_0 -> SW_0 -> DPU_0_1 ->
        //      DPU_1_0 -> SW_1 -> DPU_1_1 -> ...
        // SW_0 and DPU_1_0, SW_1 and DPU_0_1 will be executed in parallel separately
        auto isVerticalFusionPipeliningCandidate = _enableVerticalFusionPipelining && isVFPipelinePattern(op);

        auto opStorage = std::make_unique<TilingOperationStorage>();
        auto tilingRegions = calculateTilingRegions(op, tilingStrategy, _log, opStorage);
        if (mlir::failed(tilingRegions)) {
            VPUX_THROW("Incorrect tiling {0} for VF {1}", tilingStrategy, op);
        }
        auto getTilingLimit = VPU::getValidTilingLimit(op, tilingAxis, _log);
        if (mlir::failed(getTilingLimit)) {
            return;
        }
        const auto tilingLimit = getTilingLimit.value();

        auto largestOp = VPU::getLargestOp(op);

        if (largestOp == nullptr) {
            return;
        }

        // get all operations in VF block which inputs are arguments of the block only
        auto inputOps = getInputOps(op);

        // get all outputs of the block
        auto outputOps = getOutputOps(op);

        auto tilingMaxStrategy = tilingStrategy;
        tilingMaxStrategy[tilingAxis.ind()] = tilingLimit;
        if (!isVerticalFusionPipeliningCandidate) {
            // if there is bigger tile than expected for any operation,
            // there definitely will be spill
            // or there is spill even with maximum number of tiles
            if (hasSpillInSubgraph(op, inputOps, outputOps, tilingMaxStrategy, tilingLimit, largestOp, opStorage,
                                   _log)) {
                return;
            }
            // if the VF op is not a VF pipelining candidate and it has no tiling
            // which means that the operations in this VF region originally has no spilling
            // ignore this VF to avoid extra cost
            if (tilingStrategy == SmallVector<int64_t>(tilingStrategy.size(), 1)) {
                return;
            }
        }
        do {
            _log.trace("Analysis of VF block {0} for tiling strategy {1}", op->getLoc(), tilingStrategy);

            if (validateCMXSize(largestOp, inputOps, outputOps, tilingStrategy[tilingAxis.ind()], opStorage,
                                isVerticalFusionPipeliningCandidate, _log)) {
                break;
            }

            ++tilingStrategy[tilingAxis.ind()];
            opStorage = std::make_unique<TilingOperationStorage>();
            auto getValidTilingStrategy = VPU::getMinimalValidTilingStrategyFromRange(
                    op, tilingStrategy, tilingMaxStrategy, tilingAxis, opStorage, _log);
            if (mlir::failed(getValidTilingStrategy)) {
                _log.trace("Cannot find valid tiling strategy {0}", op->getLoc());
                return;
            }
            tilingStrategy = getValidTilingStrategy.value();

        } while (tilingStrategy[tilingAxis.ind()] < tilingLimit);

        _log.trace("VF block '{0}' tiling strategy is adjusted to {1}", op->getLoc(), tilingStrategy);
        op.setTilingStrategyAttr(getIntArrayAttr(op.getContext(), ArrayRef(tilingStrategy)));
    });
}

}  // namespace

//
// createAdjustVFTilingStrategyPass
//

std::unique_ptr<mlir::Pass> VPU::createAdjustVFTilingStrategyPass(bool enableVerticalFusionPipelining, Logger log) {
    return std::make_unique<AdjustVFTilingStrategyPass>(enableVerticalFusionPipelining, log);
}
