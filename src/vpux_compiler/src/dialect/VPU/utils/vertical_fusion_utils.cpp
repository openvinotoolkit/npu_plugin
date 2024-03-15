//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"

using namespace vpux;
using namespace VPU;

TilingStorage vpux::VPU::restoreTilingRegions(VPU::VerticalFusionOp vfOp, Logger log,
                                              const TilingOperationStorage::UPtr& opStorage) {
    auto storage = calculateTilingRegions(
            vfOp, ArrayRef(parseIntArrayAttr<int64_t>(vfOp.getTilingStrategy().cast<mlir::ArrayAttr>())), log,
            opStorage);

    VPUX_THROW_WHEN(mlir::failed(storage), "Restored tiling {0} of operation {1} is incorrect",
                    vfOp.getTilingStrategy(), vfOp);

    return storage.value();
}

mlir::FailureOr<TilingStorage> vpux::VPU::calculateTilingRegions(VPU::VerticalFusionOp vfOp, const OutputTiling& tiles,
                                                                 Logger log,
                                                                 const TilingOperationStorage::UPtr& opStorage) {
    auto termination = vfOp.getBody()->getTerminator();

    if (termination == nullptr) {
        return mlir::failure();
    }

    if (termination->getNumOperands() == 0) {
        return mlir::failure();
    }

    auto lastOp = termination->getOperands().back().getDefiningOp();

    if (lastOp == nullptr) {
        return mlir::failure();
    }

    return calculateTilingRegions(lastOp, tiles, log, opStorage);
}

mlir::FailureOr<TilingStorage> vpux::VPU::calculateTilingRegions(mlir::Operation* operation, const OutputTiling& tiles,
                                                                 Logger log,
                                                                 const TilingOperationStorage::UPtr& opStorage,
                                                                 std::optional<size_t> numTile) {
    TilingStorage storage;

    if (auto tilingInfoInterface = mlir::dyn_cast<VPU::TilingInfoOpInterface>(operation)) {
        if (!tilingInfoInterface.isSupportedTiling(tiles, TilingMode::ISOLATED, log)) {
            return mlir::failure();
        }
    }

    for (const auto& item : tiles | indexed) {
        auto tile = item.value();

        auto inputTiling = TilingInfo(ArrayRef({item.value()}));
        if (auto tilingBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(operation)) {
            inputTiling = tilingBuilderOp.backInferTileInfo(tile, log);
        } else if (auto tilingViewLikeOp = mlir::dyn_cast<VPU::TilingViewLikeOpInterface>(operation)) {
            inputTiling = tilingViewLikeOp.backInferTileInfo(tile, log);
        } else {
            VPUX_THROW("Unsupported operation type {0} for VF", operation->getName());
        }

        const auto tileNumber = numTile.value_or(item.index());

        if (opStorage != nullptr) {
            opStorage->insert(operation, tileNumber, std::make_pair(inputTiling, tile));
            log.trace("TileInfo inserted for operation {0} tile {1}, {2}", *operation, tileNumber, tile);
        }

        for (const auto& op : operation->getOperands() | indexed) {
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

    const auto minTilingLength = MINIMUM_LENGTH_TILING *
                                 IE::getTileExecutor(operations.front()->getParentOfType<mlir::ModuleOp>()).getCount();

    const auto minLength = *minAxisLength;

    return minLength <= minTilingLength ? 1 : minLength / minTilingLength;
}

// get a valid tiling strategy for VF block between the given range of tiling strategy
// it returns mlir::failure() if all tiling strategies in this range can't be supported by all operations or operations
// can't fit in CMX
// otherwise, return the valid strategy that is close to the lower or upper boundary according to closeToUpperLimit
// parameter
mlir::FailureOr<SmallVector<int64_t>> getValidTilingStrategyFromRange(
        VPU::VerticalFusionOp op, ArrayRef<int64_t> lowerTilingStrategy, ArrayRef<int64_t> upperTilingStrategy,
        bool closeToUpperLimit, Dim tilingAxis, TilingOperationStorage::UPtr& opStorage, Logger log) {
    SmallVector<int64_t> validTilingStrategy =
            closeToUpperLimit ? to_small_vector(upperTilingStrategy) : to_small_vector(lowerTilingStrategy);

    auto notBeyondBoundary = [](int64_t value, int64_t lowerLimit, int64_t upperLimit, bool closeToUpperLimit) {
        return closeToUpperLimit ? value >= lowerLimit : value <= upperLimit;
    };

    while (notBeyondBoundary(validTilingStrategy[tilingAxis.ind()], lowerTilingStrategy[tilingAxis.ind()],
                             upperTilingStrategy[tilingAxis.ind()], closeToUpperLimit)) {
        auto curOpStorage = std::make_unique<TilingOperationStorage>();
        auto tilingRegions = calculateTilingRegions(op, validTilingStrategy, log, curOpStorage);
        if (!mlir::failed(tilingRegions)) {
            // a valid strategy is found
            opStorage.reset(curOpStorage.release());
            return validTilingStrategy;
        }

        if (closeToUpperLimit) {
            --validTilingStrategy[tilingAxis.ind()];
        } else {
            ++validTilingStrategy[tilingAxis.ind()];
        }
    }

    // no valid strategy can be found
    return mlir::failure();
}

// get a maximal valid tiling strategy for VF block between the given range of tiling strategy
// it returns mlir::failure() if all tiling strategies in this range can't be supported by all operations or operations
// can't fit in CMX
mlir::FailureOr<SmallVector<int64_t>> vpux::VPU::getMaximalValidTilingStrategyFromRange(
        VPU::VerticalFusionOp op, ArrayRef<int64_t> lowerTilingStrategy, ArrayRef<int64_t> upperTilingStrategy,
        Dim tilingAxis, TilingOperationStorage::UPtr& opStorage, Logger log) {
    return getValidTilingStrategyFromRange(op, lowerTilingStrategy, upperTilingStrategy, true, tilingAxis, opStorage,
                                           log);
}

// get a minimal valid tiling strategy for VF block between the given range of tiling strategy
// it returns mlir::failure() if all tiling strategies in this range can't be supported by all operations or operations
// can't fit in CMX
mlir::FailureOr<SmallVector<int64_t>> vpux::VPU::getMinimalValidTilingStrategyFromRange(
        VPU::VerticalFusionOp op, ArrayRef<int64_t> lowerTilingStrategy, ArrayRef<int64_t> upperTilingStrategy,
        Dim tilingAxis, TilingOperationStorage::UPtr& opStorage, Logger log) {
    return getValidTilingStrategyFromRange(op, lowerTilingStrategy, upperTilingStrategy, false, tilingAxis, opStorage,
                                           log);
}

std::optional<Dim> vpux::VPU::getVFTilingDim(ArrayRef<int64_t> tilingStrategy) {
    auto maxTiledLen = std::max_element(tilingStrategy.begin(), tilingStrategy.end());
    if (maxTiledLen != tilingStrategy.end() && *maxTiledLen != 1) {
        return Dim(std::distance(tilingStrategy.begin(), maxTiledLen));
    }
    return std::nullopt;
}

mlir::FailureOr<Dim> vpux::VPU::getVFTilingDim(ArrayRef<int64_t> tilingStrategy,
                                               ArrayRef<mlir::Operation*> operations) {
    auto dim = getVFTilingDim(tilingStrategy);
    if (dim.has_value()) {
        return dim.value();
    }

    auto allowedDims = getAllowedDims(operations, Logger::global());
    if (allowedDims.empty()) {
        return mlir::failure();
    }

    return allowedDims.front();
}

SmallVector<mlir::Operation*> vpux::VPU::getVFOperations(VPU::VerticalFusionOp op) {
    SmallVector<mlir::Operation*> operations;
    const auto getOpPointer = [](auto& op) -> mlir::Operation* {
        return &op;
    };
    llvm::copy(op.getBody()->without_terminator() | transformed(getOpPointer), std::back_inserter(operations));

    return operations;
}

DimArr vpux::VPU::getAllowedDims(ArrayRef<mlir::Operation*> operations, Logger log) {
    const auto dimComparator = [](Dim lhs, Dim rhs) -> bool {
        auto order = DimsOrder::NHWC;
        return order.hasDim(lhs) && order.hasDim(rhs) && order.dimPos(lhs) < order.dimPos(rhs);
    };
    DimArr allowedDims = DimsOrder::NHWC.toPermutation();
    for (auto tiledOperation : operations) {
        auto currentTiling = getTileDimOrder(tiledOperation, TilingMode::ISOLATED, log);
        DimArr intersect;
        std::set_intersection(allowedDims.begin(), allowedDims.end(), currentTiling.begin(), currentTiling.end(),
                              std::back_inserter(intersect), dimComparator);
        allowedDims = std::move(intersect);
    }

    return allowedDims;
}

// calculate limit for number of tiles that can be supported by all operations in the VF block and all operations can
// fit into CMX with it
mlir::FailureOr<int64_t> vpux::VPU::getValidTilingLimit(VPU::VerticalFusionOp op, const Dim tilingAxis, Logger log) {
    auto tilingStrategy = parseIntArrayAttr<int64_t>(op.getTilingStrategy().cast<mlir::ArrayAttr>());
    const auto getOpPointer = [](auto& op) -> mlir::Operation* {
        return &op;
    };
    auto tilingLimit =
            getTilingLimit(tilingAxis, to_small_vector(op.getBody()->without_terminator() | transformed(getOpPointer)));

    SmallVector<int64_t> tilingMaxStrategy(tilingStrategy.size(), 1);
    tilingMaxStrategy[tilingAxis.ind()] = tilingLimit;

    // tilingMaxStrategy may be not valid for all operations in VF block here, needs to be legalized
    auto opStorage = std::make_unique<TilingOperationStorage>();
    auto validTilingMaxStrategy =
            getMaximalValidTilingStrategyFromRange(op, tilingStrategy, tilingMaxStrategy, tilingAxis, opStorage, log);
    if (mlir::failed(validTilingMaxStrategy)) {
        return mlir::failure();
    }

    return validTilingMaxStrategy.value()[tilingAxis.ind()];
}

mlir::Operation* vpux::VPU::getLargestOp(VPU::VerticalFusionOp op) {
    auto operations = op.getBody()->without_terminator();

    const auto sumTypes = [&](const Byte& sum, mlir::Value value) {
        return sum + value.getType().cast<vpux::NDTypeInterface>().getTotalAllocSize();
    };

    const auto getAllocationSize = [&](auto valueList) -> Byte {
        return std::accumulate(valueList.begin(), valueList.end(), Byte(0), sumTypes);
    };

    auto largestOperation = std::max_element(operations.begin(), operations.end(), [&](auto& op1, auto& op2) {
        return getAllocationSize(op1.getOperands()) + getAllocationSize(op1.getResults()) <
               getAllocationSize(op2.getOperands()) + getAllocationSize(op2.getResults());
    });

    if (largestOperation == operations.end()) {
        return nullptr;
    }

    return &(*largestOperation);
}

bool vpux::VPU::isVFPipelinePattern(VPU::VerticalFusionOp op) {
    // Only support VF Pipeline when the VF subgraph contains DPU->SW->DPU tasks
    // More generic cases will be supported in the future
    // Track [E#95184]
    const auto getPoint = [](auto& op) {
        return &op;
    };
    auto operations = to_small_vector(op.getBody()->without_terminator() | transformed(getPoint));
    if (operations.size() != VF_PIPELINE_LENGTH) {
        return false;
    }
    return mlir::isa<VPU::NCEOpInterface>(operations[0]) && mlir::isa<VPU::SWOpInterface>(operations[1]) &&
           mlir::isa<VPU::NCEOpInterface>(operations[2]);
}

void vpux::VPU::fuseOpsInBlock(mlir::PatternRewriter& rewriter, VPU::VerticalFusionOp vfOp, mlir::Operation* prevOp,
                               mlir::ArrayAttr tilingInfo /*nullptr*/) {
    SmallVector<mlir::Operation*> prevOperations;
    auto prevOperands = prevOp->getOperands();
    SmallVector<mlir::Value> prevBlockArgs = prevOp->getOperands();
    mlir::Operation* lastOp = prevOp;
    const auto getOpPointer = [](auto& op) -> mlir::Operation* {
        return &op;
    };
    if (auto prevBlock = mlir::dyn_cast<VPU::VerticalFusionOp>(prevOp)) {
        prevBlockArgs.clear();
        llvm::copy(prevBlock.getBody()->getOperations() | transformed(getOpPointer),
                   std::back_inserter(prevOperations));
        llvm::copy(prevBlock.getBody()->getArguments(), std::back_inserter(prevBlockArgs));
        lastOp = prevBlock.getBody()->getTerminator()->getOperands().back().getDefiningOp();
    } else {
        prevOperations.push_back(prevOp);
    }

    SmallVector<size_t> argNumLastOp;
    SmallVector<size_t> argNumCurrentOp;
    mlir::DenseMap<size_t, size_t> opArgMapper;
    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange blockArgs) {
        mlir::IRMapping mapper;

        const auto curBlockArgs = vfOp.getBody()->getArguments();

        // map new operands with previous ones for both blocks
        for (size_t i = 0; i < blockArgs.size(); ++i) {
            if (i < prevBlockArgs.size()) {
                // map operands of first block with current ones
                mapper.map(prevBlockArgs[i], blockArgs[i]);

                // in case there is operand in second block which also
                // can be mapped with this operands - map them too
                if (opArgMapper.count(i) != 0) {
                    mapper.map(curBlockArgs[opArgMapper[i]], blockArgs[i]);
                }
            } else {
                // map other operands
                if (argNumCurrentOp.size() > i - prevBlockArgs.size() &&
                    curBlockArgs.size() > argNumCurrentOp[i - prevBlockArgs.size()]) {
                    mapper.map(curBlockArgs[argNumCurrentOp[i - prevBlockArgs.size()]], blockArgs[i]);
                }
                if (opArgMapper.count(i) != 0) {
                    mapper.map(curBlockArgs[opArgMapper[i]], blockArgs[i]);
                }
            }
        }

        SmallVector<mlir::Value> newResults;

        const auto copyOps = [&](auto operations) {
            for (auto* op : operations) {
                if (!mlir::isa<VPU::YieldOp>(op)) {
                    auto* clonedOp = builder.clone(*op, mapper);
                    if (op == lastOp && !argNumLastOp.empty()) {
                        for (auto index : argNumLastOp) {
                            mapper.map(curBlockArgs[index], clonedOp->getResult(0));
                        }
                    }
                } else {
                    for (auto operand : op->getOperands()) {
                        if (operand.getDefiningOp() != lastOp) {
                            newResults.push_back(mapper.lookupOrDefault(operand));
                        }
                    }
                }
            }
        };

        copyOps(prevOperations);
        copyOps(vfOp.getBody()->getOperations() | transformed(getOpPointer));

        builder.create<VPU::YieldOp>(loc, newResults.back());
    };

    SmallVector<mlir::Value> newOperands(prevOperands.begin(), prevOperands.end());

    VPUX_THROW_WHEN(lastOp == nullptr, "Couldn't find last operation in region {0}", prevOp);

    // for all operands in current region
    // sort them in following baskets
    // argNumLastOp - if operand is previous region
    // argNumCurrentOp - arguments of current region
    // opArgMapper - in case operand is already in the list,
    // map this operand and argument of current block in order to
    // create right correlation
    for (auto arg : vfOp.getBody()->getArguments()) {
        auto operand = vfOp.getOperand(arg.getArgNumber());
        if (operand.getDefiningOp() == prevOp) {
            argNumLastOp.push_back(arg.getArgNumber());
        } else {
            const auto value = llvm::find(newOperands, operand);
            if (value == newOperands.end()) {
                newOperands.push_back(operand);
                argNumCurrentOp.push_back(arg.getArgNumber());
            } else {
                opArgMapper[std::distance(newOperands.begin(), value)] = arg.getArgNumber();
            }
        }
    }

    if (tilingInfo == nullptr) {
        tilingInfo = vfOp.getTilingStrategy();
    }

    auto newVFOp = rewriter.create<VPU::VerticalFusionOp>(vfOp.getLoc(), vfOp->getResultTypes(), newOperands,
                                                          bodyBuilder, tilingInfo);

    rewriter.replaceOp(vfOp, newVFOp.getResult(0));
}
