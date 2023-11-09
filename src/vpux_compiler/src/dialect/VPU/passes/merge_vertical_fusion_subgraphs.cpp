//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux/compiler/dialect/VPU/utils/tile_utils.hpp>
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>

#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Pass/PassManager.h>

using namespace vpux;
using namespace VPU;

namespace {

//
// MergeVFRegionRewriter
//

class MergeVFRegionRewriter final : public mlir::OpRewritePattern<VPU::VerticalFusionOp> {
public:
    MergeVFRegionRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::VerticalFusionOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(VPU::VerticalFusionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool checkVFCostFunction(VPU::VerticalFusionOp newBlock, VPU::VerticalFusionOp parentVFOp) const;
    bool waitOtherUsers(VPU::VerticalFusionOp newBlock, VPU::VerticalFusionOp parentVFOp) const;
    mlir::ArrayAttr getVFTilingInfo(VPU::VerticalFusionOp newBlock, VPU::VerticalFusionOp parentVFOp) const;
    void fuseBlocks(mlir::PatternRewriter& rewriter, VPU::VerticalFusionOp vfOp, VPU::VerticalFusionOp prevOp,
                    mlir::ArrayAttr tilingInfo) const;
    bool checkTiling(TilingStorage& tilingRegions, VPU::VerticalFusionOp currentOp, VPU::VerticalFusionOp prevOp) const;
    bool adjustTiling(SmallVector<int64_t>& tilingInfo, VPU::VerticalFusionOp currentOp,
                      VPU::VerticalFusionOp prevOp) const;
    bool isValidVFInput(mlir::Value operand) const;
    Logger _log;
};

bool MergeVFRegionRewriter::adjustTiling(SmallVector<int64_t>& tilingInfo, VPU::VerticalFusionOp currentOp,
                                         VPU::VerticalFusionOp prevOp) const {
    auto maxTiledAxis = std::max_element(tilingInfo.begin(), tilingInfo.end());

    if (maxTiledAxis == tilingInfo.end()) {
        return false;
    }

    auto dimTileAxis = Dim(std::distance(tilingInfo.begin(), maxTiledAxis));

    SmallVector<mlir::Operation*> operations;
    const auto getPoint = [](auto& op) {
        return &op;
    };
    llvm::copy(prevOp.getBody()->without_terminator() | transformed(getPoint), std::back_inserter(operations));
    llvm::copy(currentOp.getBody()->without_terminator() | transformed(getPoint), std::back_inserter(operations));
    const auto tilingLimit = getTilingLimit(dimTileAxis, operations);

    ++tilingInfo[dimTileAxis.ind()];

    while (tilingInfo[dimTileAxis.ind()] < tilingLimit) {
        auto calculatedRegions = calculateTilingRegions(currentOp, tilingInfo, _log);

        if (mlir::failed(calculatedRegions)) {
            continue;
        }

        if (checkTiling(calculatedRegions.value(), currentOp, prevOp)) {
            return true;
        }

        ++tilingInfo[dimTileAxis.ind()];
    }

    return false;
}

bool MergeVFRegionRewriter::checkTiling(TilingStorage& tilingRegions, VPU::VerticalFusionOp currentOp,
                                        VPU::VerticalFusionOp prevOp) const {
    for (auto& op : currentOp.getOperands() | indexed) {
        const auto operand = op.value();
        const auto index = op.index();

        auto defOp = operand.getDefiningOp();
        if (defOp == nullptr || defOp != prevOp) {
            continue;
        }

        auto outTiles = to_small_vector(tilingRegions.gatherValue(index));

        if (mlir::failed(calculateTilingRegions(prevOp, outTiles, _log))) {
            return false;
        }
    }

    return true;
}

Byte getRequiredWeightsMemory(ArrayRef<VPU::VerticalFusionOpInterface> ops) {
    auto weightsMem = Byte(0);
    for (auto& op : ops) {
        if (mlir::isa<VPU::NCEOpInterface>(*op)) {
            auto outputShape = op->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape();
            weightsMem += getRequiredCMXForWeight(op, TileInfo(outputShape));
        }
    }

    return weightsMem;
}

/*
 Function checks if two blocks suit to be merged in one on following criterias:
 1. Number of operations doesn't exceed the limit
 2. In case there is only one operation in the block, it might be merged as first op in the block
 3. All multicluster strategies are same for both blocks if there are any
 4. Required CMX memory by constant weights shouldn't exceed the size of the whole memory
*/
bool MergeVFRegionRewriter::checkVFCostFunction(VPU::VerticalFusionOp prevOp, VPU::VerticalFusionOp currentOp) const {
    const auto prevBlock = prevOp.getBody();
    const auto parentVFOp = currentOp.getBody();

    auto newOps = prevBlock->getOps<VPU::VerticalFusionOpInterface>();
    auto oldOps = parentVFOp->getOps<VPU::VerticalFusionOpInterface>();

    if (newOps.empty() || oldOps.empty()) {
        return false;
    }

    const auto oldBlockSize = std::distance(oldOps.begin(), oldOps.end());
    const auto newBlockSize = std::distance(newOps.begin(), newOps.end());
    if (oldBlockSize + newBlockSize > MAXIMUM_VF_LENGTH) {
        auto isOne = [](auto val) {
            return val == 1;
        };

        // count here operations which don't increase computational cost
        // with kernel 1x1 and strides 1
        const auto oneKernelNCE = llvm::count_if(newOps, [&](auto op) {
            auto nceOp = llvm::dyn_cast<VPU::NCEOpInterface>(op.getOperation());
            return nceOp != nullptr && llvm::all_of(nceOp.getKernelSizeVal(), isOne) &&
                   llvm::all_of(nceOp.getStridesVal(), isOne);
        });

        // recheck condition excluding operation without additional computations needed
        if (oldBlockSize + newBlockSize - oneKernelNCE > MAXIMUM_VF_LENGTH) {
            return false;
        }
    }

    // for now we don't have logic for excluding from subgraph layers later on
    // so, we need to check right now if we could add new layer in advance
    // even if region looks ok
    if (newBlockSize == 1) {
        auto vfIface = *newOps.begin();
        VPUX_THROW_WHEN(vfIface == nullptr, "There is operation in VF region {0} which doesn't implement VF interface",
                        *newOps.begin());

        if (!vfIface.availableSingleMerge()) {
            return false;
        }
    }

    // all ops have same multicluster strategies or don't have them at all
    // so, compare only first operations in each block
    const auto isClusteredOp = [](auto op) {
        return llvm::dyn_cast<VPU::ClusteredOpInterface>(op.getOperation()) != nullptr;
    };
    const auto firstOldClusterOp = llvm::find_if(oldOps, isClusteredOp);
    const auto firstNewClusterOp = llvm::find_if(newOps, isClusteredOp);

    if (firstOldClusterOp != oldOps.end() && firstNewClusterOp != newOps.end()) {
        const auto oldBlockStrategy =
                llvm::dyn_cast<VPU::ClusteredOpInterface>(**firstOldClusterOp).getMultiClusterStrategy();
        const auto newBlockStrategy =
                llvm::dyn_cast<VPU::ClusteredOpInterface>(**firstNewClusterOp).getMultiClusterStrategy();

        // if only one strategy is defined - blocks don't match
        // in case both strategies are defined, they must be same
        if (oldBlockStrategy.has_value() ^ newBlockStrategy.has_value()) {
            return false;
        }

        if (oldBlockStrategy.has_value() && newBlockStrategy.has_value() &&
            oldBlockStrategy.value() != newBlockStrategy.value()) {
            return false;
        }
    }

    // the memory required by constant weights should be less than the threshold
    // otherwise there might be spilling for the weights
    auto weightsMem = getRequiredWeightsMemory(to_small_vector(oldOps));
    weightsMem += getRequiredWeightsMemory(to_small_vector(newOps));
    const auto totalCMXSize = VPU::getTotalCMXSize(prevOp.getOperation()).count() * VF_WEIGHTS_RATIO;
    if (totalCMXSize <= weightsMem.count()) {
        _log.trace("Required weights memory exceeds the total memory size");
        return false;
    }

    return true;
}

/*
 As soon as we don't have logic right now for excluding operations or break subgraph
 check in advance that all users or previous block will be merged to current one
*/
bool MergeVFRegionRewriter::waitOtherUsers(VPU::VerticalFusionOp prevOp, VPU::VerticalFusionOp currentOp) const {
    if (prevOp->hasOneUse()) {
        return true;
    }

    for (auto user : prevOp->getUsers()) {
        if (!mlir::isa<VPU::VerticalFusionOp>(user)) {
            return false;
        }
        if (user == currentOp) {
            continue;
        }

        const auto userGoToRegion = llvm::any_of(user->getUsers(), [&](auto current) {
            return current != currentOp;
        });

        if (userGoToRegion) {
            return false;
        }
    }

    return true;
}

/*
 There are several steps in order to adjust tiling in order to get it fit in CMX
 1. Restore tiles for operation from current block
 2. Match block arguments and tiles
 3. Restore tiles for previous block starting from operations which are operands of current block
 4. In case some operations doesn't fit in CMX, try to increase number of tiles by the limit
*/
mlir::ArrayAttr MergeVFRegionRewriter::getVFTilingInfo(VPU::VerticalFusionOp prevOp,
                                                       VPU::VerticalFusionOp currentOp) const {
    const auto currentTiling = parseIntArrayAttr<int64_t>(currentOp.tilingStrategy());
    const auto prevTiling = parseIntArrayAttr<int64_t>(prevOp.tilingStrategy());

    VPUX_THROW_WHEN(currentTiling.size() != prevTiling.size(),
                    "Tiling info rank of current block {0} is not equal to tiling info rank of previous block {1}",
                    currentTiling.size(), prevTiling.size());

    SmallVector<int64_t> tilingArray;
    llvm::transform(llvm::seq<size_t>(0, currentTiling.size()), std::back_inserter(tilingArray), [&](size_t index) {
        return std::max(currentTiling[index], prevTiling[index]);
    });

    auto tilingRegions = restoreTilingRegions(currentOp, _log);
    if (!llvm::equal(currentTiling, tilingArray)) {
        auto calculatedRegions = calculateTilingRegions(currentOp, tilingArray, _log);

        if (mlir::failed(calculatedRegions)) {
            return nullptr;
        }

        tilingRegions = calculatedRegions.value();
    }

    if (!checkTiling(tilingRegions, currentOp, prevOp) && !adjustTiling(tilingArray, currentOp, prevOp)) {
        return nullptr;
    }

    return getIntArrayAttr(currentOp.getContext(), tilingArray);
}

void MergeVFRegionRewriter::fuseBlocks(mlir::PatternRewriter& rewriter, VPU::VerticalFusionOp vfOp,
                                       VPU::VerticalFusionOp prevOp, mlir::ArrayAttr tilingInfo) const {
    SmallVector<size_t> argNumLastOp;
    SmallVector<size_t> argNumCurrentOp;
    mlir::DenseMap<size_t, size_t> opArgMapper;
    mlir::Operation* lastOp = nullptr;
    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange blockArgs) {
        mlir::BlockAndValueMapping mapper;

        const auto prevBlockArgs = prevOp.getBody()->getArguments();
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
            }
        }

        SmallVector<mlir::Value> newResults;

        const auto copyOps = [&](mlir::Block* bodyBlock) {
            for (auto& op : bodyBlock->getOperations()) {
                if (!mlir::isa<VPU::YieldOp>(op)) {
                    auto* clonedOp = builder.clone(op, mapper);
                    if (&op == lastOp && !argNumLastOp.empty()) {
                        for (auto index : argNumLastOp) {
                            mapper.map(curBlockArgs[index], clonedOp->getResult(0));
                        }
                    }
                } else {
                    for (auto operand : op.getOperands()) {
                        if (operand.getDefiningOp() != lastOp) {
                            newResults.push_back(mapper.lookupOrDefault(operand));
                        }
                    }
                }
            }
        };

        copyOps(prevOp.getBody());
        copyOps(vfOp.getBody());

        builder.create<VPU::YieldOp>(loc, newResults.back());
    };

    SmallVector<mlir::Value> newOperands(prevOp->getOperands().begin(), prevOp->getOperands().end());
    lastOp = prevOp.getBody()->getTerminator()->getOperands().back().getDefiningOp();

    VPUX_THROW_WHEN(lastOp == nullptr, "Couldn't find last operation in VF region {0}", prevOp);

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

    auto newVFOp = rewriter.create<VPU::VerticalFusionOp>(vfOp.getLoc(), vfOp->getResultTypes(), newOperands,
                                                          bodyBuilder, tilingInfo);

    rewriter.replaceOp(vfOp, newVFOp.getResult(0));
}

bool MergeVFRegionRewriter::isValidVFInput(mlir::Value operand) const {
    if (operand.isa<mlir::BlockArgument>()) {
        return true;
    }

    auto operation = operand.getDefiningOp();
    while (operation != nullptr) {
        const auto allBlockArguments = llvm::all_of(operation->getOperands(), [&](mlir::Value operand) {
            return operand.isa<mlir::BlockArgument>();
        });

        if (allBlockArguments || mlir::isa<Const::DeclareOp>(operation) ||
            mlir::isa<VPU::VerticalFusionOp>(operation)) {
            return true;
        }

        if (VPU::isPureViewOp(operation) || mlir::isa<VPU::MemPermuteOp>(operation)) {
            if (operation->getNumOperands() > 1) {
                return false;
            }
            operation = operation->getOperand(0).getDefiningOp();
            continue;
        }

        return false;
    }

    return false;
}

mlir::LogicalResult MergeVFRegionRewriter::matchAndRewrite(VPU::VerticalFusionOp vfOp,
                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Vertical fusion region {0}", vfOp);
    const auto hasTiling = llvm::any_of(parseIntArrayAttr<int64_t>(vfOp.tilingStrategy()), [](auto i) {
        return i != 1;
    });

    if (!hasTiling) {
        return mlir::failure();
    }

    VPU::VerticalFusionOp vfBlock = nullptr;
    VPU::VerticalFusionOp parentVFOp = nullptr;
    mlir::ArrayAttr tilingInfo = nullptr;
    for (auto operand : vfOp->getOperands()) {
        parentVFOp = operand.getDefiningOp<VPU::VerticalFusionOp>();

        if (parentVFOp == nullptr) {
            if (isValidVFInput(operand)) {
                continue;
            }

            return mlir::failure();
        }

        _log.trace("Analize vf region {0}", parentVFOp);
        if (!checkVFCostFunction(parentVFOp, vfOp)) {
            return mlir::failure();
        }
        const bool allInOldBlock = llvm::all_of(parentVFOp->getUsers(), [&](auto user) {
            return user == vfOp;
        });
        if (!allInOldBlock) {
            if (waitOtherUsers(parentVFOp, vfOp)) {
                continue;
            }
            return mlir::failure();
        }

        tilingInfo = getVFTilingInfo(parentVFOp, vfOp);
        if (tilingInfo == nullptr) {
            return mlir::failure();
        }

        if (vfBlock == nullptr) {
            vfBlock = parentVFOp;
        }
    }

    if (vfBlock == nullptr) {
        return mlir::failure();
    }

    _log.trace("Merge regions {0} - {1}", vfOp, vfBlock);
    fuseBlocks(rewriter, vfOp, vfBlock, tilingInfo);

    return mlir::success();
}

//
// MergeVfSubgraphsPass
//

class MergeVfSubgraphsPass final : public MergeVfSubgraphsBase<MergeVfSubgraphsPass> {
public:
    explicit MergeVfSubgraphsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnModule
//

void MergeVfSubgraphsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MergeVFRegionRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMergeVfSubgraphsPass
//

std::unique_ptr<mlir::Pass> VPU::createMergeVfSubgraphsPass(Logger log) {
    return std::make_unique<MergeVfSubgraphsPass>(log);
}
