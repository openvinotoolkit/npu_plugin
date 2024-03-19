//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion_utils.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/SetOperations.h>
#include <llvm/ADT/SmallSet.h>

#include <mlir/IR/IRMapping.h>

using namespace vpux;
using namespace VPU;

namespace {

//
// MergeVFRegionRewriter
//

class MergeVFRegionRewriter final : public mlir::OpRewritePattern<VPU::VerticalFusionOp> {
public:
    MergeVFRegionRewriter(mlir::MLIRContext* ctx, bool enableVerticalFusionPipelining, Logger log)
            : mlir::OpRewritePattern<VPU::VerticalFusionOp>(ctx),
              _enableVerticalFusionPipelining(enableVerticalFusionPipelining),
              _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(VPU::VerticalFusionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool checkVFCostFunction(VPU::VerticalFusionOp newBlock, VPU::VerticalFusionOp parentVFOp) const;
    bool waitOtherUsers(VPU::VerticalFusionOp newBlock, VPU::VerticalFusionOp parentVFOp) const;
    mlir::ArrayAttr getVFTilingInfo(VPU::VerticalFusionOp newBlock, VPU::VerticalFusionOp parentVFOp) const;
    bool checkTiling(TilingStorage& tilingRegions, VPU::VerticalFusionOp currentOp, VPU::VerticalFusionOp prevOp) const;
    bool adjustTiling(SmallVector<int64_t>& tilingInfo, Dim dimTileAxis, VPU::VerticalFusionOp currentOp,
                      VPU::VerticalFusionOp prevOp) const;
    vpux::Byte getCMXUsedByTheLargestOperation(VPU::VerticalFusionOp vfOp, mlir::ArrayRef<int64_t> tilingStrategy,
                                               Logger log) const;
    bool checkMemUsedByTheLargestOperation(mlir::ArrayRef<int64_t> minTilingStrategy, const Dim dimTileAxis,
                                           VPU::VerticalFusionOp prevOp, VPU::VerticalFusionOp currentOp,
                                           Logger log) const;

    int64_t getTilingLimitForVF(Dim dimTileAxis, VPU::VerticalFusionOp currentOp, VPU::VerticalFusionOp prevOp) const;

    SmallVector<mlir::Operation*> getVFBlocksOperations(VPU::VerticalFusionOp currentOp,
                                                        VPU::VerticalFusionOp prevOp) const;

    bool _enableVerticalFusionPipelining = false;
    Logger _log;
};

inline bool hasTiling(const ArrayRef<int64_t> tilingInfo) {
    return llvm::any_of(tilingInfo, [](auto i) {
        return i != 1;
    });
}

int64_t MergeVFRegionRewriter::getTilingLimitForVF(Dim dimTileAxis, VPU::VerticalFusionOp currentOp,
                                                   VPU::VerticalFusionOp prevOp) const {
    return getTilingLimit(dimTileAxis, getVFBlocksOperations(currentOp, prevOp));
}

SmallVector<mlir::Operation*> MergeVFRegionRewriter::getVFBlocksOperations(VPU::VerticalFusionOp currentOp,
                                                                           VPU::VerticalFusionOp prevOp) const {
    auto operations = getVFOperations(currentOp);

    llvm::copy(getVFOperations(prevOp), std::back_inserter(operations));
    return operations;
}

bool MergeVFRegionRewriter::adjustTiling(SmallVector<int64_t>& tilingInfo, Dim axis, VPU::VerticalFusionOp currentOp,
                                         VPU::VerticalFusionOp prevOp) const {
    auto dimTileAxis = axis.ind();

    const auto tilingLimit = getTilingLimitForVF(axis, currentOp, prevOp);

    ++tilingInfo[dimTileAxis];

    while (tilingInfo[dimTileAxis] < tilingLimit) {
        auto calculatedRegions = calculateTilingRegions(currentOp, tilingInfo, _log);

        if (mlir::failed(calculatedRegions)) {
            ++tilingInfo[dimTileAxis];
            continue;
        }

        if (checkTiling(calculatedRegions.value(), currentOp, prevOp)) {
            return true;
        }

        ++tilingInfo[dimTileAxis];
    }

    return false;
}

bool MergeVFRegionRewriter::checkTiling(TilingStorage& tilingRegions, VPU::VerticalFusionOp currentOp,
                                        VPU::VerticalFusionOp prevOp) const {
    for (auto op : currentOp.getOperands() | indexed) {
        const auto operand = op.value();
        const auto index = op.index();

        auto defOp = operand.getDefiningOp();
        if (defOp == nullptr || defOp != prevOp) {
            continue;
        }

        // The VFValue could be empty when the index is not added to the region
        // e.g., an Eltwise op with two identical inputs, the argNumbers of each input are 1
        // so there won't be VFValue for index 0
        if (tilingRegions.gatherValue(index).empty()) {
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

vpux::Byte MergeVFRegionRewriter::getCMXUsedByTheLargestOperation(VPU::VerticalFusionOp vfOp,
                                                                  mlir::ArrayRef<int64_t> tilingStrategy,
                                                                  Logger log) const {
    const auto largestOp = VPU::getLargestOp(vfOp);

    auto opStorage = std::make_unique<TilingOperationStorage>();
    auto tilingRegions = calculateTilingRegions(vfOp, tilingStrategy, log, opStorage);
    if (mlir::failed(tilingRegions)) {
        VPUX_THROW("Failed to get memory used by the largest operation, incorrect tiling {0} for VF {1}",
                   tilingStrategy, vfOp);
    }
    auto opTiling = opStorage->get(largestOp, 0);
    VPUX_THROW_WHEN(!opTiling.has_value(), "There is no tile information of operation {0}", largestOp);

    return VPU::getRequiredCMX(largestOp, opTiling.value().second, log, opTiling.value().first);
}

/*
 Check CMX memory used percentage by the largest operation, should not exceed VF_LARGEST_OP_MEM_RATIO to prevent
 spilling

 There are several steps in order to calculate the memory used by the largest operation before 2 VF blocks
 merged:
 1. Minimal tiling is provided
 2. Get the valid maximal tiling strategies for prevOp and currentOp VF
    a. Get the valid maximal tiling of prevOp, assuming the value is M
    b. Get the valid maximal tiling of currentOp, assuming the value is N
    c. Set the smaller value in M and N - std::min(M, N) as the valid maximal tiling of merged VF temporarily, call it
       fusedMaxTiling.
       Note that std::min(M, N) can not be used for memory calculation directly.
       For example, in case M < N, M may not be valid for all the operations in currentOp.
       As well, in case M > N, N may not be valid for all the operations in prevOp.
       So it needs further adjustment in 2-d and 2-e.
    d. calculate the valid maximal tiling for prevOp by geting a valid tiling number close to fusedMaxTiling in range
 [minimalTiling, fusedMaxTiling], this valid maximal tiling can ensure validity for the operations in prev VF block
    e. calculate the valid maximal tiling for currentOp by geting a valid tiling number close to fusedMaxTiling in range
 [minimalTiling, fusedMaxTiling], this valid maximal tiling can ensure validity for the operations in current VF block
 3. Calculate the memory used by the largest operation with the maximal tiling calculated in step 2-d and 2-e
    a. Calculate the memory used by the largest operation in prevOp, assuming the value is X
    b. Calculate the memory used by the largest operation in currentOp, assuming the value is Y
    c. If prevOp and currentOp are merged, the memory used by the largest operation should be the larger one in X and Y
       and it should not exceed VF_LARGEST_OP_MEM_RATIO
*/
bool MergeVFRegionRewriter::checkMemUsedByTheLargestOperation(mlir::ArrayRef<int64_t> minTilingStrategy,
                                                              const Dim dimTileAxis, VPU::VerticalFusionOp prevOp,
                                                              VPU::VerticalFusionOp currentOp, Logger log) const {
    // check only for spatial dims
    if (dimTileAxis.ind() < Dims4D::Act::getSpatialDim(0).ind()) {
        return true;
    }

    auto prevTilingStrategy = parseIntArrayAttr<int64_t>(prevOp.getTilingStrategy().cast<mlir::ArrayAttr>());
    auto currentTilingStrategy = parseIntArrayAttr<int64_t>(currentOp.getTilingStrategy().cast<mlir::ArrayAttr>());

    // calculate valid maximal tiling strategy which can be supported by the operations in prevOp and currentOp blocks
    // and all operations can fit in CMX
    const auto getPrevMaxTiling = VPU::getValidTilingLimit(prevOp, dimTileAxis, log);
    const auto getCurrentMaxTiling = VPU::getValidTilingLimit(currentOp, dimTileAxis, log);
    if (mlir::failed(getPrevMaxTiling) || mlir::failed(getCurrentMaxTiling)) {
        log.trace("Failed to get valid tiling limit");
        return false;
    }

    auto tilingLimit = std::min(getPrevMaxTiling.value(), getCurrentMaxTiling.value());
    SmallVector<int64_t> fusedTilingMaxStrategy(prevTilingStrategy.size(), 1);
    fusedTilingMaxStrategy[dimTileAxis.ind()] = tilingLimit;

    // calculate valid maximal tiling strategies for prevOp and currentOp
    auto prevOpStorage = std::make_unique<TilingOperationStorage>();
    auto getPrevValidMaxStrategy = VPU::getMaximalValidTilingStrategyFromRange(
            prevOp, minTilingStrategy, fusedTilingMaxStrategy, dimTileAxis, prevOpStorage, log);
    auto currentOpStorage = std::make_unique<TilingOperationStorage>();
    auto getCurrentValidMaxStrategy = VPU::getMaximalValidTilingStrategyFromRange(
            currentOp, minTilingStrategy, fusedTilingMaxStrategy, dimTileAxis, currentOpStorage, log);
    if (mlir::failed(getPrevValidMaxStrategy) || mlir::failed(getCurrentValidMaxStrategy)) {
        log.trace("Failed to calculate valid maximal tiling strategy");
        return false;
    }

    // check the memory used by the largest operation in case valid maximal tiling strategy is applied
    const auto memUsedByPrevLargestOp = getCMXUsedByTheLargestOperation(prevOp, getPrevValidMaxStrategy.value(), log);
    const auto memUsedByCurrentLargestOp =
            getCMXUsedByTheLargestOperation(currentOp, getCurrentValidMaxStrategy.value(), log);
    const auto memUsedByLargestOp = std::max(memUsedByPrevLargestOp.count(), memUsedByCurrentLargestOp.count());

    const auto totalAvailableCMXSize = getTotalCMXFragmentationAwareSize(prevOp.getOperation()).count();

    if (memUsedByLargestOp > totalAvailableCMXSize * VF_LARGEST_OP_MEM_RATIO) {
        log.trace("Memory consumed by the largest operation {0} exceeds the total available memory size {1}",
                  memUsedByLargestOp, totalAvailableCMXSize);
        return false;
    }

    return true;
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

    // both VF regions should have same tiling axes
    const auto currentTiling = parseIntArrayAttr<int64_t>(currentOp.getTilingStrategy());
    const auto prevTiling = parseIntArrayAttr<int64_t>(prevOp.getTilingStrategy());

    if (currentTiling.size() != prevTiling.size()) {
        return false;
    }

    auto curAxis = getVFTilingDim(currentTiling, getVFOperations(currentOp));
    auto prevAxis = getVFTilingDim(prevTiling, getVFOperations(prevOp));

    if (mlir::failed(curAxis) || mlir::failed(prevAxis)) {
        return false;
    }

    bool curHasTiling = hasTiling(currentTiling);
    bool prevHasTiling = hasTiling(prevTiling);
    // in case both subgraphs have tiling, check if they match
    // if there is only one subgraph with tiling, check if it's allowed
    // to tile second one with such axis
    // if both doesn't have tiling, check if there is at least one
    // allowed axis for both of them
    if (curHasTiling && prevHasTiling) {
        if (curAxis.value() != prevAxis.value()) {
            return false;
        }
    } else {
        auto dimArrCurrent = getAllowedDims(getVFOperations(currentOp), _log);
        auto dimArrPrev = getAllowedDims(getVFOperations(prevOp), _log);

        if (!curHasTiling && !prevHasTiling) {
            DimArr intersect = getAllowedDims(getVFBlocksOperations(currentOp, prevOp), _log);
            if (intersect.empty()) {
                return false;
            }
        } else if ((!curHasTiling && llvm::find(dimArrCurrent, prevAxis.value()) == dimArrCurrent.end()) ||
                   (!prevHasTiling && llvm::find(dimArrPrev, curAxis.value()) == dimArrPrev.end())) {
            return false;
        }
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
 5. CMX memory used percentage by the largest operation shouldn't exceed VF_LARGEST_OP_MEM_RATIO to prevent spilling
*/
mlir::ArrayAttr MergeVFRegionRewriter::getVFTilingInfo(VPU::VerticalFusionOp prevOp,
                                                       VPU::VerticalFusionOp currentOp) const {
    const auto currentTiling = parseIntArrayAttr<int64_t>(currentOp.getTilingStrategy());
    const auto prevTiling = parseIntArrayAttr<int64_t>(prevOp.getTilingStrategy());

    VPUX_THROW_WHEN(currentTiling.size() != prevTiling.size(),
                    "Tiling info rank of current block {0} is not equal to tiling info rank of previous block {1}",
                    currentTiling.size(), prevTiling.size());

    SmallVector<int64_t> tilingArray;
    llvm::transform(llvm::seq<size_t>(0, currentTiling.size()), std::back_inserter(tilingArray), [&](size_t index) {
        return std::max(currentTiling[index], prevTiling[index]);
    });

    const auto axis = getVFTilingDim(tilingArray, getVFBlocksOperations(currentOp, prevOp));

    if (mlir::failed(axis)) {
        return nullptr;
    }

    for (auto operation : currentOp.getBody()->getOps<VPU::VerticalFusionOpInterface>()) {
        auto restrictedAxes = operation.restrictedFusionAxes();
        if (llvm::find(restrictedAxes, axis.value()) != restrictedAxes.end()) {
            return nullptr;
        }
    }

    auto tilingRegions = restoreTilingRegions(currentOp, _log);
    if (!llvm::equal(currentTiling, tilingArray)) {
        auto calculatedRegions = calculateTilingRegions(currentOp, tilingArray, _log);

        if (mlir::failed(calculatedRegions)) {
            return nullptr;
        }

        tilingRegions = calculatedRegions.value();
    }

    if (!checkTiling(tilingRegions, currentOp, prevOp) && !adjustTiling(tilingArray, axis.value(), currentOp, prevOp)) {
        return nullptr;
    }

    // the memory consumed by the largest operation should be less than the threshold
    // otherwise there might be spilling for the weights
    if (!checkMemUsedByTheLargestOperation(tilingArray, axis.value(), prevOp, currentOp, _log)) {
        _log.trace("Memory consumed by the largest operation exceeds the total available memory size");
        return nullptr;
    }

    return getIntArrayAttr(currentOp.getContext(), tilingArray);
}

mlir::LogicalResult MergeVFRegionRewriter::matchAndRewrite(VPU::VerticalFusionOp vfOp,
                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Vertical fusion region {0}", vfOp);
    if (!_enableVerticalFusionPipelining) {
        // When the vertical fusion pipelining is disabled
        // only merge operations with tiling strategy
        if (!hasTiling(parseIntArrayAttr<int64_t>(vfOp.getTilingStrategy()))) {
            return mlir::failure();
        }
    }

    VPU::VerticalFusionOp vfBlock = nullptr;
    VPU::VerticalFusionOp parentVFOp = nullptr;
    mlir::ArrayAttr tilingInfo = nullptr;
    for (auto operand : vfOp->getOperands()) {
        parentVFOp = operand.getDefiningOp<VPU::VerticalFusionOp>();

        if (parentVFOp == nullptr) {
            continue;
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
            break;
        }
    }

    if (vfBlock == nullptr) {
        return mlir::failure();
    }

    _log.trace("Merge regions {0} - {1}", vfOp, vfBlock);
    fuseOpsInBlock(rewriter, vfOp, vfBlock.getOperation(), tilingInfo);

    return mlir::success();
}

//
// MergeVfSubgraphsPass
//

class MergeVfSubgraphsPass final : public MergeVfSubgraphsBase<MergeVfSubgraphsPass> {
public:
    explicit MergeVfSubgraphsPass(bool enableVerticalFusionPipelining, Logger log)
            : _enableVerticalFusionPipelining(enableVerticalFusionPipelining) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
    bool _enableVerticalFusionPipelining = false;
};

mlir::LogicalResult MergeVfSubgraphsPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (enableVerticalFusionPipelining.hasValue()) {
        _log.trace("Overloading MergeVfSubgraphsPass argument by MLIR variable");
        _enableVerticalFusionPipelining = enableVerticalFusionPipelining;
    }
    return mlir::success();
}

//
// safeRunOnModule
//

void MergeVfSubgraphsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MergeVFRegionRewriter>(&ctx, _enableVerticalFusionPipelining, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMergeVfSubgraphsPass
//

std::unique_ptr<mlir::Pass> VPU::createMergeVfSubgraphsPass(bool enableVerticalFusionPipelining, Logger log) {
    return std::make_unique<MergeVfSubgraphsPass>(enableVerticalFusionPipelining, log);
}
