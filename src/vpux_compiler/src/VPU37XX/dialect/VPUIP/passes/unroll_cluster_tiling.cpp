//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/passes/unroll_cluster_tiling.hpp"
#include "vpux/compiler/VPU30XX/dialect/VPUIP/passes/unroll_cluster_tiling.hpp"
#include "vpux/compiler/VPU37XX/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/VPU37XX/dialect/VPUIP/passes/unroll_cluster_tiling.hpp"

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"

#include "vpux/compiler/core/cost_model_utils.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

//
// ClusterSWRewriter
//

mlir::LogicalResult VPUIP::arch37xx::ClusterSWRewriter::matchAndRewrite(VPUIP::SwKernelOp swTask,
                                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("Process SW op: '{0}'", swTask);
    auto clusterOp = swTask->getParentOfType<VPUIP::NCEClusterTilingOp>();
    if (clusterOp == nullptr) {
        return mlir::failure();
    }

    auto vpurtTask = clusterOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_UNLESS(vpurtTask != nullptr, "Can't get VPURT task operation");

    auto cycleBeginAttr = vpurtTask->getAttr(cycleBegin);
    auto cycleEndAttr = vpurtTask->getAttr(cycleEnd);

    rewriter.setInsertionPointAfter(vpurtTask);

    VPUX_THROW_UNLESS(!clusterOp.getInputs().empty(), "Wrong inputs size: {0}", clusterOp.getInputs().size());
    VPUX_THROW_UNLESS(!clusterOp.getOutputs().empty(), "Wrong outputs size: {0}", clusterOp.getOutputs().size());

    auto parentInput = *clusterOp.getInputs().begin();
    auto parentOutput = *clusterOp.getOutputs().begin();

    auto parentInputType = parentInput.getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto parentOutputType = parentOutput.getType().dyn_cast<VPUIP::DistributedBufferType>();

    VPUX_THROW_UNLESS(parentInputType != nullptr && parentOutputType != nullptr,
                      "Input and output types must have distributed type. Got: inT={0}, outT={1}", parentInputType,
                      parentOutputType);

    auto inDistribution = parentInputType.getDistribution();
    auto outDistribution = parentOutputType.getDistribution();

    VPUX_THROW_UNLESS(inDistribution.getNumClusters() == outDistribution.getNumClusters(),
                      "Input '{0}' and output '{1}' number of clusters are not equal", inDistribution.getNumClusters(),
                      outDistribution.getNumClusters());

    auto inDistributionMode = inDistribution.getMode().getValue();
    auto outDistributionMode = outDistribution.getMode().getValue();
    VPUX_THROW_WHEN(outDistributionMode == VPU::DistributionMode::OVERLAPPED,
                    "No support for SW op {0}; output in OVERLAPPED mode.", swTask->getLoc());
    VPUX_THROW_WHEN(inDistributionMode == VPU::DistributionMode::OVERLAPPED &&
                            outDistributionMode != VPU::DistributionMode::SEGMENTED,
                    "When SW op has input in OVERLAPPED mode then output must be segmented. op = {0}, out mode = '{1}'",
                    swTask->getLoc(), VPU::stringifyDistributionMode(outDistributionMode));

    auto numClusters = inDistribution.getNumClusters().getInt();
    auto loc = swTask->getLoc();

    auto parentInputBuffs = swTask.inputs();
    auto parentOutputBuffs = swTask.output_buffs();

    // store inputs/outputs per cluster
    _log.trace("Cluster inputs");
    mlir::DenseMap<int64_t, SmallVector<mlir::Value>> inputBuffs;
    mlir::DenseMap<int64_t, SmallVector<mlir::Value>> outputBuffs;
    SmallVector<TileInfo> outputTiles;
    SmallVector<TilingInfo> inputTiles;

    auto allowDiscontinuousBuffers = VPUIP::isStridedDataAccessSupported(swTask);
    for (auto input : parentInputBuffs) {
        auto currBuffs = VPUIP::getPerClusterSWMemoryBuffers(_ctx, loc, "input", clusterOp, input, numClusters,
                                                             rewriter, _log, allowDiscontinuousBuffers);
        for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
            inputBuffs[clusterId].push_back(currBuffs[clusterId]);
        }
    }

    _log.trace("Cluster outputs");
    for (auto output : parentOutputBuffs) {
        auto currBuffs = VPUIP::getPerClusterSWComputeBuffers(_ctx, loc, "outputBuff", clusterOp, output, numClusters,
                                                              rewriter, _log, true);
        for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
            outputBuffs[clusterId].push_back(currBuffs[clusterId]);
        }
    }

    auto getPerClusterTileInfo = [&numClusters](ShapeRef shape, ShapeRef offset, int64_t tileDim) {
        Shape axis(shape.size(), 1);
        axis[Dim(tileDim)] = numClusters;
        return TileInfo(shape, offset, axis);
    };

    // For overlapped input, the Swkernel's attr need to be updated according to its input/output tiles
    auto needUpdateAttrs = inDistributionMode == VPU::DistributionMode::OVERLAPPED;
    if (needUpdateAttrs) {
        auto outTileIndex = VPUIP::getTilingDimIndex(parentOutputType);
        VPUX_THROW_UNLESS(outTileIndex.has_value(), "Can not get tiling dim for {0}", parentOutputType);
        for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
            SmallVector<TileInfo> tiles;
            for (auto operand : parentInputBuffs) {
                auto clusterOperand = VPU::getDistributedOperandFromNCEClusterTiling(clusterOp, operand);
                auto distributedType = clusterOperand.getType().dyn_cast<VPUIP::DistributedBufferType>();
                auto tileIndex = VPUIP::getTilingDimIndex(distributedType);
                VPUX_THROW_UNLESS(tileIndex.has_value(), "Can not get tiling dim for {0}", distributedType);
                auto tileInfo = getPerClusterTileInfo(distributedType.getPerClusterMemoryShapes()[clusterId],
                                                      distributedType.getPerClusterMemoryShapeOffsets()[clusterId],
                                                      tileIndex.value());
                tiles.push_back(tileInfo);
            }
            auto inTiles = TilingInfo(tiles);
            auto outTile = getPerClusterTileInfo(parentOutputType.getPerClusterComputeShapes()[clusterId],
                                                 parentOutputType.getPerClusterComputeShapeOffsets()[clusterId],
                                                 outTileIndex.value());
            inputTiles.push_back(inTiles);
            outputTiles.push_back(outTile);
        }
    }

    auto profilingBuffs = VPUIP::getPerClusterSWMemoryBuffers(_ctx, loc, "profilingBuff", clusterOp,
                                                              swTask.profiling_data(), numClusters, rewriter, _log);

    mlir::OperationName kernelName = swTask->getName();
    auto kernelArgsRange = [&kernelName](VPUIP::SwKernelOp swKernelOp) {
        SmallVector<mlir::Attribute> attrStorage;

        for (auto&& kernelRun : swKernelOp.body().getOps<VPUIP::SwKernelRun>()) {
            kernelName = kernelRun->getName();
            if (kernelRun.attrs().has_value()) {
                const mlir::ArrayAttr arrayAttrs = kernelRun.attrs().value();
                const auto& attrs = arrayAttrs.getValue();
                for (const auto& attr : attrs) {
                    attrStorage.push_back(attr);
                }
            }
        }
        return attrStorage;
    };

    auto taskArgs = kernelArgsRange(swTask);

    _log.trace("Create new ops");
    for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        const auto newLoc = appendLoc(loc, "cluster_{0}", clusterId);
        mlir::Value profilingData = nullptr;
        mlir::Type profilingOutputType = nullptr;

        if (swTask.profiling_data()) {
            profilingOutputType = profilingBuffs[clusterId].getType();
            profilingData = profilingBuffs[clusterId];
        }

        _log.trace("Create new task");

        SmallVector<mlir::Type> inputTypes;
        for (auto temp : inputBuffs[clusterId]) {
            inputTypes.push_back(temp.getType());
        }
        for (auto temp : outputBuffs[clusterId]) {
            inputTypes.push_back(temp.getType());
        }

        auto newArgs = needUpdateAttrs ? VPUIP::getSwkernelNewAttrsAfterTiling(swTask, taskArgs, inputTiles[clusterId],
                                                                               outputTiles[clusterId], _log.nest())
                                       : taskArgs;
        for (auto arg : newArgs) {
            inputTypes.push_back(arg.getType());
        }

        VPUIP::createRuntimeKernelDefinition(_module, _log.nest(), VPU::getArch(swTask.getOperation()));

        auto module = swTask->getParentOfType<mlir::ModuleOp>();
        auto kernelFunc = module.lookupSymbol<mlir::func::FuncOp>(swTask.kernelFunctionAttr());
        VPUX_THROW_UNLESS(kernelFunc, "Invalid function call : '{0}', undefined kernel name",
                          swTask.kernelFunctionAttr());

        const auto kernelCode = kernelFunc->getAttrOfType<mlir::StringAttr>("VPU.kernel_code");
        const auto kernelEntryPoint = kernelFunc->getAttrOfType<mlir::StringAttr>("VPU.kernel_entry");
        auto newOperands = kernelFunc.getName();

        auto builtInFunction =
                VPUIP::createBuiltInFunction(_module, newOperands, inputTypes, kernelEntryPoint, kernelCode, _log);

        auto newTask = VPURT::wrapIntoTaskOp<VPUIP::SwKernelOp>(
                rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), newLoc, inputBuffs[clusterId],
                outputBuffs[clusterId], profilingData, builtInFunction, getIntAttr(rewriter, clusterId));

        initSwKernel(newTask, inputBuffs[clusterId], outputBuffs[clusterId], newArgs, _log.nest());

        _log.trace("Task created: {0}", newTask);

        auto newVpurtTask = newTask->getParentOfType<VPURT::TaskOp>();

        if (cycleBeginAttr) {
            newVpurtTask->setAttr(cycleBegin, cycleBeginAttr);
        }
        if (cycleEndAttr) {
            newVpurtTask->setAttr(cycleEnd, cycleEndAttr);
        }
    }

    _log.trace("Remove task");

    rewriter.eraseOp(vpurtTask);
    return mlir::success();
}

namespace {

//
// UnrollClusterTilingPass
//

class UnrollClusterTilingPass final : public VPUIP::arch37xx::UnrollClusterTilingBase<UnrollClusterTilingPass> {
public:
    explicit UnrollClusterTilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollClusterTilingPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    auto dmaOp = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    auto dmaPortCount = dmaOp.count();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<VPUIP::ClusterDMARewriter>(&ctx, dmaPortCount, _log);
    patterns.add<VPUIP::arch37xx::ClusterSWRewriter>(&ctx, module, _log);
    patterns.add<VPUIP::arch30xx::ClusterNCERewriter>(&ctx, _log);

    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUnrollClusterTilingPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::arch37xx::createUnrollClusterTilingPass(Logger log) {
    return std::make_unique<UnrollClusterTilingPass>(log);
}
