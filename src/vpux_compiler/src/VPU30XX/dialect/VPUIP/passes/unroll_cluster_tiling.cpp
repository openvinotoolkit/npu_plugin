//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/passes/unroll_cluster_tiling.hpp"
#include "vpux/compiler/VPU30XX/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/VPU30XX/dialect/VPUIP/passes/unroll_cluster_tiling.hpp"

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

//
// ClusterNCERewriter
//

void VPUIP::arch30xx::ClusterNCERewriter::getInputBuffers(
        SmallVector<mlir::Value>& parentInputBuffs, SmallVector<mlir::Value>& inputBuffs,
        SmallVector<mlir::Value>& parentInputSparsityMap, SmallVector<mlir::Value>& inputSparsityMapBuffs,
        SmallVector<mlir::Value>& parentInputSETable, SmallVector<mlir::Value>& inputSETableBuffs, mlir::Location loc,
        VPUIP::NCEClusterTilingOp clusterOp, VPUIP::NCEClusterTaskOp nceTask, const int64_t numClusters,
        mlir::PatternRewriter& rewriter) const {
    inputBuffs = VPUIP::getPerClusterMemoryBuffers(
            _ctx, loc, "input", VPU::getDistributedOperandFromNCEClusterTiling(clusterOp, nceTask.input()),
            nceTask.input(), numClusters, rewriter);

    auto parentInput = *clusterOp.getInputs().begin();
    auto parentInputType = parentInput.getType().cast<VPUIP::DistributedBufferType>();
    mlir::UnitAttr isSegmented = isSegmentedNCETask(parentInputType);

    parentInputBuffs = VPU::isSegmentedOverC(parentInputType.getDistribution())
                               ? inputBuffs
                               : SmallVector<mlir::Value>(numClusters, parentInput);

    inputSparsityMapBuffs = VPUIP::getPerClusterMemoryBuffers(
            _ctx, loc, "inputSparsityMap",
            VPU::getDistributedOperandFromNCEClusterTiling(clusterOp, nceTask.input_sparsity_map()),
            nceTask.input_sparsity_map(), numClusters, rewriter);
    inputSETableBuffs = VPUIP::getPerClusterMemoryBuffers(
            _ctx, loc, "inputSETable",
            VPU::getDistributedOperandFromNCEClusterTiling(clusterOp, nceTask.input_storage_element_table()),
            nceTask.input_storage_element_table(), numClusters, rewriter);

    auto arch = VPU::getArch(nceTask);
    bool isDWOpAndNeedsAlign = VPU::isDWOpAndNeedsAlign(arch, nceTask.task_type());
    for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        // For 37XX and 30XX arch, ensure we have H_per_cluster x W as a multiple of 4 (or 8 for sparse inputs).
        // If the storage element table is present, its segment size has to fit this restriction
        if (isSegmented && clusterId != (numClusters - 1) &&
            (nceTask.task_type() == VPUIP::NCETaskType::CONV || isDWOpAndNeedsAlign)) {
            auto inShape = inputBuffs[clusterId].getType().cast<NDTypeInterface>().getShape();
            if (nceTask.input_storage_element_table() != nullptr) {
                inShape = inputSETableBuffs[clusterId].getType().cast<NDTypeInterface>().getShape();
            }
            const auto isInputSparse =
                    nceTask.input_sparsity_map() != nullptr || nceTask.input_storage_element_table() != nullptr;
            const auto hAlignment = VPU::getSOHPerClusterHeightAlignment(inShape[Dims4D::Act::W], isInputSparse);
            VPUX_THROW_UNLESS((inShape[Dims4D::Act::H] % hAlignment) == 0,
                              "For segmented cluster we must have alignment to {0}, type: {1}", hAlignment,
                              inputBuffs[clusterId].getType());
        }
    }

    parentInputSparsityMap = SmallVector<mlir::Value>(
            numClusters, VPU::getDistributedOperandFromNCEClusterTiling(clusterOp, nceTask.input_sparsity_map()));
    parentInputSETable = SmallVector<mlir::Value>(
            numClusters,
            VPU::getDistributedOperandFromNCEClusterTiling(clusterOp, nceTask.input_storage_element_table()));
}

void VPUIP::arch30xx::ClusterNCERewriter::getOutputBuffers(SmallVector<mlir::Value>& parentOutputBuffs,
                                                           SmallVector<mlir::Value>& outputBuffs,
                                                           SmallVector<mlir::Value>& parentOutputSparsityMap,
                                                           SmallVector<mlir::Value>& outputSparsityMapBuffs,
                                                           mlir::Location loc, VPUIP::NCEClusterTilingOp clusterOp,
                                                           VPUIP::NCEClusterTaskOp nceTask, const int64_t numClusters,
                                                           mlir::PatternRewriter& rewriter) const {
    auto parentInputType = (*clusterOp.getInputs().begin()).getType().cast<VPUIP::DistributedBufferType>();
    auto parentOutputType = (*clusterOp.getOutputs().begin()).getType().cast<VPUIP::DistributedBufferType>();

    auto inDistribution = parentInputType.getDistribution();
    auto outDistribution = parentOutputType.getDistribution();

    auto inDistributionMode = inDistribution.getMode().getValue();
    auto outDistributionMode = outDistribution.getMode().getValue();
    // Elementwise operations may support overlapping for trailing convolution.
    // In that case both input and output modes are OVERLAPPED.
    const auto isEltwise = (nceTask.task_type() == VPUIP::NCETaskType::ELTWISE);
    VPUX_THROW_WHEN(!isEltwise && outDistributionMode == VPU::DistributionMode::OVERLAPPED,
                    "No support for NCE output in OVERLAPPED mode.");
    VPUX_THROW_WHEN(!isEltwise && inDistributionMode == VPU::DistributionMode::OVERLAPPED &&
                            outDistributionMode != VPU::DistributionMode::SEGMENTED,
                    "When NCE has input in OVERLAPPED mode then output must be segmented. out mode = '{0}'",
                    VPU::stringifyDistributionMode(outDistributionMode));

    parentOutputSparsityMap = SmallVector<mlir::Value>(
            numClusters, VPU::getDistributedOperandFromNCEClusterTiling(clusterOp, nceTask.output_sparsity_map_buff()));

    outputBuffs = VPUIP::getPerClusterComputeBuffers(
            _ctx, loc, "outputBuff", VPU::getDistributedOperandFromNCEClusterTiling(clusterOp, nceTask.output_buff()),
            nceTask.output_buff(), numClusters, rewriter, true);
    outputSparsityMapBuffs = VPUIP::getPerClusterComputeBuffers(
            _ctx, loc, "outputSparsityMapBuff",
            VPU::getDistributedOperandFromNCEClusterTiling(clusterOp, nceTask.output_sparsity_map_buff()),
            nceTask.output_sparsity_map_buff(), numClusters, rewriter, true);

    parentOutputBuffs = SmallVector<mlir::Value>(numClusters, *clusterOp.getOutputs().begin());
    if (VPU::isSegmentedOverC(outDistribution)) {
        // for SEG SOK parent output buffers = output buffers
        parentOutputBuffs = outputBuffs;
    }
}

mlir::UnitAttr VPUIP::arch30xx::ClusterNCERewriter::isSegmentedNCETask(VPUIP::DistributedBufferType inputType) const {
    // Only for explicit SEGMENTED mode, not in combination with
    // DUPLICATED or MULTICASTED
    if (inputType.getDistribution().getMode().getValue() != VPU::DistributionMode::SEGMENTED) {
        return nullptr;
    }

    // Segmentation not present on H axis
    const auto numTiles = parseIntArrayAttr<int64_t>(inputType.getDistribution().getNumTiles());
    if (numTiles[Dims4D::Act::H.ind()] <= 1) {
        return nullptr;
    }

    // Segmentation not supported with non NHWC input such as CM Conv
    if (inputType.getDimsOrder() != DimsOrder::NHWC) {
        return nullptr;
    }

    return mlir::UnitAttr::get(_ctx);
}

namespace {

//
// UnrollClusterTilingPass
//

class UnrollClusterTilingPass final : public VPUIP::arch30xx::UnrollClusterTilingBase<UnrollClusterTilingPass> {
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

std::unique_ptr<mlir::Pass> vpux::VPUIP::arch30xx::createUnrollClusterTilingPass(Logger log) {
    return std::make_unique<UnrollClusterTilingPass>(log);
}
