//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/attributes.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/DenseMap.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <numeric>

using namespace vpux;

namespace {

vpux::NDTypeInterface changeShape(vpux::NDTypeInterface originType, ShapeRef shape, ShapeRef offset) {
    const auto elemType = originType.getElementType();
    if (auto qType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto newQType = tileScalesAndZP(qType, shape, offset);
        auto newType = originType.changeShapeElemType(shape, newQType);
        return VPUIP::tileTypeCompressionScheme(newType, offset, shape);
    }

    auto newType = originType.changeShape(shape);
    return VPUIP::tileTypeCompressionScheme(newType, offset, shape);
}

vpux::NDTypeInterface changeShapeLeaveStrides(vpux::NDTypeInterface originType, StridesRef strides, ShapeRef shape,
                                              ShapeRef offset) {
    VPUX_THROW_UNLESS((originType.isa<mlir::MemRefType>()),
                      "Only MemRefType is supported for 'changeShapeLeaveStrides'. Got '{0}'", originType);

    auto newType = originType;
    const auto elemType = originType.getElementType();
    if (auto qType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto newQType = tileScalesAndZP(qType, shape, offset);
        newType = originType.changeShapeElemType(shape, newQType);
    } else {
        newType = originType.changeShape(shape);
    }

    newType = newType.changeStrides(strides);
    return VPUIP::tileTypeCompressionScheme(newType, offset, shape);
}

//
// ClusterNCEBaseRewriter
//

class ClusterNCEBaseRewriter : public mlir::OpRewritePattern<VPUIP::NCEClusterTaskOp> {
public:
    ClusterNCEBaseRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::NCEClusterTaskOp>(ctx), _log(log), _ctx(ctx) {
        _cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::NCEClusterTaskOp nceTask, mlir::PatternRewriter& rewriter) const final;

protected:
    virtual void getOutputBuffers(SmallVector<mlir::Value>& parentOutputBuffs, SmallVector<mlir::Value>& outputBuffs,
                                  SmallVector<mlir::Value>& parentOutputSparsityMap,
                                  SmallVector<mlir::Value>& outputSparsityMapBuffs,
                                  SmallVector<SmallVector<mlir::Value>>& outputItiBuffs, mlir::Location loc,
                                  VPUIP::NCEClusterTilingOp clusterOp, VPUIP::NCEClusterTaskOp nceTask,
                                  const int64_t numClusters, mlir::PatternRewriter& rewriter) const = 0;

    virtual void getInputBuffers(SmallVector<mlir::Value>& parentInputBuffs, SmallVector<mlir::Value>& inputBuffs,
                                 SmallVector<mlir::Value>& parentInputSparsityMap,
                                 SmallVector<mlir::Value>& inputSparsityMapBuffs,
                                 SmallVector<mlir::Value>& parentInputSETable,
                                 SmallVector<mlir::Value>& inputSETableBuffs, mlir::Location loc,
                                 VPUIP::NCEClusterTilingOp clusterOp, VPUIP::NCEClusterTaskOp nceTask,
                                 const int64_t numClusters, mlir::PatternRewriter& rewriter) const = 0;

    virtual SmallVector<mlir::Value> getWeightsBuffers(mlir::Location loc, VPUIP::NCEClusterTilingOp clusterOp,
                                                       VPUIP::NCEClusterTaskOp nceTask, const int64_t numClusters,
                                                       mlir::PatternRewriter& rewriter) const;

    virtual mlir::UnitAttr isSegmentedNCETask(VPUIP::DistributedBufferType inputType) const = 0;

private:
    SmallVector<mlir::IntegerAttr> getOutChannelOffsets(VPUIP::NCEClusterTaskOp nceTask,
                                                        VPUIP::DistributedBufferType inType,
                                                        VPUIP::DistributedBufferType outType) const;

protected:
    Logger _log;
    mlir::MLIRContext* _ctx;
    mlir::FlatSymbolRefAttr _cmxNameAttr;
};

SmallVector<mlir::IntegerAttr> ClusterNCEBaseRewriter::getOutChannelOffsets(
        VPUIP::NCEClusterTaskOp nceTask, VPUIP::DistributedBufferType inType,
        VPUIP::DistributedBufferType outType) const {
    auto inDistribution = inType.getDistribution();
    auto outDistribution = outType.getDistribution();

    auto inDistributionMode = inDistribution.mode().getValue();
    auto outDistributionMode = outDistribution.mode().getValue();

    const auto numClusters = inDistribution.num_clusters().getInt();

    const auto hasWeightsTable = nceTask.weight_table() != nullptr;
    const auto isSOKMode =
            inDistributionMode == VPU::DistributionMode::DUPLICATED &&
            outDistributionMode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED);
    if (!hasWeightsTable || !isSOKMode) {
        return SmallVector<mlir::IntegerAttr>(numClusters, nullptr);
    }

    const auto perClusterShapeOffsets = outType.getPerClusterComputeShapeOffsets();
    VPUX_THROW_UNLESS(perClusterShapeOffsets.size() == checked_cast<size_t>(numClusters),
                      "Number of shape offsets '{0}' and clusters '{1}' are mismatch", perClusterShapeOffsets.size(),
                      numClusters);

    SmallVector<mlir::IntegerAttr> outChannelOffsets(numClusters);
    for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        outChannelOffsets[clusterId] = getIntAttr(_ctx, perClusterShapeOffsets[clusterId][Dims4D::Act::C]);
    }

    return outChannelOffsets;
}

mlir::LogicalResult ClusterNCEBaseRewriter::matchAndRewrite(VPUIP::NCEClusterTaskOp nceTask,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("Process NCE op: '{0}'", nceTask);
    auto clusterOp = nceTask->getParentOfType<VPUIP::NCEClusterTilingOp>();
    if (clusterOp == nullptr) {
        return mlir::failure();
    }

    auto vpurtTask = clusterOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_UNLESS(vpurtTask != nullptr, "Can't get VPURT task operation");

    auto cycleBeginAttr = vpurtTask->getAttr(cycleBegin);
    auto cycleEndAttr = vpurtTask->getAttr(cycleEnd);
    rewriter.setInsertionPointAfter(vpurtTask);

    VPUX_THROW_UNLESS(!clusterOp.getInputs().empty(), "Wrong inputs size: {0}", clusterOp.getInputs().size());

    const auto hasOnlyDefaultOutput =
            (clusterOp.getOutputs().size() == 1 || clusterOp.getOutputs().size() == 2) && !nceTask.profiling_data();
    const auto hasOutputWithProfiling =
            (clusterOp.getOutputs().size() == 2 || clusterOp.getOutputs().size() == 3) && nceTask.profiling_data();

    VPUX_THROW_UNLESS(hasOnlyDefaultOutput || hasOutputWithProfiling, "Wrong outputs size: {0}",
                      clusterOp.getOutputs().size());

    auto parentInput = *clusterOp.getInputs().begin();
    auto parentOutput = *clusterOp.getOutputs().begin();

    auto parentInputType = parentInput.getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto parentOutputType = parentOutput.getType().dyn_cast<VPUIP::DistributedBufferType>();

    VPUX_THROW_UNLESS(parentInputType != nullptr && parentOutputType != nullptr,
                      "Input and output types must have distributed type. Got: inT={0}, outT={1}", parentInputType,
                      parentOutputType);

    auto inDistribution = parentInputType.getDistribution();
    auto outDistribution = parentOutputType.getDistribution();

    VPUX_THROW_UNLESS(inDistribution.num_clusters() == outDistribution.num_clusters(),
                      "Input '{0}' and output '{1}' number of clusters are not equal", inDistribution.num_clusters(),
                      outDistribution.num_clusters());

    auto numClusters = inDistribution.num_clusters().getInt();

    auto loc = nceTask->getLoc();
    SmallVector<mlir::Value> inputBuffs = {};
    SmallVector<mlir::Value> parentInputBuffs = {};
    SmallVector<mlir::Value> inputSparsityMapBuffs = {};
    SmallVector<mlir::Value> parentInputSparsityMap = {};
    SmallVector<mlir::Value> inputSETableBuffs = {};
    SmallVector<mlir::Value> parentInputSETable = {};

    getInputBuffers(parentInputBuffs, inputBuffs, parentInputSparsityMap, inputSparsityMapBuffs, parentInputSETable,
                    inputSETableBuffs, loc, clusterOp, nceTask, numClusters, rewriter);

    auto weightsBuffs = getWeightsBuffers(loc, clusterOp, nceTask, numClusters, rewriter);
    auto weightsSparsityMapBuffs = VPUIP::getPerClusterMemoryBuffers(
            _ctx, loc, "weightsSparsityMap",
            VPU::getDistributedOperandFromNCEClusterTiling(clusterOp, nceTask.weights_sparsity_map()),
            nceTask.weights_sparsity_map(), numClusters, rewriter);
    auto weightTableBuffs = VPUIP::getPerClusterMemoryBuffers(
            _ctx, loc, "weightTable", VPU::getDistributedOperandFromNCEClusterTiling(clusterOp, nceTask.weight_table()),
            nceTask.weight_table(), numClusters, rewriter);
    auto activationWindowBuffs = VPUIP::getPerClusterMemoryBuffers(
            _ctx, loc, "activationWindow",
            VPU::getDistributedOperandFromNCEClusterTiling(clusterOp, nceTask.activation_window()),
            nceTask.activation_window(), numClusters, rewriter);
    auto instructionListTableBuffs = VPUIP::getPerClusterMemoryBuffers(
            _ctx, loc, "instructionListTable",
            VPU::getDistributedOperandFromNCEClusterTiling(clusterOp, nceTask.instruction_list_table()),
            nceTask.instruction_list_table(), numClusters, rewriter);

    SmallVector<mlir::Value> parentOutputBuffs = {};
    SmallVector<mlir::Value> outputBuffs = {};
    SmallVector<mlir::Value> outputSparsityMapBuffs = {};
    SmallVector<mlir::Value> parentOutputSparsityMap = {};
    SmallVector<SmallVector<mlir::Value>> outputItiBuffs(numClusters);

    getOutputBuffers(parentOutputBuffs, outputBuffs, parentOutputSparsityMap, outputSparsityMapBuffs, outputItiBuffs,
                     loc, clusterOp, nceTask, numClusters, rewriter);

    auto profilingBuffs = VPUIP::getPerClusterMemoryBuffers(
            _ctx, loc, "profilingBuff",
            VPU::getDistributedOperandFromNCEClusterTiling(clusterOp, nceTask.profiling_data()),
            nceTask.profiling_data(), numClusters, rewriter);

    const auto outChannelOffsets = getOutChannelOffsets(nceTask, parentInputType, parentOutputType);

    auto padAttr = nceTask.kernel_paddingAttr();
    SmallVector<VPU::PaddingAttr> padAttrForCluster(numClusters, padAttr);

    // In case of OVERLAPPED mode padding setting in invariant needs to be calculated
    // for each cluster based on distributed type properties
    // However, there might be a case when elementwise operation has OVERLAPPED consumer.
    // In that scenario padding must be calculated only to determine per-cluster shape.
    // Elementwise operations do not support kernel padding.
    const auto isEltwise = (nceTask.task_type() == VPUIP::NCETaskType::ELTWISE);
    auto inDistributionMode = inDistribution.mode().getValue();
    if (inDistributionMode == VPU::DistributionMode::OVERLAPPED && !isEltwise) {
        auto perClusterPadInfo = parentInputType.getPerClusterPadding();
        VPUX_THROW_UNLESS(perClusterPadInfo.size() == static_cast<size_t>(numClusters),
                          "Mismatch between number of padding settings ({0}) and number of clusters ({1})",
                          perClusterPadInfo.size(), numClusters);
        for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
            padAttrForCluster[clusterId] = VPU::getPaddingAttr(_ctx, perClusterPadInfo[clusterId]);
        }
    }

    for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        const auto newLoc = appendLoc(loc, "_cluster_{0}", clusterId);

        mlir::Value profilingData = nullptr;
        mlir::Type profilingOutputType = nullptr;
        mlir::Type outputType = outputBuffs[clusterId].getType();
        mlir::Value outputSparsityMap = nullptr;
        mlir::Type outputSparsityMapType = nullptr;

        if (nceTask.output_sparsity_map_buff()) {
            outputSparsityMap = outputSparsityMapBuffs[clusterId];
            outputSparsityMapType = outputSparsityMap.getType();
        }

        if (nceTask.profiling_data()) {
            profilingOutputType = profilingBuffs[clusterId].getType();
            profilingData = profilingBuffs[clusterId];
        }

        auto newTask = VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
                rewriter, vpurtTask.waitBarriers(), vpurtTask.updateBarriers(), newLoc, outputType,
                outputSparsityMapType, profilingOutputType, inputBuffs[clusterId], inputSparsityMapBuffs[clusterId],
                inputSETableBuffs[clusterId], weightsBuffs[clusterId], weightsSparsityMapBuffs[clusterId],
                weightTableBuffs[clusterId], instructionListTableBuffs[clusterId], activationWindowBuffs[clusterId],
                parentInputBuffs[clusterId], parentInputSparsityMap[clusterId], parentInputSETable[clusterId],
                parentOutputBuffs[clusterId], parentOutputSparsityMap[clusterId],
                outputBuffs[clusterId], outputSparsityMap, profilingData,
                nceTask.task_type(), nceTask.kernel_sizeAttr(), nceTask.kernel_stridesAttr(),
                padAttrForCluster[clusterId], nceTask.activation_window_channel_lengthAttr(),
                nceTask.is_continuedAttr(), nceTask.cm_sp_patternAttr(), isSegmentedNCETask(parentInputType),
                outChannelOffsets[clusterId], nceTask.input_channels_compressionAttr(), nceTask.is_superdenseAttr(),
                nceTask.is_inplaceAttr(), nceTask.input_se_sizeAttr(), nceTask.output_se_sizeAttr(),
                nceTask.is_permute_quantizeAttr());

        for (auto& region : newTask->getRegions()) {
            region.emplaceBlock();
        }

        {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToEnd(&newTask.variants().front());

            for (auto variant : nceTask.variants().getOps<VPUIP::DPUTaskOp>()) {
                VPUX_THROW_UNLESS(variant.cluster_id().hasValue(), "Unable to distribute workload");
                if (variant.cluster_id().getValue() == clusterId) {
                    rewriter.clone(*variant);
                }
            }
        }

        {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToEnd(&newTask.ppe().front());

            for (auto& ppe : nceTask.ppe().getOps()) {
                rewriter.clone(ppe);
            }
        }

        auto newVpurtTask = newTask->getParentOfType<VPURT::TaskOp>();
        if (cycleBeginAttr) {
            newVpurtTask->setAttr(cycleBegin, cycleBeginAttr);
        }
        if (cycleEndAttr) {
            newVpurtTask->setAttr(cycleEnd, cycleEndAttr);
        }

        _log.trace("Insert new NCE task: '{0}'", newTask);
    }

    rewriter.eraseOp(vpurtTask);

    return mlir::success();
}

SmallVector<mlir::Value> ClusterNCEBaseRewriter::getWeightsBuffers(mlir::Location loc,
                                                                   VPUIP::NCEClusterTilingOp clusterOp,
                                                                   VPUIP::NCEClusterTaskOp nceTask,
                                                                   const int64_t numClusters,
                                                                   mlir::PatternRewriter& rewriter) const {
    auto clusterOperand = VPU::getDistributedOperandFromNCEClusterTiling(clusterOp, nceTask.weights());
    if (clusterOperand == nullptr) {
        return SmallVector<mlir::Value>(numClusters, nullptr);
    }

    auto operandType = clusterOperand.getType();
    auto distributedType = operandType.dyn_cast<VPUIP::DistributedBufferType>();
    VPUX_THROW_UNLESS(distributedType != nullptr, "Unsupported operand type {0}", operandType);

    const auto distribution = distributedType.getDistribution();
    const auto distributionMode = distribution.mode().getValue();
    if (distributionMode != (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED)) {
        return VPUIP::getPerClusterMemoryBuffers(
                _ctx, loc, "weights", VPU::getDistributedOperandFromNCEClusterTiling(clusterOp, nceTask.weights()),
                nceTask.weights(), numClusters, rewriter);
    }

    // For weights with Duplicated|Segmented mode, unroll the weight buffer according to its compute shapes and offsets
    auto declBuff = clusterOperand.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(declBuff != nullptr, "Can't get buffer offset for operand: {0}", clusterOperand);
    auto perClusterShapes = distributedType.getPerClusterComputeShapes();
    VPUX_THROW_UNLESS(perClusterShapes.size() == checked_cast<size_t>(numClusters),
                      "Number of shapes '{0}' and clusters '{1}' are mismatch", perClusterShapes.size(), numClusters);
    const auto perClusterShapeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    VPUX_THROW_UNLESS(perClusterShapeOffsets.size() == checked_cast<size_t>(numClusters),
                      "Number of shape offsets '{0}' and clusters '{1}' are mismatch", perClusterShapeOffsets.size(),
                      numClusters);
    const auto tilingScheme = parseIntArrayAttr<int64_t>(distribution.num_tiles());
    const auto axis = vpux::VPU::getDistributedTilingAxis(tilingScheme);
    VPUX_THROW_UNLESS(axis == Dims4D::Act::N.ind(), "Invalid Tile dim, get {0}, expect tiling on N.", axis);

    const auto cmxNameAttr = mlir::FlatSymbolRefAttr::get(_ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    const auto innerOperandType = nceTask.weights().getType().cast<vpux::NDTypeInterface>();
    SmallVector<mlir::Value> perClusterBuffers(numClusters);
    auto insertionPoint = declBuff.getOperation();
    for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        auto cmxBuffType =
                changeShape(innerOperandType, perClusterShapes[clusterId], perClusterShapeOffsets[clusterId]);
        const auto symbolAttr = vpux::IndexedSymbolAttr::get(_ctx, {cmxNameAttr, vpux::getIntAttr(_ctx, clusterId)});
        cmxBuffType = cmxBuffType.changeMemSpace(symbolAttr);
        auto offset = declBuff.byteOffset();
        const auto newLoc = appendLoc(loc, "_weights_cluster_{0}", clusterId);
        offset += Byte(perClusterShapeOffsets[clusterId][Dim(axis)] * distributedType.getStrides()[Dim(axis)]).count();
        auto newCmxBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
                rewriter, insertionPoint, newLoc, cmxBuffType, VPURT::BufferSection::CMX_NN,
                getIntArrayAttr(_ctx, makeArrayRef({clusterId})), offset, declBuff.swizzlingKeyAttr());
        insertionPoint = newCmxBuffer.getOperation();
        perClusterBuffers[clusterId] = newCmxBuffer;
    }
    return perClusterBuffers;
}

//
// ClusterNCERewriterVPUX3XXX
//

class ClusterNCERewriterVPUX3XXX final : public ClusterNCEBaseRewriter {
public:
    ClusterNCERewriterVPUX3XXX(mlir::MLIRContext* ctx, Logger log): ClusterNCEBaseRewriter(ctx, log) {
        setDebugName("ClusterNCERewriterVPUX3XXX");
    }

private:
    void getOutputBuffers(SmallVector<mlir::Value>& parentOutputBuffs, SmallVector<mlir::Value>& outputBuffs,
                          SmallVector<mlir::Value>& parentOutputSparsityMap,
                          SmallVector<mlir::Value>& outputSparsityMapBuffs,
                          SmallVector<SmallVector<mlir::Value>>& outputItiBuffs, mlir::Location loc,
                          VPUIP::NCEClusterTilingOp clusterOp, VPUIP::NCEClusterTaskOp nceTask,
                          const int64_t numClusters, mlir::PatternRewriter& rewriter) const override;

    void getInputBuffers(SmallVector<mlir::Value>& parentInputBuffs, SmallVector<mlir::Value>& inputBuffs,
                         SmallVector<mlir::Value>& parentInputSparsityMap,
                         SmallVector<mlir::Value>& inputSparsityMapBuffs, SmallVector<mlir::Value>& parentInputSETable,
                         SmallVector<mlir::Value>& inputSETableBuffs, mlir::Location loc,
                         VPUIP::NCEClusterTilingOp clusterOp, VPUIP::NCEClusterTaskOp nceTask,
                         const int64_t numClusters, mlir::PatternRewriter& rewriter) const override;

    mlir::UnitAttr isSegmentedNCETask(VPUIP::DistributedBufferType inputType) const override;
};

void ClusterNCERewriterVPUX3XXX::getOutputBuffers(SmallVector<mlir::Value>& parentOutputBuffs,
                                                  SmallVector<mlir::Value>& outputBuffs,
                                                  SmallVector<mlir::Value>& parentOutputSparsityMap,
                                                  SmallVector<mlir::Value>& outputSparsityMapBuffs,
                                                  SmallVector<SmallVector<mlir::Value>>& /*outputItiBuffs*/,
                                                  mlir::Location loc, VPUIP::NCEClusterTilingOp clusterOp,
                                                  VPUIP::NCEClusterTaskOp nceTask, const int64_t numClusters,
                                                  mlir::PatternRewriter& rewriter) const {
    auto parentInputType = (*clusterOp.getInputs().begin()).getType().cast<VPUIP::DistributedBufferType>();
    auto parentOutputType = (*clusterOp.getOutputs().begin()).getType().cast<VPUIP::DistributedBufferType>();

    auto inDistribution = parentInputType.getDistribution();
    auto outDistribution = parentOutputType.getDistribution();

    auto inDistributionMode = inDistribution.mode().getValue();
    auto outDistributionMode = outDistribution.mode().getValue();
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
    if (VPUIP::isSegmentedOverC(outDistribution)) {
        // for SEG SOK parent output buffers = output buffers
        parentOutputBuffs = outputBuffs;
    }
}

mlir::UnitAttr ClusterNCERewriterVPUX3XXX::isSegmentedNCETask(VPUIP::DistributedBufferType inputType) const {
    // Only for explicit SEGMENTED mode, not in combination with
    // DUPLICATED or MULTICASTED
    if (inputType.getDistribution().mode().getValue() != VPU::DistributionMode::SEGMENTED) {
        return nullptr;
    }

    // Segmentation not present on H axis
    const auto numTiles = parseIntArrayAttr<int64_t>(inputType.getDistribution().num_tiles());
    if (numTiles[Dims4D::Act::H.ind()] <= 1) {
        return nullptr;
    }

    // Segmentation not supported with non NHWC input such as CM Conv
    if (inputType.getDimsOrder() != DimsOrder::NHWC) {
        return nullptr;
    }

    return mlir::UnitAttr::get(_ctx);
}

void ClusterNCERewriterVPUX3XXX::getInputBuffers(SmallVector<mlir::Value>& parentInputBuffs,
                                                 SmallVector<mlir::Value>& inputBuffs,
                                                 SmallVector<mlir::Value>& parentInputSparsityMap,
                                                 SmallVector<mlir::Value>& inputSparsityMapBuffs,
                                                 SmallVector<mlir::Value>& parentInputSETable,
                                                 SmallVector<mlir::Value>& inputSETableBuffs, mlir::Location loc,
                                                 VPUIP::NCEClusterTilingOp clusterOp, VPUIP::NCEClusterTaskOp nceTask,
                                                 const int64_t numClusters, mlir::PatternRewriter& rewriter) const {
    inputBuffs = VPUIP::getPerClusterMemoryBuffers(
            _ctx, loc, "input", VPU::getDistributedOperandFromNCEClusterTiling(clusterOp, nceTask.input()),
            nceTask.input(), numClusters, rewriter);

    auto parentInput = *clusterOp.getInputs().begin();
    auto parentInputType = parentInput.getType().cast<VPUIP::DistributedBufferType>();
    mlir::UnitAttr isSegmented = isSegmentedNCETask(parentInputType);

    auto arch = VPU::getArch(nceTask);
    bool isDWOpAndNeedsAlign = VPU::isDWOpAndNeedsAlign(arch, nceTask.task_type());
    for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        // For 37XX and 30XX arch, ensure we have H_per_cluster x W as a multiple of 4.
        if (isSegmented && clusterId != (numClusters - 1) &&
            (nceTask.task_type() == VPUIP::NCETaskType::CONV || isDWOpAndNeedsAlign)) {
            auto inShape = inputBuffs[clusterId].getType().cast<NDTypeInterface>().getShape();
            const auto hAlignment = VPU::getSOHPerClusterHeightAlignment(inShape[Dims4D::Filter::KX]);

            VPUX_THROW_UNLESS((inShape[Dims4D::Filter::KY] % hAlignment) == 0,
                              "For segmented cluster we must have alignment to {0}, type: {1}", hAlignment,
                              inputBuffs[clusterId].getType());
        }
    }

    parentInputBuffs = VPUIP::isSegmentedOverC(parentInputType.getDistribution())
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

    parentInputSparsityMap = SmallVector<mlir::Value>(
            numClusters, VPU::getDistributedOperandFromNCEClusterTiling(clusterOp, nceTask.input_sparsity_map()));
    parentInputSETable = SmallVector<mlir::Value>(
            numClusters,
            VPU::getDistributedOperandFromNCEClusterTiling(clusterOp, nceTask.input_storage_element_table()));
}

//
// ClusterDMARewriter
//

class ClusterDMARewriter final : public mlir::OpRewritePattern<VPUIP::NNDMAOp> {
public:
    ClusterDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount, Logger log)
            : mlir::OpRewritePattern<VPUIP::NNDMAOp>(ctx), _log(log), _ctx(ctx), _dmaPortCount(dmaPortCount) {
        setDebugName("ClusterDMARewriter");

        _cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::NNDMAOp nndmaOp, mlir::PatternRewriter& rewriter) const final;

private:
    void unrollSegmentedOrOverlapped(mlir::Location loc, VPUIP::NCEClusterTilingOp clusterOp, VPURT::TaskOp vpurtTask,
                                     VPUIP::DistributedBufferType distributedType,
                                     mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
    mlir::MLIRContext* _ctx;
    int64_t _dmaPortCount;
    mlir::FlatSymbolRefAttr _cmxNameAttr;
};

void ClusterDMARewriter::unrollSegmentedOrOverlapped(mlir::Location loc, VPUIP::NCEClusterTilingOp clusterOp,
                                                     VPURT::TaskOp vpurtTask,
                                                     VPUIP::DistributedBufferType distributedType,
                                                     mlir::PatternRewriter& rewriter) const {
    const auto input = *clusterOp.getInputs().begin();
    const auto output = *clusterOp.getOutputs().begin();

    const auto inputType = input.getType().cast<vpux::NDTypeInterface>();
    const auto outputType = output.getType().cast<vpux::NDTypeInterface>();

    const auto innerInput = *clusterOp.getInnerInputs().begin();
    const auto innerOutput = *clusterOp.getInnerOutputs().begin();

    const auto innerInputType = innerInput.getType().cast<vpux::NDTypeInterface>();
    const auto innerOutputType = innerOutput.getType().cast<vpux::NDTypeInterface>();

    const auto distributionAttr = distributedType.getDistribution();
    const auto numClusters = distributionAttr.num_clusters().getInt();

    const auto originInShape = inputType.getShape().raw();
    const auto originOutShape = outputType.getShape().raw();

    auto cycleBeginAttr = vpurtTask->getAttr(cycleBegin);
    auto cycleEndAttr = vpurtTask->getAttr(cycleEnd);

    const auto strideInReqs = StrideReqs::compact(originInShape.size());
    const auto strideOutReqs = StrideReqs::compact(originOutShape.size());

    if (!strideInReqs.checkStrides(input)) {
        _log.trace("DMA at {0} is not compact for the input, strides = {1}, shape = {2}", loc, inputType.getStrides(),
                   originInShape);
    }
    if (!strideOutReqs.checkStrides(output)) {
        _log.trace("DMA at {0} is not compact for the output, strides = {1}, shape = {2}", loc, outputType.getStrides(),
                   originOutShape);
    }

    const auto numTiles = parseIntArrayAttr<int64_t>(distributionAttr.num_tiles());
    VPUX_THROW_UNLESS(originInShape.size() == numTiles.size(),
                      "Input shape size '{0}' and tiles array size '{1}' are mismatch", originInShape.size(),
                      numTiles.size());

    const auto perClusterShapes = distributedType.getPerClusterMemoryShapes();
    VPUX_THROW_UNLESS(perClusterShapes.size() == checked_cast<size_t>(numClusters),
                      "Number of shapes '{0}' and clusters '{1}' are mismatch", perClusterShapes.size(), numClusters);
    const auto perClusterShapeOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    VPUX_THROW_UNLESS(perClusterShapeOffsets.size() == checked_cast<size_t>(numClusters),
                      "Number of shape offsets '{0}' and clusters '{1}' are mismatch", perClusterShapeOffsets.size(),
                      numClusters);

    const auto tileInnerType = [&](vpux::NDTypeInterface innerType) {
        SmallVector<vpux::NDTypeInterface> newTypes(numClusters);
        for (size_t clusterId = 0; clusterId < perClusterShapes.size(); ++clusterId) {
            newTypes[clusterId] =
                    changeShape(innerType, perClusterShapes[clusterId], perClusterShapeOffsets[clusterId]);
        }

        return newTypes;
    };

    const auto tileInnerTypeLeaveStrides = [&](vpux::NDTypeInterface innerType, StridesRef strides) {
        SmallVector<vpux::NDTypeInterface> newTypes(numClusters);
        for (size_t clusterId = 0; clusterId < perClusterShapes.size(); ++clusterId) {
            newTypes[clusterId] = changeShapeLeaveStrides(innerType, strides, perClusterShapes[clusterId],
                                                          perClusterShapeOffsets[clusterId]);
        }

        return newTypes;
    };

    const auto tilingScheme = parseIntArrayAttr<int64_t>(distributionAttr.num_tiles());
    const auto axis = vpux::VPU::getDistributedTilingAxis(tilingScheme);
    const auto strides = distributedType.getStrides();

    // Check if per-cluster DMA input will not be a contiguous block of memory.
    // In such case DMA input buffers should have strides according to parent input tensor
    bool useParentTensorStridesForInput = !strideInReqs.checkStrides(input);
    bool useParentTensorStridesForOutput = !strideOutReqs.checkStrides(output);
    // ODU permutations enabled, and tested only for SOH and NCHW order
    // also middle network permutations are disabled for now [Track number: S#67423]
    const bool tileNCHWOutOverH = numTiles.size() == 4 && numTiles[Dims4D::Act::N.ind()] == 1 &&
                                  numTiles[Dims4D::Act::C.ind()] == 1 && numTiles[Dims4D::Act::H.ind()] > 1 &&
                                  numTiles[Dims4D::Act::W.ind()] == 1 && inputType.getDimsOrder() == DimsOrder::NCHW &&
                                  outputType.getDimsOrder() == DimsOrder::NCHW;

    // ClusterTiling DMA only has distributedType on one side. If the distributedType is not memory contiguous with
    // the tiling, per-cluster DMA will need stride access on the non-distributed side.

    if (!isMemoryContiguousWithTiling(distributedType)) {
        if (input.getType().isa<VPUIP::DistributedBufferType>()) {
            useParentTensorStridesForOutput = true;
        }

        if (output.getType().isa<VPUIP::DistributedBufferType>()) {
            useParentTensorStridesForInput = true;
        }
    }

    const auto inTypes =
            (useParentTensorStridesForInput ? tileInnerTypeLeaveStrides(innerInputType, inputType.getStrides())
                                            : tileInnerType(innerInputType));
    const auto outTypes =
            (useParentTensorStridesForOutput ? tileInnerTypeLeaveStrides(innerOutputType, outputType.getStrides())
                                             : tileInnerType(innerOutputType));

    const auto getOperand = [&](int64_t clusterId, mlir::Value operand, vpux::NDTypeInterface newType,
                                bool isTiledOperand, mlir::Operation* insertionPoint) -> mlir::Value {
        // For example, copy of weights in case of SOK
        // <32x16x1x1xfp16, @DDR>  -> <16x16x1x1xfp16, [@CMX, 0]>
        //                         -> <16x16x1x1xfp16, [@CMX, 1]>
        if (auto cst = operand.getDefiningOp<Const::DeclareOp>()) {
            VPUX_THROW_UNLESS(outputType.getMemoryKind() == VPU::MemoryKind::CMX_NN,
                              "Output operand type must have NN_CMX memory space. Got: {0}",
                              outputType.getMemoryKind());

            return rewriter.create<VPUIP::SubViewOp>(loc, cst, perClusterShapeOffsets[clusterId].raw(),
                                                     perClusterShapes[clusterId].raw());
        }

        auto declBuff = operand.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(declBuff != nullptr, "Can't get buffer offset for operand: {0}", operand);

        if (isTiledOperand) {
            const auto symbolAttr =
                    vpux::IndexedSymbolAttr::get(_ctx, {_cmxNameAttr, vpux::getIntAttr(_ctx, clusterId)});
            auto newCMXType = newType.changeMemSpace(symbolAttr);
            if (tileNCHWOutOverH) {
                const auto shape = newCMXType.getShape();
                const auto strides = newCMXType.getStrides();
                const int64_t dimC = shape[Dims4D::Act::C];
                const int64_t dimH = shape[Dims4D::Act::H];
                const Bit strideW = strides[Dims4D::Act::W];
                const Bit strideH = strides[Dims4D::Act::H];
                const Bit strideC = strideH * dimH;
                const Bit strideN = strideC * dimC;
                const auto newStrides = SmallVector<Bit>{strideN, strideC, strideH, strideW};
                newCMXType = newCMXType.changeStrides(StridesRef(newStrides));
            }

            return VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, loc, newCMXType,
                                                           VPURT::BufferSection::CMX_NN,
                                                           getIntArrayAttr(_ctx, makeArrayRef({clusterId})),
                                                           declBuff.byteOffset(), declBuff.swizzlingKeyAttr());
        }

        // For example, copy of input in case of SOH
        // <1x16x33x32xf16, @DDR>  -> <1x16x17x32xf16, [@CMX, 0]>
        //                         -> <1x16x16x32xf16, [@CMX, 1]>

        // OR copy back of output in case of SOH
        // <1x16x17x32xf16, [@CMX, 0]>  -> <1x16x33x32xf16, @DDR>
        // <1x16x16x32xf16, [@CMX, 1]>  /

        // OR copy data from cmx to cmx
        // <1x16x17x32xf16, [@CMX, 0]>  -> <1x16x33x32xf16, [@CMX, 0]>
        // <1x16x16x32xf16, [@CMX, 1]>  /

        Byte offset{declBuff.byteOffset()};
        offset += static_cast<Byte>(perClusterShapeOffsets[clusterId][Dim(axis)] * newType.getStrides()[Dim(axis)]);

        auto section = declBuff.section();
        auto sectionIndex = declBuff.sectionIndex();

        vpux::IndexedSymbolAttr symbolAttr;
        if (newType.getMemoryKind() == VPU::MemoryKind::CMX_NN) {
            VPUX_THROW_UNLESS(sectionIndex.hasValue(), "Cannot get section index for {0}", declBuff);
            auto sectionIndexVal = parseIntArrayAttr<int64_t>(sectionIndex.getValue());
            VPUX_THROW_UNLESS(sectionIndexVal.size() == 1, "Invalid section index list size for {0}", declBuff);

            symbolAttr = vpux::IndexedSymbolAttr::get(_ctx, stringifyEnum(VPURT::getMemoryKind(section)),
                                                      static_cast<size_t>(sectionIndexVal[0]));
        } else {
            symbolAttr = vpux::IndexedSymbolAttr::get(_ctx, stringifyEnum(VPURT::getMemoryKind(section)));
        }
        newType = newType.changeMemSpace(symbolAttr);
        if (tileNCHWOutOverH) {
            const auto shape = newType.getShape();
            const auto strides = newType.getStrides();
            const int64_t dimC = shape[Dims4D::Act::C];
            const int64_t parentDimH = distributedType.getShape()[Dims4D::Act::H];
            const Bit strideW = strides[Dims4D::Act::W];
            const Bit strideH = strides[Dims4D::Act::H];
            const Bit strideC = strideH * parentDimH;
            const Bit strideN = strideC * dimC;
            const auto newStrides = SmallVector<Bit>{strideN, strideC, strideH, strideW};
            const auto strideReqs = StrideReqs::compact(newType.getRank());
            if (strideReqs.checkStrides(newType)) {
                newType = newType.changeStrides(StridesRef(newStrides));
            }
        }

        if (sectionIndex.hasValue()) {
            return VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, loc, newType, section,
                                                           sectionIndex.getValue(), offset.count(),
                                                           declBuff.swizzlingKeyAttr());
        }
        return VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, loc, newType, section, nullptr,
                                                       offset.count(), declBuff.swizzlingKeyAttr());
    };

    double origCycleCost = 0.0;
    auto runInParallel = _dmaPortCount > 1;
    int64_t unrolledDMACycleBegin = 0;
    if (cycleBeginAttr && cycleEndAttr) {
        unrolledDMACycleBegin = cycleBeginAttr.cast<mlir::IntegerAttr>().getInt();
        origCycleCost = static_cast<double>(cycleEndAttr.cast<mlir::IntegerAttr>().getInt() -
                                            cycleBeginAttr.cast<mlir::IntegerAttr>().getInt());
    }

    auto isDistributedInput = inputType.isa<VPUIP::DistributedBufferType>();
    auto isDistributedOutput = outputType.isa<VPUIP::DistributedBufferType>();

    auto origNNDMA = clusterOp.getInnerTaskOpOfType<VPUIP::NNDMAOp>();
    VPUX_THROW_WHEN(origNNDMA == nullptr, "Inner task is not NNDMA");
    auto spillIdAttr = origNNDMA.spillIdAttr();
    auto inputInsertionPoint = input.getDefiningOp();
    auto outputInsertionPoint = output.getDefiningOp();
    for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        const auto newInputType = inTypes[clusterId];
        const auto newOutType = outTypes[clusterId];

        const auto inputBuffer = getOperand(clusterId, input, newInputType, isDistributedInput, inputInsertionPoint);
        inputInsertionPoint = inputBuffer.getDefiningOp();
        _log.trace("Insert new input buffer declaration: '{0}'", inputBuffer);

        const auto outBuffer = getOperand(clusterId, output, newOutType, isDistributedOutput, outputInsertionPoint);
        outputInsertionPoint = outBuffer.getDefiningOp();
        _log.trace("Insert new output buffer declaration: '{0}'", outBuffer);

        const auto newLoc = appendLoc(loc, "_cluster_{0}", clusterId);
        auto newDMAPort = clusterId % _dmaPortCount;
        const auto newNNDMA = VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
                rewriter, vpurtTask.waitBarriers(), vpurtTask.updateBarriers(), newLoc, inputBuffer, outBuffer,
                newDMAPort, origNNDMA.channelTypeAttr(), false, false, spillIdAttr, nullptr);
        _log.trace("Insert new NNDMA op: '{0}'", newNNDMA);

        auto newVpurtTask = newNNDMA->getParentOfType<VPURT::TaskOp>();
        // Calculate the cycle info for the unrolled DMA op.
        // [Track number: E#48048]
        // 1. For multi dma ports, the original cycle is expected to take multi ports into consideration that the
        // unrolled tasks will execute in parallel. So the new cycle info is same as the original one.
        // 2. For single dma port, the task is expected to run in sequence. So the new cycle should be updated
        // according to the unrolled dma data size.
        if (cycleBeginAttr && cycleEndAttr) {
            if (runInParallel) {
                newVpurtTask->setAttr(cycleBegin, cycleBeginAttr);
                newVpurtTask->setAttr(cycleEnd, cycleEndAttr);
            } else {
                auto newCycleCost = static_cast<double>(newInputType.getTotalAllocSize().count()) /
                                    innerInputType.getTotalAllocSize().count() * origCycleCost;
                newVpurtTask->setAttr(cycleBegin, vpux::getIntAttr(rewriter, unrolledDMACycleBegin));
                unrolledDMACycleBegin = std::min(static_cast<int64_t>(unrolledDMACycleBegin + newCycleCost),
                                                 cycleEndAttr.cast<mlir::IntegerAttr>().getInt());
                newVpurtTask->setAttr(cycleEnd, vpux::getIntAttr(rewriter, unrolledDMACycleBegin));
            }
        }
    }
}

mlir::LogicalResult ClusterDMARewriter::matchAndRewrite(VPUIP::NNDMAOp nndmaOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Process NNDMA op: {0}", nndmaOp);

    auto clusterOp = nndmaOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
    if (clusterOp == nullptr) {
        _log.trace("NNDMA is not a child of NCEClusterTiling op");
        return mlir::failure();
    }

    VPUX_THROW_UNLESS(clusterOp.getInputs().size() == 1, "Wrong inputs size: {0}", clusterOp.getInputs().size());
    VPUX_THROW_UNLESS(clusterOp.getOutputs().size() == 1, "Wrong outputs size: {0}", clusterOp.getOutputs().size());

    const auto input = *clusterOp.getInputs().begin();
    const auto output = *clusterOp.getOutputs().begin();

    const auto inputType = input.getType();
    const auto outputType = output.getType();

    VPUX_THROW_UNLESS(clusterOp.getInnerInputs().size() == 1, "Wrong inputs size: {0}",
                      clusterOp.getInnerInputs().size());
    VPUX_THROW_UNLESS(clusterOp.getInnerOutputs().size() == 1, "Wrong outputs size: {0}",
                      clusterOp.getInnerOutputs().size());

    auto vpurtTask = clusterOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_UNLESS(vpurtTask != nullptr, "Can't get VPURT task operation");
    auto cycleBeginAttr = vpurtTask->getAttr(cycleBegin);
    auto cycleEndAttr = vpurtTask->getAttr(cycleEnd);
    rewriter.setInsertionPointAfter(vpurtTask);

    const auto distributedType = inputType.isa<VPUIP::DistributedBufferType>()
                                         ? inputType.dyn_cast<VPUIP::DistributedBufferType>()
                                         : outputType.dyn_cast<VPUIP::DistributedBufferType>();

    VPUX_THROW_UNLESS(distributedType != nullptr, "One of operands must have DistributedBuffer type");
    VPUX_THROW_WHEN(inputType.isa<VPUIP::DistributedBufferType>() && outputType.isa<VPUIP::DistributedBufferType>(),
                    "Only one operand can have DistributedBuffer type");

    const auto loc = nndmaOp->getLoc();
    const auto distributionAttr = distributedType.getDistribution();
    const auto mode = distributionAttr.mode().getValue();
    auto spillIdAttr = nndmaOp.spillIdAttr();
    if (mode == VPU::DistributionMode::SEGMENTED || mode == VPU::DistributionMode::OVERLAPPED) {
        _log.trace("Process {0} mode", VPU::stringifyDistributionMode(mode));
        unrollSegmentedOrOverlapped(loc, clusterOp, vpurtTask, distributedType, rewriter);
    } else if (outputType.isa<VPUIP::DistributedBufferType>() &&
               (VPU::bitEnumContains(mode, VPU::DistributionMode::DUPLICATED) ||
                mode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::MULTICASTED))) {
        // For example, copy of weights in case of SOH (output type is DUPLICATED)
        // Or copy spilled input of NCE task in case of SOK (output type is DUPLICATED|SEGMENTED)
        // <16x16x1x1xf16, @DDR> -> <16x16x1x1xf16, [@CMX, 0]>
        //                       -> <16x16x1x1xf16, [@CMX, 1]>
        // SEGMENTED|MULTICASTED which can be a result of spilling of NCE output in SEGMENTED|MULTICASTED mode
        // should be treated as a DUPLICATED mode

        _log.trace("Process DUPLICATED output");

        auto outDeclBuff = output.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(outDeclBuff != nullptr, "Can't get output buffer offset");

        const auto numClusters = distributionAttr.num_clusters().getInt();
        SmallVector<int64_t> clusters(numClusters);
        std::iota(clusters.begin(), clusters.end(), 0);

        auto cmxBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
                rewriter, outDeclBuff, loc, outDeclBuff.getType(), VPURT::BufferSection::CMX_NN,
                getIntArrayAttr(_ctx, clusters), outDeclBuff.byteOffset(), outDeclBuff.swizzlingKeyAttr());

        _log.trace("Insert new CMX buffer declaration: '{0}'", cmxBuffer);

        const auto newLoc = appendLoc(loc, "_broadcast_copy_to_CMX[{0},{1}]", clusters.front(), clusters.back());
        const auto newNNDMA = VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
                rewriter, vpurtTask.waitBarriers(), vpurtTask.updateBarriers(), newLoc, input, cmxBuffer,
                nndmaOp.port(), nndmaOp.channelTypeAttr(), false, false, spillIdAttr, nullptr);
        _log.trace("Insert new NNDMA op: '{0}'", newNNDMA);

        auto newVpurtTask = newNNDMA->getParentOfType<VPURT::TaskOp>();
        if (cycleBeginAttr) {
            newVpurtTask->setAttr(cycleBegin, cycleBeginAttr);
        }
        if (cycleEndAttr) {
            newVpurtTask->setAttr(cycleEnd, cycleEndAttr);
        }
    } else if (inputType.isa<VPUIP::DistributedBufferType>() &&
               (VPU::bitEnumContains(mode, VPU::DistributionMode::DUPLICATED) ||
                VPU::bitEnumContains(mode, VPU::DistributionMode::MULTICASTED))) {
        // For example, copy back of output of NCE task in case of SOK (input type is DUPLICATED|SEGMENTED)
        // Or copy output of NCE task in case of Clustering strategy (input type is DUPLICATED)
        // Or copy output of NCE task in case of HKSwitch strategy (input type is MULTICASTED|SEGMENTED)
        // <1x32x32x32xf16, [@CMX, 0]> -> <1x32x32x32xf16, @DDR>
        // <1x32x32x32xf16, [@CMX, 1]>

        _log.trace("Process DUPLICATED|SEGMENTED input");

        auto inDeclBuff = input.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(inDeclBuff != nullptr, "Can't get input buffer offset");

        const auto symbolAttr = vpux::IndexedSymbolAttr::get(_ctx, {_cmxNameAttr, vpux::getIntAttr(_ctx, 0)});

        const auto innerInput = *clusterOp.getInnerInputs().begin();
        const auto innerInputType = innerInput.getType().cast<vpux::NDTypeInterface>();
        const auto newInType = innerInputType.changeMemSpace(symbolAttr);

        auto cmxBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
                rewriter, inDeclBuff, loc, newInType, VPURT::BufferSection::CMX_NN,
                getIntArrayAttr(_ctx, makeArrayRef({0})), inDeclBuff.byteOffset(), inDeclBuff.swizzlingKeyAttr());

        _log.trace("Insert new CMX buffer declaration: '{0}'", cmxBuffer);

        const auto newNNDMA = VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
                rewriter, vpurtTask.waitBarriers(), vpurtTask.updateBarriers(), loc, cmxBuffer, output, nndmaOp.port(),
                nndmaOp.channelTypeAttr(), false, false, spillIdAttr, nullptr);
        _log.trace("Insert new NNDMA op: '{0}'", newNNDMA);
        auto newVpurtTask = newNNDMA->getParentOfType<VPURT::TaskOp>();
        if (cycleBeginAttr) {
            newVpurtTask->setAttr(cycleBegin, cycleBeginAttr);
        }
        if (cycleEndAttr) {
            newVpurtTask->setAttr(cycleEnd, cycleEndAttr);
        }
    } else {
        VPUX_THROW("Unsupported distribution mode: {0}", VPU::stringifyDistributionMode(mode));
    }

    rewriter.eraseOp(vpurtTask);

    return mlir::success();
}

//
// ClusterSWRewriter
//

class ClusterSWRewriter final : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    ClusterSWRewriter(mlir::MLIRContext* ctx, mlir::ModuleOp module, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _log(log), _ctx(ctx), _module(module) {
        setDebugName("ClusterSWRewriter");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp nceTask, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    mlir::MLIRContext* _ctx;
    mlir::ModuleOp _module;
};

mlir::LogicalResult ClusterSWRewriter::matchAndRewrite(VPUIP::SwKernelOp swTask,
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

    const auto hasOnlyDefaultOutput = clusterOp.getOutputs().size() == 1 && !swTask.profiling_data();
    const auto hasOutputWithProfiling = clusterOp.getOutputs().size() == 2 && swTask.profiling_data();

    VPUX_THROW_UNLESS(hasOnlyDefaultOutput || hasOutputWithProfiling, "Wrong outputs size: {0}",
                      clusterOp.getOutputs().size());

    auto parentInput = *clusterOp.getInputs().begin();
    auto parentOutput = *clusterOp.getOutputs().begin();

    auto parentInputType = parentInput.getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto parentOutputType = parentOutput.getType().dyn_cast<VPUIP::DistributedBufferType>();

    VPUX_THROW_UNLESS(parentInputType != nullptr && parentOutputType != nullptr,
                      "Input and output types must have distributed type. Got: inT={0}, outT={1}", parentInputType,
                      parentOutputType);

    auto inDistribution = parentInputType.getDistribution();
    auto outDistribution = parentOutputType.getDistribution();

    VPUX_THROW_UNLESS(inDistribution.num_clusters() == outDistribution.num_clusters(),
                      "Input '{0}' and output '{1}' number of clusters are not equal", inDistribution.num_clusters(),
                      outDistribution.num_clusters());

    auto inDistributionMode = inDistribution.mode().getValue();
    auto outDistributionMode = outDistribution.mode().getValue();
    VPUX_THROW_WHEN(outDistributionMode == VPU::DistributionMode::OVERLAPPED,
                    "No support for SW op output in OVERLAPPED mode.");
    VPUX_THROW_WHEN(inDistributionMode == VPU::DistributionMode::OVERLAPPED &&
                            outDistributionMode != VPU::DistributionMode::SEGMENTED,
                    "When SW op has input in OVERLAPPED mode then output must be segmented. out mode = '{0}'",
                    VPU::stringifyDistributionMode(outDistributionMode));

    auto numClusters = inDistribution.num_clusters().getInt();
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
        VPUX_THROW_UNLESS(outTileIndex.hasValue(), "Can not get tiling dim for {0}", parentOutputType);
        for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
            SmallVector<TileInfo> tiles;
            for (auto operand : parentInputBuffs) {
                auto clusterOperand = VPU::getDistributedOperandFromNCEClusterTiling(clusterOp, operand);
                auto distributedType = clusterOperand.getType().dyn_cast<VPUIP::DistributedBufferType>();
                auto tileIndex = VPUIP::getTilingDimIndex(distributedType);
                VPUX_THROW_UNLESS(tileIndex.hasValue(), "Can not get tiling dim for {0}", distributedType);
                auto tileInfo = getPerClusterTileInfo(distributedType.getPerClusterMemoryShapes()[clusterId],
                                                      distributedType.getPerClusterMemoryShapeOffsets()[clusterId],
                                                      tileIndex.getValue());
                tiles.push_back(tileInfo);
            }
            auto inTiles = TilingInfo(tiles);
            auto outTile = getPerClusterTileInfo(parentOutputType.getPerClusterComputeShapes()[clusterId],
                                                 parentOutputType.getPerClusterComputeShapeOffsets()[clusterId],
                                                 outTileIndex.getValue());
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
            if (kernelRun.attrs().hasValue()) {
                const mlir::ArrayAttr arrayAttrs = kernelRun.attrs().getValue();
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
        const auto newLoc = appendLoc(loc, "_cluster_{0}", clusterId);
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
                rewriter, vpurtTask.waitBarriers(), vpurtTask.updateBarriers(), newLoc, inputBuffs[clusterId],
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

//
// UnrollClusterTilingPass
//

class UnrollClusterTilingPass final : public VPUIP::UnrollClusterTilingBase<UnrollClusterTilingPass> {
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
    patterns.add<ClusterDMARewriter>(&ctx, dmaPortCount, _log);
    patterns.add<ClusterSWRewriter>(&ctx, module, _log);

    patterns.add<ClusterNCERewriterVPUX3XXX>(&ctx, _log);

    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUnrollClusterTilingPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUnrollClusterTilingPass(Logger log) {
    return std::make_unique<UnrollClusterTilingPass>(log);
}
