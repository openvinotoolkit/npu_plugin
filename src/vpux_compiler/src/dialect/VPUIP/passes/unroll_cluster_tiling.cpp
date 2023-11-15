//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/passes/unroll_cluster_tiling.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"

#include "vpux/compiler/core/cost_model_utils.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

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

VPUIP::DpuProfilingMetadataAttr extendDpuProfAttrWithClusterInfo(VPUIP::DpuProfilingMetadataAttr metaAttr,
                                                                 unsigned numVariants, unsigned clusterId) {
    mlir::MLIRContext* ctx = metaAttr.getContext();
    return VPUIP::DpuProfilingMetadataAttr::get(ctx, metaAttr.getBufferId(), metaAttr.getTaskId(),
                                                metaAttr.getMaxVariants(), getIntAttr(ctx, numVariants),
                                                getIntAttr(ctx, clusterId));
}

void updateProfilingMetadata(VPUIP::NCEClusterTaskOp nceTask, VPUIP::NCEClusterTaskOp newTask, int64_t clusterId) {
    if (nceTask.profiling_data() == nullptr) {
        return;
    }

    const auto oldMetadata = nceTask.profilingMetadataAttr();
    VPUX_THROW_WHEN(oldMetadata == nullptr, "Missed profiling attribute for '{0}'.", nceTask);

    const auto variantsRange = newTask.variants().getOps<VPUIP::DPUTaskOp>();
    const auto numVariants = std::distance(variantsRange.begin(), variantsRange.end());
    newTask.profilingMetadataAttr(extendDpuProfAttrWithClusterInfo(oldMetadata, numVariants, clusterId));
}

}  // namespace

//
// ClusterNCEBaseRewriter
//

SmallVector<mlir::IntegerAttr> VPUIP::ClusterNCEBaseRewriter::getOutChannelOffsets(
        VPUIP::NCEClusterTaskOp nceTask, VPUIP::DistributedBufferType inType,
        VPUIP::DistributedBufferType outType) const {
    auto inDistribution = inType.getDistribution();
    auto outDistribution = outType.getDistribution();

    auto inDistributionMode = inDistribution.getMode().getValue();
    auto outDistributionMode = outDistribution.getMode().getValue();

    const auto numClusters = inDistribution.getNumClusters().getInt();

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

mlir::LogicalResult VPUIP::ClusterNCEBaseRewriter::matchAndRewrite(VPUIP::NCEClusterTaskOp nceTask,
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

    VPUX_THROW_UNLESS(inDistribution.getNumClusters() == outDistribution.getNumClusters(),
                      "Input '{0}' and output '{1}' number of clusters are not equal", inDistribution.getNumClusters(),
                      outDistribution.getNumClusters());

    auto numClusters = inDistribution.getNumClusters().getInt();

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

    getOutputBuffers(parentOutputBuffs, outputBuffs, parentOutputSparsityMap, outputSparsityMapBuffs, loc, clusterOp,
                     nceTask, numClusters, rewriter);

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
    auto inDistributionMode = inDistribution.getMode().getValue();
    if (inDistributionMode == VPU::DistributionMode::OVERLAPPED && !isEltwise) {
        auto nceTaskKernelPadValue = PadInfo(padAttr.getLeft().getInt(), padAttr.getRight().getInt(),
                                             padAttr.getTop().getInt(), padAttr.getBottom().getInt());
        auto perClusterPadInfo = parentInputType.getPerClusterPadding(nceTaskKernelPadValue);
        VPUX_THROW_UNLESS(perClusterPadInfo.size() == static_cast<size_t>(numClusters),
                          "Mismatch between number of padding settings ({0}) and number of clusters ({1})",
                          perClusterPadInfo.size(), numClusters);
        for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
            padAttrForCluster[clusterId] = VPU::getPaddingAttr(_ctx, perClusterPadInfo[clusterId]);
        }
    }

    for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        const auto newLoc = appendLoc(loc, "cluster_{0}", clusterId);

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
                rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), newLoc, outputType,
                outputSparsityMapType, profilingOutputType, inputBuffs[clusterId], inputSparsityMapBuffs[clusterId],
                inputSETableBuffs[clusterId], weightsBuffs[clusterId], weightsSparsityMapBuffs[clusterId],
                weightTableBuffs[clusterId], instructionListTableBuffs[clusterId], activationWindowBuffs[clusterId],
                parentInputBuffs[clusterId], parentInputSparsityMap[clusterId], parentInputSETable[clusterId],
                parentOutputBuffs[clusterId], parentOutputSparsityMap[clusterId], outputBuffs[clusterId],
                outputSparsityMap, profilingData, nceTask.task_type(), nceTask.kernel_sizeAttr(),
                nceTask.kernel_stridesAttr(), padAttrForCluster[clusterId],
                nceTask.activation_window_channel_lengthAttr(), nceTask.is_continuedAttr(), nceTask.cm_sp_patternAttr(),
                isSegmentedNCETask(parentInputType), outChannelOffsets[clusterId],
                nceTask.input_channels_compressionAttr(), nceTask.is_superdenseAttr(), nceTask.is_inplaceAttr(),
                nceTask.input_se_sizeAttr(), nceTask.output_se_sizeAttr(), nceTask.is_permute_quantizeAttr());

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

        updateProfilingMetadata(nceTask, newTask, clusterId);

        _log.trace("Insert new NCE task: '{0}'", newTask);
    }

    rewriter.eraseOp(vpurtTask);

    return mlir::success();
}

SmallVector<mlir::Value> VPUIP::ClusterNCEBaseRewriter::getWeightsBuffers(mlir::Location loc,
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
    const auto distributionMode = distribution.getMode().getValue();
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
    const auto tilingScheme = parseIntArrayAttr<int64_t>(distribution.getNumTiles());
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
        auto offset = declBuff.getByteOffset();
        const auto newLoc = appendLoc(loc, "_weights_cluster_{0}", clusterId);
        offset += Byte(perClusterShapeOffsets[clusterId][Dim(axis)] * distributedType.getStrides()[Dim(axis)]).count();
        auto newCmxBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
                rewriter, insertionPoint, newLoc, cmxBuffType, VPURT::BufferSection::CMX_NN,
                getIntArrayAttr(_ctx, makeArrayRef({clusterId})), offset, declBuff.getSwizzlingKeyAttr());
        insertionPoint = newCmxBuffer.getOperation();
        perClusterBuffers[clusterId] = newCmxBuffer;
    }
    return perClusterBuffers;
}

//
// ClusterDMARewriter
//

void VPUIP::ClusterDMARewriter::unrollSegmentedOrOverlapped(mlir::Location loc, VPUIP::NCEClusterTilingOp clusterOp,
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
    const auto numClusters = distributionAttr.getNumClusters().getInt();

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

    const auto numTiles = parseIntArrayAttr<int64_t>(distributionAttr.getNumTiles());
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

    const auto tilingScheme = parseIntArrayAttr<int64_t>(distributionAttr.getNumTiles());
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
                                                           declBuff.getByteOffset(), declBuff.getSwizzlingKeyAttr());
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

        Byte offset{declBuff.getByteOffset()};
        offset += static_cast<Byte>(perClusterShapeOffsets[clusterId][Dim(axis)] * newType.getStrides()[Dim(axis)]);

        auto section = declBuff.getSection();
        auto sectionIndex = declBuff.getSectionIndex();

        vpux::IndexedSymbolAttr symbolAttr;
        if (newType.getMemoryKind() == VPU::MemoryKind::CMX_NN) {
            VPUX_THROW_UNLESS(sectionIndex.has_value(), "Cannot get section index for {0}", declBuff);
            auto sectionIndexVal = parseIntArrayAttr<int64_t>(sectionIndex.value());
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

        if (sectionIndex.has_value()) {
            return VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, loc, newType, section,
                                                           sectionIndex.value(), offset.count(),
                                                           declBuff.getSwizzlingKeyAttr());
        }
        return VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, loc, newType, section, nullptr,
                                                       offset.count(), declBuff.getSwizzlingKeyAttr());
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
                rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), newLoc, inputBuffer, outBuffer,
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

mlir::LogicalResult VPUIP::ClusterDMARewriter::matchAndRewrite(VPUIP::NNDMAOp nndmaOp,
                                                               mlir::PatternRewriter& rewriter) const {
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
    const auto mode = distributionAttr.getMode().getValue();
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

        const auto numClusters = distributionAttr.getNumClusters().getInt();
        SmallVector<int64_t> clusters(numClusters);
        std::iota(clusters.begin(), clusters.end(), 0);

        auto cmxBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
                rewriter, outDeclBuff, loc, outDeclBuff.getType(), VPURT::BufferSection::CMX_NN,
                getIntArrayAttr(_ctx, clusters), outDeclBuff.getByteOffset(), outDeclBuff.getSwizzlingKeyAttr());

        _log.trace("Insert new CMX buffer declaration: '{0}'", cmxBuffer);

        const auto newLoc = appendLoc(loc, "_broadcast_copy_to_CMX[{0},{1}]", clusters.front(), clusters.back());
        const auto newNNDMA = VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
                rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), newLoc, input, cmxBuffer,
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
                getIntArrayAttr(_ctx, makeArrayRef({0})), inDeclBuff.getByteOffset(), inDeclBuff.getSwizzlingKeyAttr());

        _log.trace("Insert new CMX buffer declaration: '{0}'", cmxBuffer);

        const auto newNNDMA = VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
                rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), loc, cmxBuffer, output,
                nndmaOp.port(), nndmaOp.channelTypeAttr(), false, false, spillIdAttr, nullptr);
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
