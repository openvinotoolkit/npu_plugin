//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/passes/unroll_cluster_tiling.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/utils/compression_utils.hpp"
#include "vpux/compiler/utils/strings.hpp"

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
    if (nceTask.getProfilingData() == nullptr) {
        return;
    }

    const auto oldMetadata = nceTask.getProfilingMetadataAttr();
    VPUX_THROW_WHEN(oldMetadata == nullptr, "Missed profiling attribute for '{0}'.", nceTask);

    const auto variantsRange = newTask.getVariants().getOps<VPUIP::DPUTaskOp>();
    const auto numVariants = checked_cast<unsigned int>(std::distance(variantsRange.begin(), variantsRange.end()));
    newTask.setProfilingMetadataAttr(
            extendDpuProfAttrWithClusterInfo(oldMetadata, numVariants, checked_cast<unsigned int>(clusterId)));
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

    const auto hasWeightsTable = nceTask.getWeightTable() != nullptr;
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

void VPUIP::ClusterNCEBaseRewriter::matchAndRewrite(VPUIP::NCEClusterTaskOp nceTask, mlir::OpBuilder& builder) const {
    _log.trace("Process NCE op: '{0}'", nceTask);

    auto vpurtTask = nceTask->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_UNLESS(vpurtTask != nullptr, "Can't get VPURT task operation");

    builder.setInsertionPointAfter(vpurtTask);

    VPUX_THROW_UNLESS(!nceTask.getInputs().empty(), "Wrong inputs size: {0}", nceTask.getInputs().size());

    const auto hasOnlyDefaultOutput =
            (nceTask.getOutputs().size() == 1 || nceTask.getOutputs().size() == 2) && !nceTask.getProfilingData();
    const auto hasOutputWithProfiling =
            (nceTask.getOutputs().size() == 2 || nceTask.getOutputs().size() == 3) && nceTask.getProfilingData();

    VPUX_THROW_UNLESS(hasOnlyDefaultOutput || hasOutputWithProfiling, "Wrong outputs size: {0}",
                      nceTask.getOutputs().size());

    auto parentInput = *nceTask.getInputs().begin();
    auto parentOutput = *nceTask.getOutputs().begin();

    auto parentInputType = parentInput.getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto parentOutputType = parentOutput.getType().dyn_cast<VPUIP::DistributedBufferType>();

    auto loc = nceTask->getLoc();
    if (parentInputType == nullptr && parentOutputType == nullptr) {
        // nothing to unroll
        VPUX_THROW_WHEN(stringifyPrimaryLocation(loc).find("/cluster_") != std::string::npos,
                        "/cluster_ suffix should not be present yet but was found in {0}", loc);
        nceTask->setLoc(appendLoc(loc, "cluster_0"));
        return;
    }

    auto inDistribution = parentInputType.getDistribution();
    auto outDistribution = parentOutputType.getDistribution();

    VPUX_THROW_UNLESS(inDistribution.getNumClusters() == outDistribution.getNumClusters(),
                      "Input '{0}' and output '{1}' number of clusters are not equal", inDistribution.getNumClusters(),
                      outDistribution.getNumClusters());

    auto numClusters = inDistribution.getNumClusters().getInt();

    SmallVector<mlir::Value> inputBuffs = {};
    SmallVector<mlir::Value> parentInputBuffs = {};
    SmallVector<mlir::Value> inputSparsityMapBuffs = {};
    SmallVector<mlir::Value> parentInputSparsityMap = {};
    SmallVector<mlir::Value> inputSETableBuffs = {};
    SmallVector<mlir::Value> parentInputSETable = {};

    getInputBuffers(parentInputBuffs, inputBuffs, parentInputSparsityMap, inputSparsityMapBuffs, parentInputSETable,
                    inputSETableBuffs, loc, nceTask, numClusters, builder);

    auto weightsBuffs = getWeightsBuffers(loc, nceTask, numClusters, builder);
    auto weightsSparsityMapBuffs = VPUIP::getPerClusterMemoryBuffers(
            _ctx, loc, "weightsSparsityMap", nceTask.getWeightsSparsityMap(), numClusters, builder);
    auto weightTableBuffs =
            VPUIP::getPerClusterMemoryBuffers(_ctx, loc, "weightTable", nceTask.getWeightTable(), numClusters, builder);
    auto activationWindowBuffs = VPUIP::getPerClusterMemoryBuffers(_ctx, loc, "activationWindow",
                                                                   nceTask.getActivationWindow(), numClusters, builder);
    auto instructionListTableBuffs = VPUIP::getPerClusterMemoryBuffers(
            _ctx, loc, "instructionListTable", nceTask.getInstructionListTable(), numClusters, builder);

    SmallVector<mlir::Value> parentOutputBuffs = {};
    SmallVector<mlir::Value> outputBuffs = {};
    SmallVector<mlir::Value> outputSparsityMapBuffs = {};
    SmallVector<mlir::Value> parentOutputSparsityMap = {};

    getOutputBuffers(parentOutputBuffs, outputBuffs, parentOutputSparsityMap, outputSparsityMapBuffs, loc, nceTask,
                     numClusters, builder);

    auto profilingBuffs = VPUIP::getPerClusterMemoryBuffers(_ctx, loc, "profilingBuff", nceTask.getProfilingData(),
                                                            numClusters, builder);

    const auto outChannelOffsets = getOutChannelOffsets(nceTask, parentInputType, parentOutputType);

    auto padAttr = nceTask.getKernelPaddingAttr();
    SmallVector<VPU::PaddingAttr> padAttrForCluster(numClusters, padAttr);

    // In case of OVERLAPPED mode padding setting in invariant needs to be calculated
    // for each cluster based on distributed type properties
    // However, there might be a case when elementwise operation has OVERLAPPED consumer.
    // In that scenario padding must be calculated only to determine per-cluster shape.
    // Elementwise operations do not support kernel padding.
    const auto isEltwise = (nceTask.getTaskType() == VPUIP::NCETaskType::ELTWISE);
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

        if (nceTask.getOutputSparsityMapBuff()) {
            outputSparsityMap = outputSparsityMapBuffs[clusterId];
            outputSparsityMapType = outputSparsityMap.getType();
        }

        if (nceTask.getProfilingData()) {
            profilingOutputType = profilingBuffs[clusterId].getType();
            profilingData = profilingBuffs[clusterId];
        }

        auto newTask = VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
                builder, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), newLoc, outputType,
                outputSparsityMapType, profilingOutputType, inputBuffs[clusterId], inputSparsityMapBuffs[clusterId],
                inputSETableBuffs[clusterId], weightsBuffs[clusterId], weightsSparsityMapBuffs[clusterId],
                weightTableBuffs[clusterId], instructionListTableBuffs[clusterId], activationWindowBuffs[clusterId],
                parentInputBuffs[clusterId], parentInputSparsityMap[clusterId], parentInputSETable[clusterId],
                parentOutputBuffs[clusterId], parentOutputSparsityMap[clusterId], outputBuffs[clusterId],
                outputSparsityMap, profilingData, nceTask.getTaskType(), nceTask.getKernelSizeAttr(),
                nceTask.getKernelStridesAttr(), padAttrForCluster[clusterId],
                nceTask.getActivationWindowChannelLengthAttr(), nceTask.getIsContinuedAttr(),
                nceTask.getCmSpPatternAttr(), isSegmentedNCETask(parentInputType), outChannelOffsets[clusterId],
                nceTask.getInputChannelsCompressionAttr(), nceTask.getIsSuperdenseAttr(), nceTask.getIsInplaceAttr(),
                nceTask.getInputSeSizeAttr(), nceTask.getOutputSeSizeAttr(), nceTask.getIsPermuteQuantizeAttr());

        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToEnd(&newTask.getVariants().front());

            for (auto variant : nceTask.getVariants().getOps<VPUIP::DPUTaskOp>()) {
                VPUX_THROW_UNLESS(variant.getClusterId().has_value(), "Unable to distribute workload");
                if (variant.getClusterId().value() == clusterId) {
                    builder.clone(*variant);
                }
            }
        }

        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToEnd(&newTask.getPpe().front());

            for (auto& ppe : nceTask.getPpe().getOps()) {
                builder.clone(ppe);
            }
        }

        updateProfilingMetadata(nceTask, newTask, clusterId);

        _log.trace("Insert new NCE task: '{0}'", newTask);
    }

    vpurtTask->dropAllReferences();
    vpurtTask->remove();
}

SmallVector<mlir::Value> VPUIP::ClusterNCEBaseRewriter::getWeightsBuffers(mlir::Location loc,
                                                                          VPUIP::NCEClusterTaskOp nceTask,
                                                                          const int64_t numClusters,
                                                                          mlir::OpBuilder& builder) const {
    auto clusterOperand = nceTask.getWeights();
    if (clusterOperand == nullptr) {
        return SmallVector<mlir::Value>(numClusters, nullptr);
    }

    auto operandType = clusterOperand.getType();
    auto distributedType = operandType.dyn_cast<VPUIP::DistributedBufferType>();
    VPUX_THROW_UNLESS(distributedType != nullptr, "Unsupported operand type {0}", operandType);

    const auto distribution = distributedType.getDistribution();
    const auto distributionMode = distribution.getMode().getValue();
    if (distributionMode != (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED)) {
        return VPUIP::getPerClusterMemoryBuffers(_ctx, loc, "weights", nceTask.getWeights(), numClusters, builder);
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
    const auto innerOperandType = distributedType.getCompactType().cast<vpux::NDTypeInterface>();
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
                builder, insertionPoint, newLoc, cmxBuffType, VPURT::BufferSection::CMX_NN,
                getIntArrayAttr(_ctx, ArrayRef({clusterId})), offset, declBuff.getSwizzlingKeyAttr());
        insertionPoint = newCmxBuffer.getOperation();
        perClusterBuffers[clusterId] = newCmxBuffer;
    }
    return perClusterBuffers;
}

//
// ClusterPerElementDMABaseRewriter
//

void VPUIP::ClusterPerElementDMABaseRewriter::matchAndRewrite(VPUIP::DMATypeOpInterface dmaOp, mlir::OpBuilder& builder,
                                                              bool isDataOverlapped) const {
    if (!isTargetOp(dmaOp)) {
        return;
    }

    _log.trace("Processing DMAOp: {0}", dmaOp);

    auto vpurtTask = dmaOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_UNLESS(vpurtTask != nullptr, "Can't get VPURT task operation");

    const auto inputType = dmaOp.getInput().getType().dyn_cast<NDTypeInterface>();
    const auto outputType = dmaOp.getOutputBuff().getType().dyn_cast<NDTypeInterface>();

    const auto loc = dmaOp->getLoc();

    const auto inputDistType = inputType.dyn_cast<VPUIP::DistributedBufferType>();
    const auto outputDistType = outputType.dyn_cast<VPUIP::DistributedBufferType>();
    if (inputDistType == nullptr && outputDistType == nullptr) {
        // nothing to unroll
        return;
    }

    if (inputDistType != nullptr && outputDistType != nullptr) {
        VPUX_THROW_UNLESS(
                mlir::succeeded(VPU::areDistributionAttrsCompatible(inputDistType, outputDistType,
                                                                    /*allowDifferentPerClusterMemoryView = */ false)),
                "Failed to unroll incompatible cluster distributions: {0} and {1}", inputDistType, outputDistType);
    }

    const auto inputDistMode = inputDistType != nullptr ? inputDistType.getDistribution().getMode().getValue()
                                                        : VPU::DistributionMode::NONE;
    const auto outputDistMode = outputDistType != nullptr ? outputDistType.getDistribution().getMode().getValue()
                                                          : VPU::DistributionMode::NONE;

    const auto unrollingType = getUnrollingType(inputDistMode, outputDistMode);
    VPUX_THROW_WHEN(unrollingType == UnrollingType::FAILED,
                    "Failed to decide unrolling method for DMA op: {0}, with input mode: '{1}' and output mode '{2}'",
                    dmaOp, inputDistMode, outputDistMode);

    builder.setInsertionPointAfter(vpurtTask);

    if (unrollingType == UnrollingType::SEGMENTED) {
        _log.nest().trace("Unrolling with SEGMENDTED or OVERLAPPED mode");
        unrollSegmentedOrOverlapped(loc, vpurtTask, builder, isDataOverlapped);
    } else if (unrollingType == UnrollingType::DUPLICATED) {
        _log.nest().trace("Unrolling with DUPLICATED mode");
        unrollDuplicated(loc, vpurtTask, builder);
    } else {
        VPUX_THROW("Unsupported unrolling mode");
    }

    vpurtTask->dropAllReferences();
    vpurtTask->remove();
}

bool isStorageElementTableConstantOp(Const::DeclareOp constOp) {
    auto elementType = constOp.getType().cast<NDTypeInterface>().getElementType();
    if (!elementType.isInteger(32) || constOp.getResult().use_empty()) {
        return false;
    }

    for (auto constUser : constOp.getResult().getUsers()) {
        auto copyOp = mlir::dyn_cast<VPUIP::NNDMAOp>(constUser);
        if (copyOp == nullptr) {
            return false;
        }

        for (auto copyUser : copyOp.getOutputBuff().getUsers()) {
            if (copyUser == copyOp) {
                continue;
            }

            auto nceTask = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(copyUser);
            if (nceTask == nullptr) {
                return false;
            }

            if (nceTask.getInputStorageElementTable() != copyOp.getOutputBuff()) {
                return false;
            }
        }
    }

    return true;
}

// SE pointers have the following format:
//   31-29 28                            9 8         0
//   -------------------------------------------------
//   | xx |           DATA_PTR            | BASE_PTR |
//   -------------------------------------------------
// There is an example: Bilinear Interpolate H size from 5 to 10
// Input Date:               0         1       2       3       4
// Effective Data:           0 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 4
// - For 30XX and 37XX:
// BASE_PTR at Cluster 0:    0 0 0 0 0 0 0 0 0 0 0 0 0
// BASE_PTR at Cluster 1:                              1 1 1 1 1 1 1 1 1
mlir::Value patchSETableValue(mlir::Location loc, Const::DeclareOp constOp, const int64_t clusterId,
                              mlir::OpBuilder& builder) {
    auto seTableContent = constOp.getContent();
    auto seTableSize = seTableContent.getType().getShape().totalSize();
    auto seTableVals = to_small_vector(seTableContent.getValues<int32_t>());
    VPUX_THROW_UNLESS(seTableVals.size() == checked_cast<size_t>(seTableSize),
                      "Unable to correctly obtain the seTable values");
    const auto baseSEPointer = seTableVals.front();
    for (int64_t index = 0; index < seTableSize; ++index) {
        const int32_t basePtr = seTableVals[index] & 0x1FF;
        if (clusterId != basePtr) {
            seTableVals[index] = seTableVals[index] - baseSEPointer + clusterId;
        }
    }
    const auto denseAttr = mlir::DenseElementsAttr::get(seTableContent.getType().cast<mlir::RankedTensorType>(),
                                                        ArrayRef(seTableVals));
    return builder.create<Const::DeclareOp>(loc, constOp.getType(), Const::ContentAttr::get(denseAttr));
}

void VPUIP::ClusterPerElementDMABaseRewriter::unrollSegmentedOrOverlapped(mlir::Location loc, VPURT::TaskOp vpurtTask,
                                                                          mlir::OpBuilder& builder,
                                                                          bool isDataOverlapped) const {
    auto dmaOp = vpurtTask.getInnerTaskOpOfType<VPUIP::DMATypeOpInterface>();
    VPUX_THROW_WHEN(dmaOp == nullptr, "Inner task is not DMA op");

    const auto input = dmaOp.getInput();
    const auto output = dmaOp.getOutputBuff();

    const auto inputType = input.getType().cast<NDTypeInterface>();
    const auto outputType = output.getType().cast<NDTypeInterface>();
    const auto innerInputType =
            inputType.isa<VPUIP::DistributedBufferType>()
                    ? inputType.cast<VPUIP::DistributedBufferType>().getCompactType().cast<vpux::NDTypeInterface>()
                    : inputType;
    const auto innerOutputType =
            outputType.isa<VPUIP::DistributedBufferType>()
                    ? outputType.cast<VPUIP::DistributedBufferType>().getCompactType().cast<vpux::NDTypeInterface>()
                    : outputType;

    const auto inputDistType = inputType.dyn_cast<VPUIP::DistributedBufferType>();
    const auto outputDistType = outputType.dyn_cast<VPUIP::DistributedBufferType>();

    VPUX_THROW_UNLESS(inputDistType != nullptr || outputDistType != nullptr,
                      "One of operands must have DistributedBuffer type");

    const auto distributionAttr =
            inputDistType != nullptr ? inputDistType.getDistribution() : outputDistType.getDistribution();

    const auto numClusters = distributionAttr.getNumClusters().getInt();
    const auto numTiles = parseIntArrayAttr<int64_t>(distributionAttr.getNumTiles());
    const auto tilingAxis = vpux::VPU::getDistributedTilingAxis(numTiles);

    const auto originInShape = inputType.getShape();
    const auto originOutShape = outputType.getShape();

    VPUX_THROW_UNLESS(originInShape.size() == numTiles.size() && originOutShape.size() == numTiles.size(),
                      "Input shape size '{0}', output shape size '{1}' and tiles array size '{1}' are mismatch",
                      originInShape.size(), originOutShape.size(), numTiles.size());

    const auto perClusterShapes = inputDistType != nullptr ? inputDistType.getPerClusterMemoryShapes()
                                                           : outputDistType.getPerClusterMemoryShapes();

    VPUX_THROW_UNLESS(perClusterShapes.size() == checked_cast<size_t>(numClusters),
                      "Number of shapes '{0}' and clusters '{1}' are mismatch", perClusterShapes.size(), numClusters);

    const auto perClusterShapeOffsets = inputDistType != nullptr ? inputDistType.getPerClusterMemoryShapeOffsets()
                                                                 : outputDistType.getPerClusterMemoryShapeOffsets();

    VPUX_THROW_UNLESS(perClusterShapeOffsets.size() == checked_cast<size_t>(numClusters),
                      "Number of shape offsets '{0}' and clusters '{1}' are mismatch", perClusterShapeOffsets.size(),
                      numClusters);

    // Check if per-cluster DMA input will not be a contiguous block of memory.
    // In such case DMA input buffers should have strides according to parent input tensor
    const auto strideInReqs = StrideReqs::compact(originInShape.size());
    const auto strideOutReqs = StrideReqs::compact(originOutShape.size());

    bool useParentTensorStridesForInput = !strideInReqs.checkStrides(input);
    bool useParentTensorStridesForOutput = !strideOutReqs.checkStrides(output);
    if (useParentTensorStridesForInput) {
        _log.trace("DMA at {0} is not compact for the input, strides = {1}, shape = {2}", loc, inputType.getStrides(),
                   originInShape);
    }
    if (useParentTensorStridesForOutput) {
        _log.trace("DMA at {0} is not compact for the output, strides = {1}, shape = {2}", loc, outputType.getStrides(),
                   originOutShape);
    }

    // If ClusterTiling DMA only has distributedType on one side and the distributedType is not memory contiguous with
    // the tiling, per-cluster DMA will need stride access on the non-distributed side.
    if (inputDistType == nullptr && !isMemoryContiguousWithTiling(outputDistType)) {
        useParentTensorStridesForInput = true;
    }
    if (outputDistType == nullptr && !isMemoryContiguousWithTiling(inputDistType)) {
        useParentTensorStridesForOutput = true;
    }

    // ODU permutations enabled, and tested only for SOH and NCHW order
    // also middle network permutations are disabled for now [Track number: S#67423]
    const bool tileNCHWOutOverH = numTiles.size() == 4 && numTiles[Dims4D::Act::N.ind()] == 1 &&
                                  numTiles[Dims4D::Act::C.ind()] == 1 && numTiles[Dims4D::Act::H.ind()] > 1 &&
                                  numTiles[Dims4D::Act::W.ind()] == 1 && inputType.getDimsOrder() == DimsOrder::NCHW &&
                                  outputType.getDimsOrder() == DimsOrder::NCHW;
    // Reference distributed type
    const auto refDistType = inputDistType != nullptr ? inputDistType : outputDistType;

    // Get spill id attribute if dma is NNDMA op
    auto maybeNNDMAOp = vpurtTask.getInnerTaskOpOfType<VPUIP::NNDMAOp>();
    const auto spillIdAttr = maybeNNDMAOp != nullptr ? maybeNNDMAOp.getSpillIdAttr() : nullptr;
    // Get total alloc size for spill
    auto spillAllocSizeForDistType = refDistType.cast<vpux::NDTypeInterface>().getTotalAllocSize();

    // Get new input and output types
    const auto getNewTypes = [&](NDTypeInterface origType, NDTypeInterface origInnerType, bool useParentStrides) {
        SmallVector<NDTypeInterface> newTypes;
        if (useParentStrides) {
            const auto strides = origType.getStrides();
            for (size_t clusterId = 0; clusterId < perClusterShapes.size(); ++clusterId) {
                const auto newType = changeShapeLeaveStrides(origInnerType, strides, perClusterShapes[clusterId],
                                                             perClusterShapeOffsets[clusterId]);
                newTypes.push_back(newType);
            }
        } else {
            for (size_t clusterId = 0; clusterId < perClusterShapes.size(); ++clusterId) {
                const auto newType =
                        changeShape(origInnerType, perClusterShapes[clusterId], perClusterShapeOffsets[clusterId]);
                newTypes.push_back(newType);
            }
        }
        return newTypes;
    };

    // Get new operand for each cluster
    const auto getNewOperand = [&](int64_t clusterId, mlir::Value operand, VPUIP::DistributedBufferType origDistType,
                                   NDTypeInterface newType, mlir::Operation* insertionPoint) -> mlir::Value {
        // For example, copy of weights in case of SOK
        // <32x16x1x1xfp16, @DDR>  -> <16x16x1x1xfp16, [@CMX, 0]>
        //                         -> <16x16x1x1xfp16, [@CMX, 1]>
        if (auto cst = operand.getDefiningOp<Const::DeclareOp>()) {
            VPUX_THROW_UNLESS(outputType.getMemoryKind() == VPU::MemoryKind::CMX_NN,
                              "Output operand type must have NN_CMX memory space. Got: {0}",
                              outputType.getMemoryKind());

            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointAfter(operand.getDefiningOp());

            auto subviewOp = builder.createOrFold<VPUIP::SubViewOp>(loc, cst, perClusterShapeOffsets[clusterId].raw(),
                                                                    perClusterShapes[clusterId].raw());

            if (isDataOverlapped && isStorageElementTableConstantOp(cst)) {
                auto newCstOp = subviewOp.getDefiningOp<Const::DeclareOp>();
                VPUX_THROW_WHEN(newCstOp == nullptr, "Cannot get the constant operation of SETable");
                return patchSETableValue(loc, newCstOp, clusterId, builder);
            }

            return subviewOp;
        }

        auto declBuff = operand.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(declBuff != nullptr, "Can't get buffer offset for operand: {0}", operand);

        if (origDistType != nullptr) {
            const auto symbolAttr =
                    vpux::IndexedSymbolAttr::get(_ctx, {_cmxNameAttr, vpux::getIntAttr(_ctx, clusterId)});
            newType = vpux::updateSwizzlingSchemeBasedOnDistributedType(origDistType, newType);
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

            return VPURT::createOp<VPURT::DeclareBufferOp>(builder, insertionPoint, loc, newCMXType,
                                                           VPURT::BufferSection::CMX_NN,
                                                           getIntArrayAttr(_ctx, ArrayRef({clusterId})),
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

        Byte buffOffset{declBuff.getByteOffset()};
        auto offset = buffOffset;

        auto isSwizzSpill = vpux::getSwizzlingSchemeAttr(refDistType) != nullptr && spillIdAttr != nullptr;
        auto isCompression = maybeNNDMAOp != nullptr && maybeNNDMAOp.getCompressCandidateAttr() != nullptr;
        if (isSwizzSpill || isCompression) {
            // In case of spilling swizzled buffer each per cluster buffer needs to be copied as is together with
            // additional alignment to DDR. In case of OVERLAPPED mode there cannot be any overlap as this
            // would destroy swizzled data content
            // 0                          Parent Buffer                                         25088
            // |---------------------------------------------------------------------------------|
            // 0              Adjusted Parent Buffer (sizeAlignment numClusters x 512)                          26624
            // |-------------------------------------------------------------------------------------------------|
            //
            //                           Offsets without swizzling alignment
            // 0                6272                  12544                 18816                  25088
            // |------------------|---------------------|---------------------|---------------------|
            //
            //                           Offsets with swizzling alignment
            // 0                 6272 + 384           12544 + (384 + 384)      18816 + (384 + 384 + 384)       26624
            // |----------------------|---------------------|------------------------|---------------------------|
            //
            // Offset for next cluster takes in account all the extra bytes added to per cluster buffer for swizzling
            // Total alloc size already takes this alignment into consideration
            // Same needs to be taken into account in case of compression, where compression buffer size has additional
            // reserved space requirement
            if (isSwizzSpill) {
                newType = vpux::updateSwizzlingSchemeBasedOnDistributedType(refDistType, newType);
            }
            auto perClusterSize = spillAllocSizeForDistType;

            if (isCompression) {
                perClusterSize = Byte(updateSizeForCompression(perClusterSize.count()));
            }

            offset += perClusterSize * clusterId;
        } else {
            offset += static_cast<Byte>(perClusterShapeOffsets[clusterId][Dim(tilingAxis)] *
                                        newType.getStrides()[Dim(tilingAxis)]);
        }

        const auto distType = refDistType.changeElemType(newType.getElementType()).cast<VPUIP::DistributedBufferType>();

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
            const int64_t parentDimH = distType.getShape()[Dims4D::Act::H];
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
            return VPURT::createOp<VPURT::DeclareBufferOp>(builder, insertionPoint, loc, newType, section,
                                                           sectionIndex.value(), offset.count(),
                                                           declBuff.getSwizzlingKeyAttr());
        }
        return VPURT::createOp<VPURT::DeclareBufferOp>(builder, insertionPoint, loc, newType, section, nullptr,
                                                       offset.count(), declBuff.getSwizzlingKeyAttr());
    };

    const auto newInTypes = getNewTypes(inputType, innerInputType, useParentTensorStridesForInput);
    const auto newOutTypes = getNewTypes(outputType, innerOutputType, useParentTensorStridesForOutput);

    auto origDMAOp = vpurtTask.getInnerTaskOpOfType<VPUIP::DMATypeOpInterface>();
    VPUX_THROW_WHEN(origDMAOp == nullptr, "Inner task is not DMA op");
    auto inputInsertionPoint = input.getDefiningOp();
    auto outputInsertionPoint = output.getDefiningOp();
    for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        const auto newInputType = newInTypes[clusterId];
        const auto newOutType = newOutTypes[clusterId];

        const auto inputBuffer = getNewOperand(clusterId, input, inputDistType, newInputType, inputInsertionPoint);
        inputInsertionPoint = inputBuffer.getDefiningOp();
        _log.trace("Insert new input buffer declaration: '{0}'", inputBuffer);

        const auto outBuffer = getNewOperand(clusterId, output, outputDistType, newOutType, outputInsertionPoint);
        outputInsertionPoint = outBuffer.getDefiningOp();
        _log.trace("Insert new output buffer declaration: '{0}'", outBuffer);

        const auto newLoc = appendLoc(loc, "_cluster_{0}", clusterId);
        const auto newDMAOp = wrapIntoTaskOp(origDMAOp, vpurtTask, newLoc, inputBuffer, outBuffer,
                                             clusterId % _dmaPortCount, builder);
        _log.trace("Insert new DMA op: '{0}'", newDMAOp);
    }
}

void VPUIP::ClusterPerElementDMABaseRewriter::unrollDuplicated(mlir::Location loc, VPURT::TaskOp vpurtTask,
                                                               mlir::OpBuilder& builder) const {
    auto dmaOp = vpurtTask.getInnerTaskOpOfType<VPUIP::DMATypeOpInterface>();
    VPUX_THROW_WHEN(dmaOp == nullptr, "Inner task is not DMA op");

    const auto input = dmaOp.getInput();
    const auto output = dmaOp.getOutputBuff();

    const auto inputDistType = input.getType().dyn_cast<VPUIP::DistributedBufferType>();
    const auto outputDistType = output.getType().dyn_cast<VPUIP::DistributedBufferType>();
    VPUX_THROW_UNLESS(inputDistType != nullptr || outputDistType != nullptr,
                      "One of operands must have DistributedBuffer type");

    const auto getInputOperand = [&](mlir::Value input) -> mlir::Value {
        if (!input.getType().isa<VPUIP::DistributedBufferType>()) {
            return input;
        }

        _log.trace("Process DUPLICATED|SEGMENTED input");

        auto inDeclBuff = input.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(inDeclBuff != nullptr, "Can't get input buffer");

        const auto symbolAttr = vpux::IndexedSymbolAttr::get(_ctx, {_cmxNameAttr, vpux::getIntAttr(_ctx, 0)});
        const auto innerInputType =
                input.getType().cast<VPUIP::DistributedBufferType>().getCompactType().cast<vpux::NDTypeInterface>();
        const auto newInType = innerInputType.changeMemSpace(symbolAttr);

        return VPURT::createOp<VPURT::DeclareBufferOp>(
                builder, inDeclBuff, loc, newInType, VPURT::BufferSection::CMX_NN, getIntArrayAttr(_ctx, ArrayRef({0})),
                inDeclBuff.getByteOffset(), inDeclBuff.getSwizzlingKeyAttr());
    };

    const auto getOutputOperand = [&](mlir::Value output) -> mlir::Value {
        const auto outputDistType = output.getType().dyn_cast<VPUIP::DistributedBufferType>();
        if (outputDistType == nullptr) {
            return output;
        }

        auto outDeclBuff = output.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(outDeclBuff != nullptr, "Can't get output buffer");

        const auto numClusters = outputDistType.getDistribution().getNumClusters().getInt();
        SmallVector<int64_t> clusters(numClusters);
        std::iota(clusters.begin(), clusters.end(), 0);

        return VPURT::createOp<VPURT::DeclareBufferOp>(builder, outDeclBuff, loc, outDeclBuff.getType(),
                                                       VPURT::BufferSection::CMX_NN, getIntArrayAttr(_ctx, clusters),
                                                       outDeclBuff.getByteOffset(), outDeclBuff.getSwizzlingKeyAttr());
    };

    builder.setInsertionPointAfter(vpurtTask);

    auto newInputOperand = getInputOperand(input);
    auto newOutputOperand = getOutputOperand(output);

    const auto newDMAOp = wrapIntoTaskOp(dmaOp, vpurtTask, loc, newInputOperand, newOutputOperand,
                                         dmaOp.getPortAttribute().getInt(), builder);

    _log.trace("Insert new DMA op: '{0}'", newDMAOp);
}

bool VPUIP::ClusterDMARewriter::isTargetOp(VPUIP::DMATypeOpInterface dmaOp) const {
    return mlir::isa<VPUIP::NNDMAOp>(dmaOp.getOperation());
}

VPUIP::DMATypeOpInterface VPUIP::ClusterDMARewriter::wrapIntoTaskOp(VPUIP::DMATypeOpInterface dmaOp,
                                                                    VPURT::TaskOp vpurtTask, mlir::Location loc,
                                                                    mlir::Value input, mlir::Value output_buff,
                                                                    int64_t port, mlir::OpBuilder& builder) const {
    auto origNNDMAOp = mlir::dyn_cast<VPUIP::NNDMAOp>(dmaOp.getOperation());
    return VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(builder, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(),
                                                 loc, input, output_buff, port, false, false,
                                                 origNNDMAOp.getSpillIdAttr(), origNNDMAOp.getCompressCandidateAttr());
}

VPUIP::ClusterDMARewriter::UnrollingType VPUIP::ClusterDMARewriter::getUnrollingType(
        VPU::DistributionMode inputMode, VPU::DistributionMode outputMode) const {
    VPUX_THROW_WHEN((inputMode == VPU::DistributionMode::NONE && outputMode == VPU::DistributionMode::NONE) ||
                            (inputMode != VPU::DistributionMode::NONE && outputMode != VPU::DistributionMode::NONE),
                    "One and only one of input/output can be distributed type for cluster NNDMAOp");
    if (inputMode == VPU::DistributionMode::SEGMENTED || inputMode == VPU::DistributionMode::OVERLAPPED ||
        outputMode == VPU::DistributionMode::SEGMENTED || outputMode == VPU::DistributionMode::OVERLAPPED) {
        return UnrollingType::SEGMENTED;
    }
    if (VPU::bitEnumContainsAny(inputMode, VPU::DistributionMode::DUPLICATED) ||
        VPU::bitEnumContainsAny(inputMode, VPU::DistributionMode::MULTICASTED)) {
        return UnrollingType::DUPLICATED;
    }
    if (VPU::bitEnumContainsAny(outputMode, VPU::DistributionMode::DUPLICATED) ||
        outputMode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::MULTICASTED)) {
        return UnrollingType::DUPLICATED;
    }
    return UnrollingType::FAILED;
}
