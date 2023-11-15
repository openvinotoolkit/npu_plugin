//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/utils.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPURT/attributes.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

uint16_t vpux::VPUIP::getProfWorkloadSize(mlir::ModuleOp module) {
    uint16_t profilingWorkloadSize;
    switch (VPU::getArch(module)) {
    case VPU::ArchKind::VPUX30XX:
        profilingWorkloadSize = VPUIP::HW_DPU_PROFILING_SIZE_BYTES_30XX;
        break;
    case VPU::ArchKind::VPUX37XX:
        profilingWorkloadSize = VPUIP::HW_DPU_PROFILING_SIZE_BYTES_37XX;
        break;
    default:
        VPUX_THROW("Not supported architecture");
    }
    VPUX_THROW_WHEN(profilingWorkloadSize % sizeof(uint64_t) != 0, "Not supported size of workload");
    return profilingWorkloadSize;
}

//
// Run-time info
//

double vpux::VPUIP::getMemoryDerateFactor(IE::MemoryResourceOp mem) {
    VPUX_THROW_UNLESS(mem.getKind() != nullptr, "Got empty memory resource kind");
    VPUX_THROW_UNLESS(mem.getKind().isa<VPU::MemoryKindAttr>(), "Unsupported memory resource kind '{0}'",
                      mem.getKind());

    auto attr = mem->getAttr(VPU::getMemoryDerateAttrName());
    VPUX_THROW_UNLESS(attr != nullptr, "Memory resource '{0}' has no '{1}' attribute", mem.getKind(),
                      VPU::getMemoryDerateAttrName());
    VPUX_THROW_UNLESS(attr.isa<mlir::FloatAttr>(), "Memory resource '{0}' has wrong '{1}' attribute : '{2}'",
                      mem.getKind(), VPU::getMemoryDerateAttrName(), attr);

    return attr.cast<mlir::FloatAttr>().getValueAsDouble();
}

uint32_t vpux::VPUIP::getMemoryBandwidth(IE::MemoryResourceOp mem) {
    VPUX_THROW_UNLESS(mem.getKind() != nullptr, "Got empty memory resource kind");
    VPUX_THROW_UNLESS(mem.getKind().isa<VPU::MemoryKindAttr>(), "Unsupported memory resource kind '{0}'",
                      mem.getKind());

    auto attr = mem->getAttr(VPU::getMemoryBandwidthAttrName());
    VPUX_THROW_UNLESS(attr != nullptr, "Memory resource '{0}' has no '{1}' attribute", mem.getKind(),
                      VPU::getMemoryBandwidthAttrName());
    VPUX_THROW_UNLESS(attr.isa<mlir::IntegerAttr>(), "Memory resource '{0}' has wrong '{1}' attribute : '{2}'",
                      mem.getKind(), VPU::getMemoryBandwidthAttrName(), attr);

    return checked_cast<uint32_t>(attr.cast<mlir::IntegerAttr>().getInt());
}

int64_t vpux::VPUIP::getNumClusterUsed(mlir::ModuleOp module) {
    auto nceResOp = IE::getAvailableExecutor(module, VPU::ExecutorKind::NCE);
    VPUX_THROW_UNLESS(nceResOp != nullptr, "Failed to get NCE Executor information");

    return nceResOp.count();
}

int64_t vpux::VPUIP::getNumAvailableBarriers(mlir::Operation* parentOp) {
    // TODO: E#78647 refactor to use api/vpu_cmx_info_{arch}.h
    const EnumMap<VPU::ArchKind, int64_t> MAX_BARRIERS_PER_INFERENCE = {
            {VPU::ArchKind::VPUX30XX, 64 / 2},  // half barries are used (runtime limitation)
            {VPU::ArchKind::VPUX37XX, 64},      //
    };

    const auto arch = VPU::getArch(parentOp);

    auto module = parentOp->getParentOfType<mlir::ModuleOp>();

    const auto numClusters = VPUIP::getNumClusterUsed(module);

    const auto maxNumClustersForArch = VPU::getMaxDPUClusterNum(module);
    VPUX_THROW_UNLESS(maxNumClustersForArch != 0, "Failed to get maxNumClustersForArch");

    const auto barIt = MAX_BARRIERS_PER_INFERENCE.find(arch);
    VPUX_THROW_WHEN(barIt == MAX_BARRIERS_PER_INFERENCE.end(), "Unsupported VPU architecture '{0}'", arch);

    const auto maxBarriersPerInference = barIt->second;

    const auto barriersPerCluster = maxBarriersPerInference / maxNumClustersForArch;
    const auto maxNumBarriers = std::min(maxBarriersPerInference, barriersPerCluster * numClusters);

    return maxNumBarriers;
}

size_t vpux::VPUIP::getBarrierMaxVariantCount(mlir::Operation* parentOp) {
    // TODO: E#78647 refactor to use api/vpu_cmx_info_{arch}.h
    const EnumMap<VPU::ArchKind, size_t> maxVariantCount = {
            {VPU::ArchKind::VPUX30XX, 512},
            {VPU::ArchKind::VPUX37XX, 256},
    };

    const auto arch = VPU::getArch(parentOp);

    const auto barIt = maxVariantCount.find(arch);
    VPUX_THROW_WHEN(barIt == maxVariantCount.end(), "Unsupported VPU architecture '{0}'", arch);

    return barIt->second;
}

int64_t vpux::VPUIP::getNumberOfIndependentDmaQueues(mlir::Operation* parentOp) {
    auto module = parentOp->getParentOfType<mlir::ModuleOp>();
    auto dmaPorts = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    VPUX_THROW_UNLESS(dmaPorts != nullptr, "Failed to get DMA information");
    auto dmaCount = dmaPorts.count();

    return dmaCount;
}

//
// DW Convolution utility
//

namespace {

mlir::Value getAlignedConstWeights(mlir::OpBuilder& builder, mlir::Location loc, Const::DeclareOp weightsConst,
                                   Shape flatWeightShape, int64_t padding) {
    auto weightsContentAttr = weightsConst.getContentAttr();
    auto nchwWeightsContentAttr = weightsContentAttr.reorder(DimsOrder::NCHW);

    auto flatWeightsContentAttr = nchwWeightsContentAttr.reshape(flatWeightShape);
    auto alignedWeightsContentAttr = flatWeightsContentAttr.padWithZero({0, 0, 0, 0}, {0, padding, 0, 0});
    auto nhwcWeightsContentAttr = alignedWeightsContentAttr.reorder(DimsOrder::NHWC);

    const auto OC = flatWeightShape[Dims4D::Filter::OC];
    const auto flatWeightChannelsCount = flatWeightShape[Dims4D::Filter::IC];
    const auto alignedWeightShape = SmallVector<int64_t>{OC, flatWeightChannelsCount + padding, 1, 1};
    const auto origFilterType = weightsConst.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto outAllocType =
            mlir::MemRefType::get(alignedWeightShape, origFilterType.getElementType()).cast<vpux::NDTypeInterface>();
    const auto outAllocTypeNHWC = outAllocType.changeDimsOrder(DimsOrder::NHWC);
    auto alignedWeightsOp = builder.create<Const::DeclareOp>(loc, outAllocTypeNHWC, nhwcWeightsContentAttr);

    return alignedWeightsOp.getOutput();
}

mlir::Value getAlignedNonConstWeights(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value origFilter,
                                      Shape flatWeightShape, int64_t padding) {
    auto ctx = builder.getContext();
    // Step 1: Flatten input to OCxICx1x1, where IC = filters * KY * KX.
    const auto origFilterType = origFilter.getType().cast<vpux::NDTypeInterface>();
    const auto flatWeightType =
            origFilterType.changeShape(flatWeightShape).changeDimsOrder(DimsOrder::fromValue(origFilter));
    auto flatWeightsOp = builder.create<VPUIP::GenericReshapeOp>(loc, flatWeightType, origFilter);

    // Step 2: Permute flat input to NCHW.
    auto flatWeightTypeNCHWType = flatWeightType.changeDimsOrder(DimsOrder::NCHW);
    const auto nchwAttr = mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(ctx));
    const auto flatWeightsDimsAttr =
            mlir::AffineMapAttr::get(DimsOrder::fromValue(flatWeightsOp.output()).toAffineMap(ctx));
    auto flatWeightsNCHW = builder.create<VPUIP::PermuteCastOp>(loc, flatWeightTypeNCHWType, flatWeightsOp.output(),
                                                                nchwAttr, flatWeightsDimsAttr);

    // Step 3: Create padding for flat NCHW input. IC must be a multiple of 16.
    const auto OC = flatWeightShape[Dims4D::Filter::OC];
    const auto flatWeightChannelsCount = flatWeightShape[Dims4D::Filter::IC];
    const auto alignedWeightShape = SmallVector<int64_t>{OC, flatWeightChannelsCount + padding, 1, 1};
    const auto outShapedType =
            mlir::MemRefType::get(alignedWeightShape, origFilterType.getElementType()).cast<vpux::NDTypeInterface>();
    const auto outAllocType = outShapedType.changeDimsOrder(DimsOrder::NCHW);

    const auto padShape = SmallVector<int64_t>{OC, padding, 1, 1};
    const auto padValues = std::vector<ngraph::float16>(OC * padding, 0.f);
    const auto padShapedType =
            mlir::RankedTensorType::get(padShape, origFilterType.getElementType()).cast<vpux::NDTypeInterface>();
    const auto padType = padShapedType.changeDimsOrder(DimsOrder::NCHW);
    const auto padAttr = mlir::DenseElementsAttr::get(padType.cast<mlir::RankedTensorType>(), makeArrayRef(padValues));
    const auto padContentAttr = Const::ContentAttr::get(padAttr);

    const auto padAllocType =
            mlir::MemRefType::get(padShape, origFilterType.getElementType()).cast<vpux::NDTypeInterface>();
    const auto padAllocTypeNHWC = padAllocType.changeDimsOrder(DimsOrder::NCHW);
    auto paddedTensor = builder.create<Const::DeclareOp>(loc, padAllocTypeNHWC, padContentAttr);

    // Step 4: Concatenate flat NCHW input with padding.
    auto subViewAlloc = builder.create<mlir::memref::AllocOp>(loc, outAllocType.cast<mlir::MemRefType>());

    const SmallVector<int64_t> filterOffsets = {0, 0, 0, 0};
    const auto filterOffsetsAttr = getIntArrayAttr(ctx, filterOffsets);
    const auto flatWeightShapeAttr = getIntArrayAttr(ctx, flatWeightShape);

    const SmallVector<int64_t> paddingOffsets = {0, flatWeightChannelsCount, 0, 0};
    const auto paddingOffsetsAttr = getIntArrayAttr(ctx, paddingOffsets);
    const auto padShapeAttr = getIntArrayAttr(ctx, padShape);

    auto subViewFilter = builder.create<VPUIP::SubViewOp>(loc, subViewAlloc, filterOffsetsAttr, flatWeightShapeAttr);
    auto subViewPadding = builder.create<VPUIP::SubViewOp>(loc, subViewAlloc, paddingOffsetsAttr, padShapeAttr);

    auto subViewFilterCopy = builder.create<VPUIP::CopyOp>(loc, flatWeightsNCHW.result(), subViewFilter);
    auto subViewPaddingCopy = builder.create<VPUIP::CopyOp>(loc, paddedTensor.getOutput(), subViewPadding);

    auto concatViewOp = builder.create<VPUIP::ConcatViewOp>(
            loc, SmallVector<mlir::Value>{subViewFilterCopy.output(), subViewPaddingCopy.output()}, subViewAlloc);

    // Step 5: Permute the result to NHWC.
    auto outNHWCType = outAllocType.changeDimsOrder(DimsOrder::NHWC);
    const auto nhwcAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(ctx));

    auto outOpNCHW = builder.create<VPUIP::PermuteCastOp>(loc, outNHWCType, concatViewOp.output(), nhwcAttr, nchwAttr);

    return outOpNCHW.result();
}

}  // namespace

mlir::Value vpux::VPUIP::alignDepthWiseWeightsTensor(mlir::OpBuilder& builder, mlir::Location loc,
                                                     mlir::Value origFilter) {
    const auto filterShape = getShape(origFilter);
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto filtersPerInChan = filterShape[Dims4D::Filter::IC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto origFilterType = origFilter.getType().cast<vpux::NDTypeInterface>();
    const auto alignment = VPU::NCEInvariant::getAlignment(origFilterType.getElementType());

    const auto remainder = (filtersPerInChan * KY * KX) % alignment;
    VPUX_THROW_UNLESS(remainder >= 0, "Channel alignment cannot be negative: {0}", remainder);

    if (remainder == 0) {
        return origFilter;
    }

    const auto padding = alignment - remainder;

    const auto flatWeightChannelsCount = filtersPerInChan * KY * KX;
    const auto flatWeightShape = Shape{OC, flatWeightChannelsCount, 1, 1};

    if (auto weightsConst = origFilter.getDefiningOp<Const::DeclareOp>()) {
        return getAlignedConstWeights(builder, loc, weightsConst, flatWeightShape, padding);
    } else {
        return getAlignedNonConstWeights(builder, loc, origFilter, flatWeightShape, padding);
    }
}

// In case operation is wrapped in NCEClusterTiling this method will return mlir::Value at parent level
// corresponding to mlir::Value used by wrapped operation
// In case operation is not wrapped in NCEClusterTiling then just return same mlir::Value
mlir::Value vpux::VPUIP::getTopBufferOfNCEClusterTiling(mlir::Operation* innerOp, mlir::Value buffer) {
    if (buffer == nullptr) {
        return buffer;
    }

    if (auto nceClustOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(innerOp->getParentOp())) {
        auto* bodyBlock = &nceClustOp.body().front();
        const auto blockArg = buffer.dyn_cast<mlir::BlockArgument>();
        VPUX_THROW_WHEN(blockArg == nullptr || blockArg.getOwner() != bodyBlock,
                        "Matching argument was not identified");

        return nceClustOp->getOperand(blockArg.getArgNumber());
    }
    return buffer;
}

void vpux::VPUIP::moveRootAllocBefore(mlir::Operation* root, mlir::Operation* targetOp) {
    root->moveBefore(targetOp);
    if (mlir::isa<VPUIP::GroupSparseBufferOp>(root)) {
        for (auto operand : root->getOperands()) {
            operand.getDefiningOp()->moveBefore(root);
        }
    }
}

mlir::Type vpux::VPUIP::extractDataType(mlir::Value val) {
    return extractDataType(val.getType());
}

mlir::Type vpux::VPUIP::extractDataType(mlir::Type type) {
    if (auto sparseType = type.dyn_cast<VPUIP::SparseBufferType>()) {
        return sparseType.getData();
    }
    return type;
}

//
// Unrolling Utilities
//

namespace {

bool isDiscontinuousBufferType(vpux::NDTypeInterface bufferType) {
    const auto strideReqs = StrideReqs::compact(bufferType.getShape().size());
    return !strideReqs.checkStrides(bufferType);
}

vpux::NDTypeInterface changeShape(vpux::NDTypeInterface originType, ShapeRef shape, ShapeRef offset) {
    return originType.extractDenseTile(offset, shape);
}

vpux::NDTypeInterface changeShapeLeaveStrides(vpux::NDTypeInterface originType, StridesRef strides, ShapeRef shape,
                                              ShapeRef offset) {
    VPUX_THROW_UNLESS((originType.isa<mlir::MemRefType>()),
                      "Only MemRefType is supported for 'changeShapeLeaveStrides'. Got '{0}'", originType);
    return originType.extractDenseTile(offset, shape).changeStrides(strides);
}

mlir::Type getElementType(VPUIP::DistributedBufferType distributedType, ShapeRef perClusterShape,
                          ShapeRef perClusterShapeOffset) {
    const auto elemType = distributedType.getElementType();
    if (const auto qType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        return tileScalesAndZP(qType, perClusterShape, perClusterShapeOffset);
    }
    return elemType;
}

// Get per-cluster buffers for distributed type
SmallVector<mlir::Value> getPerClusterBuffers(mlir::MLIRContext* ctx, mlir::Location loc, StringRef bufferName,
                                              mlir::Value clusterOperand, mlir::Type compactType,
                                              ArrayRef<Shape> perClusterShapes, ArrayRef<Shape> perClusterShapeOffsets,
                                              int64_t numClusters, mlir::PatternRewriter& rewriter,
                                              bool allowDiscontinuousBuffers) {
    const auto cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));

    auto compactTypeND = compactType.cast<vpux::NDTypeInterface>();

    auto operandType = clusterOperand.getType();
    auto distributedType = operandType.dyn_cast<VPUIP::DistributedBufferType>();
    VPUX_THROW_UNLESS(distributedType != nullptr, "Unsupported operand type {0}", operandType);

    const auto distribution = distributedType.getDistribution();
    const auto distributionMode = distribution.getMode().getValue();

    auto declBuff = clusterOperand.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(declBuff != nullptr, "Can't get buffer offset for operand: {0}", clusterOperand);

    SmallVector<mlir::Value> perClusterBuffers(numClusters);
    if (distributionMode == VPU::DistributionMode::SEGMENTED || distributionMode == VPU::DistributionMode::DUPLICATED ||
        distributionMode == VPU::DistributionMode::OVERLAPPED) {
        auto insertionPoint = declBuff.getOperation();
        for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
            auto cmxBuffType =
                    (allowDiscontinuousBuffers && isDiscontinuousBufferType(compactTypeND))
                            ? changeShapeLeaveStrides(compactTypeND, compactTypeND.getStrides(),
                                                      perClusterShapes[clusterId], perClusterShapeOffsets[clusterId])
                            : changeShape(compactTypeND, perClusterShapes[clusterId],
                                          perClusterShapeOffsets[clusterId]);

            const auto symbolAttr = vpux::IndexedSymbolAttr::get(ctx, {cmxNameAttr, vpux::getIntAttr(ctx, clusterId)});
            cmxBuffType = cmxBuffType.changeMemSpace(symbolAttr);

            const auto newLoc = appendLoc(loc, "_{0}_cluster_{1}", bufferName, clusterId);

            auto newCmxBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
                    rewriter, insertionPoint, newLoc, cmxBuffType, VPURT::BufferSection::CMX_NN,
                    getIntArrayAttr(ctx, makeArrayRef({clusterId})), declBuff.getByteOffset(),
                    declBuff.getSwizzlingKeyAttr());

            insertionPoint = newCmxBuffer.getOperation();

            perClusterBuffers[clusterId] = newCmxBuffer;
        }

        return perClusterBuffers;
    }

    //       Task1(SOK)
    // CMX0 |-out part1-|-out part2-|
    // CMX1 |-out part1-|-out part2-|
    //                    Task2(SOK)
    if (distributionMode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED)) {
        SmallVector<int64_t> clusters(numClusters);
        std::iota(clusters.begin(), clusters.end(), 0);

        const auto elemSize = distributedType.getElemTypeSize();
        const auto elemStrides = to_small_vector(distributedType.getStrides() | transformed([&](Bit stride) {
                                                     return stride.count() / elemSize.count();
                                                 }));
        const auto order = distributedType.getDimsOrder();
        const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(ctx));
        const auto stridesAttr = getIntArrayAttr(ctx, elemStrides);
        auto layout = VPUIP::MemRefAttr::get(orderAttr, stridesAttr, /*swizzlingScheme=*/nullptr,
                                             distributedType.getCompressionScheme(), /*allocSize=*/nullptr, ctx);

        auto insertionPoint = declBuff.getOperation();
        for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
            const auto elemType =
                    getElementType(distributedType, perClusterShapes[clusterId], perClusterShapeOffsets[clusterId]);
            const auto newDistributedType =
                    VPUIP::DistributedBufferType::get(ctx, perClusterShapes[clusterId].raw(), elemType, layout,
                                                      distributedType.getMemSpace(), distributedType.getDistribution());

            const auto newLoc = appendLoc(loc, "_{0}_cluster_{1}", bufferName, clusterId);

            auto newCmxBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
                    rewriter, insertionPoint, newLoc, newDistributedType, VPURT::BufferSection::CMX_NN,
                    getIntArrayAttr(ctx, clusters), declBuff.getByteOffset(), declBuff.getSwizzlingKeyAttr());

            insertionPoint = newCmxBuffer.getOperation();

            perClusterBuffers[clusterId] = newCmxBuffer;
        }

        return perClusterBuffers;
    }

    //      Task1(HKSwitch)
    // CMX0 |-out part1-|-out part2-|
    // CMX1 |-out part1-|-out part2-|
    //                  Task2(HKSwitch)
    if (distributionMode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::MULTICASTED)) {
        SmallVector<int64_t> clusters(numClusters);
        std::iota(clusters.begin(), clusters.end(), 0);

        auto insertionPoint = declBuff.getOperation();
        for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
            const auto elemType =
                    getElementType(distributedType, perClusterShapes[clusterId], perClusterShapeOffsets[clusterId]);
            const auto newDistributedType = VPUIP::DistributedBufferType::get(
                    ctx, perClusterShapes[clusterId].raw(), elemType, distributedType.getLayout(),
                    distributedType.getMemSpace(), distributedType.getDistribution());

            // It's a specific workaround for HK switch strategy. HK switch computes output offsets both by variants
            // start/end_x/y/z AND ODU base address. So we need to provide different ODU base address for each cluster.
            // There's a ticket E#29671 describing the work to remove such special handling for HK switch.
            // This workaround can be removed after it's done.
            const auto strides = distributedType.getStrides();
            Byte cmxOffset{declBuff.getByteOffset()};
            for (size_t axis = 0; axis < strides.size(); axis++) {
                cmxOffset += static_cast<Byte>(perClusterShapeOffsets[clusterId][Dim(axis)] * strides[Dim(axis)]);
            }

            const auto newLoc = appendLoc(loc, "_{0}_cluster_{1}", bufferName, clusterId);

            auto newCmxBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
                    rewriter, insertionPoint, newLoc, newDistributedType, VPURT::BufferSection::CMX_NN,
                    getIntArrayAttr(ctx, clusters), cmxOffset.count(), declBuff.getSwizzlingKeyAttr());

            insertionPoint = newCmxBuffer.getOperation();

            perClusterBuffers[clusterId] = newCmxBuffer;
        }

        return perClusterBuffers;
    }

    VPUX_THROW("Unsupported distribution mode: {0}", VPU::stringifyDistributionMode(distributionMode));
}

SmallVector<mlir::Value> getPerClusterSWBuffers(mlir::MLIRContext* ctx, mlir::Location loc, StringRef bufferName,
                                                VPUIP::NCEClusterTilingOp clusterOp, mlir::Value innerOperand,
                                                ArrayRef<Shape> perClusterShapes,
                                                ArrayRef<Shape> perClusterShapeOffsets, int64_t numClusters,
                                                mlir::PatternRewriter& rewriter, Logger log,
                                                bool allowDiscontinuousBuffers) {
    const auto cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));

    auto clusterOperand = VPUIP::getClusterOperand(clusterOp, innerOperand);
    if (clusterOperand == nullptr) {
        return SmallVector<mlir::Value>(numClusters, nullptr);
    }
    auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(clusterOp.getInnerTaskOp());
    VPUX_THROW_WHEN(swKernelOp == nullptr, "cannot get inner sw kernel op");

    auto innerOperandType = innerOperand.getType().cast<vpux::NDTypeInterface>();

    auto operandType = clusterOperand.getType();
    auto distributedType = operandType.dyn_cast<VPUIP::DistributedBufferType>();
    VPUX_THROW_UNLESS(distributedType != nullptr, "Unsupported operand type {0}", operandType);

    const auto distribution = distributedType.getDistribution();
    const auto distributionMode = distribution.getMode().getValue();

    auto declBuff = clusterOperand.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(declBuff != nullptr, "Can't get buffer offset for operand: {0}", clusterOperand);

    SmallVector<mlir::Value> perClusterBuffers(numClusters);
    if (distributionMode == VPU::DistributionMode::SEGMENTED || distributionMode == VPU::DistributionMode::DUPLICATED ||
        distributionMode == VPU::DistributionMode::OVERLAPPED) {
        auto insertionPoint = declBuff.getOperation();
        for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
            auto cmxBuffType =
                    changeShape(innerOperandType, perClusterShapes[clusterId], perClusterShapeOffsets[clusterId]);
            if (allowDiscontinuousBuffers && isDiscontinuousBufferType(innerOperandType)) {
                auto newStrides = innerOperandType.getStrides();
                if (swKernelOp.stridesAttr() != nullptr) {
                    newStrides.clear();
                    auto perClusterStrides = parseIntArrayOfArrayAttr<int64_t>(swKernelOp.stridesAttr());
                    Bit elemSize = distributedType.getElemTypeSize();
                    for (auto val : perClusterStrides[clusterId]) {
                        newStrides.push_back(Bit(val * elemSize.count()));
                    }
                }

                cmxBuffType = changeShapeLeaveStrides(innerOperandType, vpux::StridesRef(newStrides),
                                                      perClusterShapes[clusterId], perClusterShapeOffsets[clusterId]);
            }
            const auto symbolAttr = vpux::IndexedSymbolAttr::get(ctx, {cmxNameAttr, vpux::getIntAttr(ctx, clusterId)});
            cmxBuffType = cmxBuffType.changeMemSpace(symbolAttr);

            const auto newLoc = appendLoc(loc, "_{0}_cluster_{1}", bufferName, clusterId);
            auto newCmxBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
                    rewriter, insertionPoint, newLoc, cmxBuffType, VPURT::BufferSection::CMX_NN,
                    getIntArrayAttr(ctx, makeArrayRef({clusterId})), declBuff.getByteOffset(),
                    declBuff.getSwizzlingKeyAttr());

            insertionPoint = newCmxBuffer.getOperation();
            log.trace("Insert new CMX buffer: '{0}'", newCmxBuffer);

            perClusterBuffers[clusterId] = newCmxBuffer;
        }

        return perClusterBuffers;
    }

    //       Task1(SOK)
    // CMX0 |-out part1-|-out part2-|
    // CMX1 |-out part1-|-out part2-|
    //                    Task2(SOK)
    if (distributionMode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED)) {
        SmallVector<int64_t> clusters(numClusters);
        std::iota(clusters.begin(), clusters.end(), 0);

        const auto elemSize = distributedType.getElemTypeSize();
        const auto elemStrides = to_small_vector(distributedType.getStrides() | transformed([&](Bit stride) {
                                                     return stride.count() / elemSize.count();
                                                 }));
        const auto order = distributedType.getDimsOrder();
        const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(ctx));
        const auto stridesAttr = getIntArrayAttr(ctx, elemStrides);
        auto layout = VPUIP::MemRefAttr::get(orderAttr, stridesAttr, nullptr, distributedType.getCompressionScheme(),
                                             /*allocSize=*/nullptr, ctx);
        auto insertionPoint = declBuff.getOperation();
        for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
            const auto elemType =
                    getElementType(distributedType, perClusterShapes[clusterId], perClusterShapeOffsets[clusterId]);
            const auto newDistributedType =
                    VPUIP::DistributedBufferType::get(ctx, perClusterShapes[clusterId].raw(), elemType, layout,
                                                      distributedType.getMemSpace(), distributedType.getDistribution());

            const auto newLoc = appendLoc(loc, "_{0}_cluster_{1}", bufferName, clusterId);

            auto newCmxBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
                    rewriter, insertionPoint, newLoc, newDistributedType, VPURT::BufferSection::CMX_NN,
                    getIntArrayAttr(ctx, clusters), declBuff.getByteOffset(), declBuff.getSwizzlingKeyAttr());

            log.trace("Insert new CMX buffer: '{0}'", newCmxBuffer);
            insertionPoint = newCmxBuffer.getOperation();

            perClusterBuffers[clusterId] = newCmxBuffer;
        }

        return perClusterBuffers;
    }

    //      Task1(HKSwitch)
    // CMX0 |-out part1-|-out part2-|
    // CMX1 |-out part1-|-out part2-|
    //                  Task2(HKSwitch)
    if (distributionMode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::MULTICASTED)) {
        SmallVector<int64_t> clusters(numClusters);
        std::iota(clusters.begin(), clusters.end(), 0);

        auto insertionPoint = declBuff.getOperation();
        for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
            const auto elemType =
                    getElementType(distributedType, perClusterShapes[clusterId], perClusterShapeOffsets[clusterId]);
            const auto newDistributedType = VPUIP::DistributedBufferType::get(
                    ctx, perClusterShapes[clusterId].raw(), elemType, distributedType.getLayout(),
                    distributedType.getMemSpace(), distributedType.getDistribution());

            const auto strides = distributedType.getStrides();
            Byte cmxOffset{declBuff.getByteOffset()};
            for (size_t axis = 0; axis < strides.size(); axis++) {
                cmxOffset += perClusterShapeOffsets[clusterId][Dim(axis)] * static_cast<Byte>(strides[Dim(axis)]);
            }

            const auto newLoc = appendLoc(loc, "_{0}_cluster_{1}", bufferName, clusterId);

            auto newCmxBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
                    rewriter, insertionPoint, newLoc, newDistributedType, VPURT::BufferSection::CMX_NN,
                    getIntArrayAttr(ctx, clusters), cmxOffset.count(), declBuff.getSwizzlingKeyAttr());

            insertionPoint = newCmxBuffer.getOperation();
            log.trace("Insert new CMX buffer: '{0}'", newCmxBuffer);

            perClusterBuffers[clusterId] = newCmxBuffer;
        }

        return perClusterBuffers;
    }

    VPUX_THROW("Unsupported distribution mode: {0}", VPU::stringifyDistributionMode(distributionMode));
}

}  // namespace

mlir::Value vpux::VPUIP::getClusterOperand(VPUIP::NCEClusterTilingOp clusterOp, mlir::Value innerOperand) {
    if (innerOperand == nullptr) {
        return nullptr;
    }

    const auto blockArg = innerOperand.dyn_cast<mlir::BlockArgument>();
    VPUX_THROW_WHEN(blockArg == nullptr, "Inner operand '{0}' is not a block argument", innerOperand);
    VPUX_THROW_WHEN(blockArg.getOwner() != clusterOp.getInnerTaskOp()->getBlock(),
                    "Cannot match the origin operand with the inner one '{0}'", innerOperand);

    return clusterOp->getOperand(blockArg.getArgNumber());
}

// TODO: Replace this version of getPerClusterMemoryBuffers with the one that doesn't use clusterOp / innerOp
// distinction and works purely on DistributedBufferType.
SmallVector<mlir::Value> vpux::VPUIP::getPerClusterMemoryBuffers(mlir::MLIRContext* ctx, mlir::Location loc,
                                                                 StringRef bufferName, mlir::Value clusterOperand,
                                                                 mlir::Value innerOperand, int64_t numClusters,
                                                                 mlir::PatternRewriter& rewriter,
                                                                 bool allowDiscontinuousBuffers) {
    if (clusterOperand == nullptr) {
        return SmallVector<mlir::Value>(numClusters, nullptr);
    }

    auto operandType = clusterOperand.getType();
    auto distributedType = operandType.dyn_cast<VPUIP::DistributedBufferType>();
    VPUX_THROW_UNLESS(distributedType != nullptr, "Unsupported operand type {0}", operandType);

    auto perClusterShapes = distributedType.getPerClusterMemoryShapes();
    VPUX_THROW_UNLESS(perClusterShapes.size() == checked_cast<size_t>(numClusters),
                      "Mismatch in shapes '{0}' and clusters '{1}'", perClusterShapes.size(), numClusters);
    const auto perClusterShapeOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    VPUX_THROW_UNLESS(perClusterShapeOffsets.size() == checked_cast<size_t>(numClusters),
                      "Number of shape offsets '{0}' and clusters '{1}'", perClusterShapeOffsets.size(), numClusters);

    return getPerClusterBuffers(ctx, loc, bufferName, clusterOperand, innerOperand.getType(), perClusterShapes,
                                perClusterShapeOffsets, numClusters, rewriter, allowDiscontinuousBuffers);
}

// NOTE: This is a temporary version of getPerClusterMemoryBuffers that works purely on DistributedBufferType.
// It's going to replace original getPerClusterMemoryBuffers eventually.
SmallVector<mlir::Value> vpux::VPUIP::getPerClusterMemoryBuffers(mlir::MLIRContext* ctx, mlir::Location loc,
                                                                 StringRef bufferName, mlir::Value operand,
                                                                 VPUIP::DistributedBufferType distributedType,
                                                                 int64_t numClusters, mlir::PatternRewriter& rewriter,
                                                                 bool allowDiscontinuousBuffers) {
    if (distributedType == nullptr) {
        return SmallVector<mlir::Value>(numClusters, nullptr);
    }

    auto perClusterShapes = distributedType.getPerClusterMemoryShapes();
    VPUX_THROW_UNLESS(perClusterShapes.size() == checked_cast<size_t>(numClusters),
                      "Mismatch in shapes '{0}' and clusters '{1}'", perClusterShapes.size(), numClusters);
    const auto perClusterShapeOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    VPUX_THROW_UNLESS(perClusterShapeOffsets.size() == checked_cast<size_t>(numClusters),
                      "Mismatch in shape offsets '{0}' and clusters '{1}'", perClusterShapeOffsets.size(), numClusters);

    return getPerClusterBuffers(ctx, loc, bufferName, operand, distributedType.getCompactType(), perClusterShapes,
                                perClusterShapeOffsets, numClusters, rewriter, allowDiscontinuousBuffers);
}

SmallVector<mlir::Value> vpux::VPUIP::getPerClusterComputeBuffers(mlir::MLIRContext* ctx, mlir::Location loc,
                                                                  StringRef bufferName, mlir::Value clusterOperand,
                                                                  mlir::Value innerOperand, int64_t numClusters,
                                                                  mlir::PatternRewriter& rewriter,
                                                                  bool allowDiscontinuousBuffers) {
    if (clusterOperand == nullptr) {
        return SmallVector<mlir::Value>(numClusters, nullptr);
    }

    auto operandType = clusterOperand.getType();
    auto distributedType = operandType.dyn_cast<VPUIP::DistributedBufferType>();
    VPUX_THROW_UNLESS(distributedType != nullptr, "Unsupported operand type {0}", operandType);

    auto perClusterShapes = distributedType.getPerClusterComputeShapes();
    VPUX_THROW_UNLESS(perClusterShapes.size() == checked_cast<size_t>(numClusters),
                      "Mismatch in shapes '{0}' and clusters '{1}'", perClusterShapes.size(), numClusters);
    const auto perClusterShapeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    VPUX_THROW_UNLESS(perClusterShapeOffsets.size() == checked_cast<size_t>(numClusters),
                      "Mismatch in shape offsets '{0}' and clusters '{1}'", perClusterShapeOffsets.size(), numClusters);

    return getPerClusterBuffers(ctx, loc, bufferName, clusterOperand, innerOperand.getType(), perClusterShapes,
                                perClusterShapeOffsets, numClusters, rewriter, allowDiscontinuousBuffers);
}

SmallVector<mlir::Value> vpux::VPUIP::getPerClusterSWMemoryBuffers(mlir::MLIRContext* ctx, mlir::Location loc,
                                                                   StringRef bufferName,
                                                                   VPUIP::NCEClusterTilingOp clusterOp,
                                                                   mlir::Value innerOperand, int64_t numClusters,
                                                                   mlir::PatternRewriter& rewriter, Logger log,
                                                                   bool allowDiscontinuousBuffers) {
    auto clusterOperand = VPU::getDistributedOperandFromNCEClusterTiling(clusterOp, innerOperand);
    if (clusterOperand == nullptr) {
        return SmallVector<mlir::Value>(numClusters, nullptr);
    }

    auto operandType = clusterOperand.getType();
    auto distributedType = operandType.dyn_cast<VPUIP::DistributedBufferType>();
    VPUX_THROW_UNLESS(distributedType != nullptr, "Unsupported operand type {0}", operandType);
    auto perClusterShapes = distributedType.getPerClusterMemoryShapes();
    VPUX_THROW_UNLESS(perClusterShapes.size() == checked_cast<size_t>(numClusters),
                      "Mismatch in shapes '{0}' and clusters '{1}'", perClusterShapes.size(), numClusters);
    const auto perClusterShapeOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    VPUX_THROW_UNLESS(perClusterShapeOffsets.size() == checked_cast<size_t>(numClusters),
                      "Mismatch in shape offsets '{0}' and clusters '{1}'", perClusterShapeOffsets.size(), numClusters);

    return getPerClusterSWBuffers(ctx, loc, bufferName, clusterOp, innerOperand, perClusterShapes,
                                  perClusterShapeOffsets, numClusters, rewriter, log, allowDiscontinuousBuffers);
}

SmallVector<mlir::Value> vpux::VPUIP::getPerClusterSWComputeBuffers(mlir::MLIRContext* ctx, mlir::Location loc,
                                                                    StringRef bufferName,
                                                                    VPUIP::NCEClusterTilingOp clusterOp,
                                                                    mlir::Value innerOperand, int64_t numClusters,
                                                                    mlir::PatternRewriter& rewriter, Logger log,
                                                                    bool allowDiscontinuousBuffers) {
    auto clusterOperand = VPU::getDistributedOperandFromNCEClusterTiling(clusterOp, innerOperand);
    if (clusterOperand == nullptr) {
        return SmallVector<mlir::Value>(numClusters, nullptr);
    }

    auto operandType = clusterOperand.getType();
    auto distributedType = operandType.dyn_cast<VPUIP::DistributedBufferType>();
    VPUX_THROW_UNLESS(distributedType != nullptr, "Unsupported operand type {0}", operandType);

    auto perClusterShapes = distributedType.getPerClusterComputeShapes();
    VPUX_THROW_UNLESS(perClusterShapes.size() == checked_cast<size_t>(numClusters),
                      "Mismatch in shapes '{0}' and clusters '{1}'", perClusterShapes.size(), numClusters);
    const auto perClusterShapeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    VPUX_THROW_UNLESS(perClusterShapeOffsets.size() == checked_cast<size_t>(numClusters),
                      "Mismatch in shape offsets '{0}' and clusters '{1}'", perClusterShapeOffsets.size(), numClusters);

    return getPerClusterSWBuffers(ctx, loc, bufferName, clusterOp, innerOperand, perClusterShapes,
                                  perClusterShapeOffsets, numClusters, rewriter, log, allowDiscontinuousBuffers);
}

// Get split buffers of single-cluster CMX or DDR to match with subshapes
SmallVector<mlir::Value> vpux::VPUIP::getSplitBuffers(mlir::MLIRContext* ctx, mlir::Location loc, StringRef bufferName,
                                                      mlir::Value operand, SmallVector<vpux::Shape> shapes,
                                                      SmallVector<vpux::Shape> shapeOffsets, int64_t splitNum,
                                                      mlir::PatternRewriter& rewriter) {
    auto declBuff = operand.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(declBuff != nullptr, "Failed to get buffer offset for operand: {0}", operand);

    auto declBuffType = declBuff.getType().cast<vpux::NDTypeInterface>();
    auto operandType = operand.getType().cast<vpux::NDTypeInterface>();

    VPUX_THROW_UNLESS(shapes.size() == checked_cast<size_t>(splitNum), "Mismatch in shapes '{0}' and buffers '{1}'",
                      shapes.size(), splitNum);
    VPUX_THROW_UNLESS(shapeOffsets.size() == checked_cast<size_t>(splitNum),
                      "Mismatch in shape offsets '{0}' and buffers '{1}'", shapeOffsets.size(), splitNum);

    const auto memSpaceId = declBuffType.getMemSpace().getIndex();
    const auto memKind = declBuffType.getMemoryKind();
    VPUX_THROW_UNLESS(memSpaceId.has_value(), "Failed to extract section id");
    const auto symbolAttr = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(memKind), memSpaceId.value());
    const auto originStride = operandType.getStrides();

    auto insertionPoint = declBuff.getOperation();
    SmallVector<mlir::Value> buffers(splitNum);
    for (int64_t bufferId = 0; bufferId < splitNum; ++bufferId) {
        auto cmxBuffType = operandType.extractDenseTile(shapeOffsets[bufferId], shapes[bufferId]);
        cmxBuffType = cmxBuffType.changeStrides(originStride);
        cmxBuffType = cmxBuffType.changeMemSpace(symbolAttr);

        const auto strides = operandType.getStrides();
        Byte cmxOffset{declBuff.getByteOffset()};
        for (size_t axis = 0; axis < strides.size(); axis++) {
            cmxOffset += static_cast<Byte>(shapeOffsets[bufferId][Dim(axis)] * strides[Dim(axis)]);
        }

        const auto newLoc = appendLoc(loc, "_{0}_split_{1}", bufferName, bufferId);
        auto newCmxBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, newLoc, cmxBuffType,
                                                                    declBuff.getSection(), cmxOffset.count());
        insertionPoint = newCmxBuffer.getOperation();

        buffers[bufferId] = newCmxBuffer;
    }

    return buffers;
}

//
// MovePureViewOpBeforeCopy Utilities
//

VPU::DistributedTensorAttr vpux::VPUIP::getSOHDistAttrWithNewShape(mlir::MLIRContext* ctx,
                                                                   VPUIP::DistributedBufferType origDistType,
                                                                   ShapeRef newShape, VPU::ArchKind arch) {
    const auto origDistAttr = origDistType.getDistribution();
    VPUX_THROW_UNLESS(VPU::isSegmentedOverH(origDistAttr), "Input dist type is not SEGMENTED over H");

    const auto origShape = origDistType.getShape();
    if (origShape == newShape) {
        return origDistAttr;
    }

    auto isInputSparse = origDistType.isa<VPUIP::SparseBufferType>();
    const auto newHeightAlignment =
            VPU::getSOHMinimalHeightAlignment(newShape, origDistAttr.getNumClusters().getInt(), isInputSparse, arch);
    const auto newAlignment =
            newHeightAlignment == 1 ? nullptr : getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1, newHeightAlignment, 1});

    if (!VPU::isDistributedAttrWithExplicitShapesAndOffsets(origDistAttr)) {
        return VPU::DistributedTensorAttr::get(ctx, origDistAttr.getMode(), origDistAttr.getNumTiles(),
                                               origDistAttr.getKernel(), origDistAttr.getPads(),
                                               origDistAttr.getStrides(), origDistAttr.getNumClusters(), newAlignment,
                                               origDistAttr.getUniformDistributedSegments(), nullptr, nullptr, nullptr,
                                               nullptr, origDistAttr.getEqualMemoryAndComputeView());
    }

    // When DistributedTensorAttr has explicit per cluster memory/compute shapes, recompute them for the new shape
    // Since this method throws for any distribution mode other than SEGMENTED over H, it is safe to recompute the
    // memory/compute view
    auto perClusterMemoryShapes = vpux::getIntArrayOfArray(ctx, VPU::getPerClusterMemoryShapes(newShape, origDistAttr));
    auto perClusterMemoryOffsets =
            vpux::getIntArrayOfArray(ctx, VPU::getPerClusterMemoryShapeOffsets(newShape, origDistAttr));
    auto perClusterComputeShapes =
            vpux::getIntArrayOfArray(ctx, VPU::getPerClusterComputeShapes(newShape, origDistAttr));
    auto perClusterComputeOffsets =
            vpux::getIntArrayOfArray(ctx, VPU::getPerClusterComputeShapeOffsets(newShape, origDistAttr));

    return VPU::DistributedTensorAttr::get(
            ctx, origDistAttr.getMode(), origDistAttr.getNumTiles(), origDistAttr.getKernel(), origDistAttr.getPads(),
            origDistAttr.getStrides(), origDistAttr.getNumClusters(), newAlignment,
            origDistAttr.getUniformDistributedSegments(), perClusterComputeShapes, perClusterComputeOffsets,
            perClusterMemoryShapes, perClusterMemoryOffsets, origDistAttr.getEqualMemoryAndComputeView());
}

bool vpux::VPUIP::isDistributedCompatibleAfterShapeChange(VPUIP::DistributedBufferType inDistType,
                                                          VPUIP::DistributedBufferType outDistType) {
    auto inMode = inDistType.getDistribution().getMode().getValue();
    auto outMode = outDistType.getDistribution().getMode().getValue();

    if (inMode != outMode) {
        return false;
    }

    const auto inShape = inDistType.getShape();
    const auto outShape = outDistType.getShape();

    if (inShape.totalSize() != outShape.totalSize()) {
        return false;
    }

    if (inShape.size() != outShape.size()) {
        return false;
    }

    if (outDistType.getDistribution().getNumClusters() != inDistType.getDistribution().getNumClusters()) {
        return false;
    }

    if (VPU::bitEnumContains(inMode, VPU::DistributionMode::DUPLICATED) ||
        VPU::bitEnumContains(inMode, VPU::DistributionMode::MULTICASTED)) {
        return true;
    }

    // Check per-cluster shape compatible
    const auto inPerClusterShapes = inDistType.getPerClusterMemoryShapes();
    const auto inPerClusterShapeOffsets = inDistType.getPerClusterMemoryShapeOffsets();
    const auto outPerClusterShapes = outDistType.getPerClusterMemoryShapes();
    const auto outPerClusterShapeOffsets = outDistType.getPerClusterMemoryShapeOffsets();
    const auto inStrides = inDistType.getStrides();
    const auto outStrides = outDistType.getStrides();
    const auto calcBufferOffset = [](ShapeRef shapeOffset, Strides strides) {
        Byte bufOffset{0};
        for (size_t axis = 0; axis < strides.size(); axis++) {
            bufOffset += shapeOffset[Dim(axis)] * static_cast<Byte>(strides[Dim(axis)]);
        }
        return bufOffset.count();
    };
    const auto isPerClusterCompatible = [&](ShapeRef inShape, ShapeRef outShape, ShapeRef inShapeOffset,
                                            ShapeRef outShapeOffset) {
        if (inShape.totalSize() != outShape.totalSize()) {
            return false;
        }
        const auto inDataOffset = calcBufferOffset(inShapeOffset, inStrides);
        const auto outDataOffset = calcBufferOffset(outShapeOffset, outStrides);
        return inDataOffset == outDataOffset;
    };
    return llvm::all_of_zip(inPerClusterShapes, outPerClusterShapes, inPerClusterShapeOffsets,
                            outPerClusterShapeOffsets, isPerClusterCompatible);
}

bool vpux::VPUIP::isDistributedCompatibleAfterShapeChange(VPUIP::DistributedBufferType inDistType, ShapeRef shape,
                                                          VPU::ArchKind arch) {
    const auto mode = inDistType.getDistribution().getMode().getValue();
    VPUX_THROW_UNLESS(VPU::bitEnumContains(mode, VPU::DistributionMode::DUPLICATED) ||
                              VPU::bitEnumContains(mode, VPU::DistributionMode::SEGMENTED),
                      "Only support DUPLICATED and SEGMENTED mode.");
    const auto inShape = inDistType.getShape();
    if (inShape == shape) {
        return true;
    }
    if (inShape.totalSize() != shape.totalSize()) {
        return false;
    }
    if (VPU::bitEnumContains(mode, VPU::DistributionMode::DUPLICATED) ||
        VPU::bitEnumContains(mode, VPU::DistributionMode::MULTICASTED)) {
        return true;
    }
    // Check both original and new shape are 4D
    if (inShape.size() != shape.size() || inShape.size() != 4) {
        return false;
    }
    // Only NHWC layout is supported in SOH
    if (inDistType.getDimsOrder() != DimsOrder::NHWC) {
        return false;
    }
    // only SOH supported for SEGMENTED
    const auto inDistAttr = inDistType.getDistribution();
    if (!VPU::isSegmentedOverH(inDistAttr)) {
        return false;
    }
    if (shape[Dims4D::Act::H] < inDistAttr.getNumClusters().getInt()) {
        return false;
    }

    const auto isInputSparse = inDistType.isa<VPUIP::SparseBufferType>();
    const auto minHeightAlignment =
            VPU::getSOHMinimalHeightAlignment(shape, inDistAttr.getNumClusters().getInt(), isInputSparse, arch);
    const auto tilingScheme = parseIntArrayAttr<int64_t>(inDistAttr.getNumTiles());
    if (inDistAttr.getUniformDistributedSegments() == nullptr) {
        auto tiledShapeHeight = divUp(shape[Dims4D::Act::H], tilingScheme[Dims4D::Act::H.ind()]);
        tiledShapeHeight = alignValUp(tiledShapeHeight, minHeightAlignment);
        const auto remainderTileSize =
                shape[Dims4D::Act::H] - tiledShapeHeight * (tilingScheme[Dims4D::Act::H.ind()] - 1);
        if (remainderTileSize <= 0) {
            return false;
        }
    } else {
        auto tiledShapeHeight = shape[Dims4D::Act::H] / tilingScheme[Dims4D::Act::H.ind()];
        tiledShapeHeight = alignValDown(tiledShapeHeight, minHeightAlignment);
        if (tiledShapeHeight <= 0) {
            return false;
        }

        auto remainderCount = shape[Dims4D::Act::H] - tiledShapeHeight * tilingScheme[Dims4D::Act::H.ind()];
        if (remainderCount % minHeightAlignment) {
            return false;
        }
    }

    // Create dist type with new shape
    const auto ctx = inDistType.getContext();
    const auto order = mlir::AffineMapAttr::get(inDistType.getDimsOrder().toAffineMap(ctx));
    const auto newDistribution = getSOHDistAttrWithNewShape(ctx, inDistType, shape, arch);
    const auto outDistType = VPUIP::DistributedBufferType::get(ctx, shape.raw(), inDistType.getElementType(), order,
                                                               inDistType.getMemSpace(), newDistribution);
    if (newDistribution.getAlignment()) {
        auto newShape = outDistType.getShape();
        auto newAlignment = parseIntArrayAttr<int64_t>(newDistribution.getAlignment());
        if (newShape[Dims4D::Act::H] < newAlignment[Dims4D::Act::H.ind()]) {
            return false;
        }
    }
    return VPUIP::isDistributedCompatibleAfterShapeChange(inDistType, outDistType);
}

mlir::Operation* vpux::VPUIP::getRootConst(mlir::Value val) {
    if (auto rootGroup = val.getDefiningOp<VPUIP::GroupSparseBufferOp>()) {
        if (rootGroup.data().getDefiningOp<Const::DeclareOp>() == nullptr) {
            return nullptr;
        }
        const auto sparsityMap = rootGroup.sparsityMap();
        if (sparsityMap && sparsityMap.getDefiningOp<Const::DeclareOp>() == nullptr) {
            return nullptr;
        }
        return rootGroup;
    }
    return val.getDefiningOp<Const::DeclareOp>();
}

//
// Get tiling index of Distributed Type
//
namespace {
template <typename T>
Optional<int64_t> getTilingDimIndexFromDistributedType(T distributedType) {
    // Get tile index
    int64_t tileIndex = -1;

    const auto distributionAttr = distributedType.getDistribution();
    const auto mode = distributionAttr.getMode().getValue();

    if (VPU::bitEnumContains(mode, VPU::DistributionMode::DUPLICATED) ||
        VPU::bitEnumContains(mode, VPU::DistributionMode::MULTICASTED)) {
        // return None if no tiling dim
        return None;
    }

    const auto numTiles = parseIntArrayAttr<int64_t>(distributedType.getDistribution().getNumTiles());
    for (size_t i = 0; i < numTiles.size(); ++i) {
        if (numTiles[i] > 1) {
            VPUX_THROW_WHEN(tileIndex != -1, "distributed buffer only support tiling on one axis");
            tileIndex = static_cast<int64_t>(i);
        }
    }
    return tileIndex;
}

}  // namespace

Optional<int64_t> vpux::VPUIP::getTilingDimIndex(mlir::Type type) {
    if (auto distributedBufferType = type.dyn_cast<VPUIP::DistributedBufferType>()) {
        return getTilingDimIndexFromDistributedType(distributedBufferType);
    } else if (auto distributedTensorType = type.dyn_cast<VPU::DistributedTensorType>()) {
        return getTilingDimIndexFromDistributedType(distributedTensorType);
    }
    VPUX_THROW("Unsupported type {0} for checking tiling dim", type);
}

//
// Check if memory is contiguous with tiling
//

bool vpux::VPUIP::isMemoryContiguousWithTiling(VPUIP::DistributedBufferType distributedBufferType) {
    const auto distributionAttr = distributedBufferType.getDistribution();
    const auto mode = distributionAttr.getMode().getValue();

    if (VPU::bitEnumContains(mode, VPU::DistributionMode::DUPLICATED) ||
        VPU::bitEnumContains(mode, VPU::DistributionMode::MULTICASTED)) {
        return true;
    }

    // Get tile index
    const auto tileIndex = VPUIP::getTilingDimIndex(distributedBufferType);
    VPUX_THROW_UNLESS(tileIndex.has_value(), "Can not get tiling dim for {0}", distributedBufferType);
    const auto order = distributedBufferType.getDimsOrder();
    // Get tile dim position
    const auto tileDimPos = order.dimPos(Dim(tileIndex.value()));
    const auto memShape = distributedBufferType.getMemShape().raw();
    // Check if all dims outter than tile dim is 1
    for (size_t i = 0; i < tileDimPos; ++i) {
        if (memShape[i] != 1) {
            return false;
        }
    }

    return true;
}

//
// Compressed Convolution utility
//
namespace {
bool hasTransposeAttr(Const::ContentAttr content) {
    const auto transformations = content.getTransformations();
    for (auto transform : transformations) {
        if (auto transpose = transform.dyn_cast<vpux::Const::TransposeAttr>()) {
            return true;
        }
    }
    return false;
}

bool inChannelGreaterThanAlignValue(Const::DeclareOp weightsInput) {
    auto weightsContentAttr = weightsInput.getContentAttr();
    const auto origShape = weightsContentAttr.getBaseContent().getType().cast<NDTypeInterface>().getShape();
    const auto channelAlignValue =
            VPU::NCEInvariant::getAlignment(weightsInput.getType().cast<NDTypeInterface>().getElementType());

    return origShape[Dims4D::Filter::IC] >= channelAlignValue;
}
}  // namespace

bool vpux::VPUIP::canWeightsBeCompressed(VPUIP::NCEClusterTaskOp op) {
    if (op.task_type() != VPUIP::NCETaskType::CONV) {
        return false;
    }
    // Avoid compressing weights that are previously compressed in VPU dialect alongside input compression
    if (op.input_channels_compressionAttr() != nullptr && op.cm_sp_patternAttr() != nullptr) {
        return false;
    }

    // The compressed convolution feature makes use of a sparsity map for the weights internally
    // so it cannot work if a custom one is provided as well
    if (op.weights_sparsity_map() != nullptr) {
        return false;
    }

    auto weights = op.weights().getDefiningOp<VPUIP::CopyOp>();
    if (weights == nullptr) {
        return false;
    }

    auto weightsInput = weights.input().getDefiningOp<Const::DeclareOp>();
    if (weightsInput == nullptr) {
        return false;
    }

    // Temporary solution until [E#57202] implementation
    if (hasTransposeAttr(weightsInput.getContentAttr())) {
        return false;
    }
    return !inChannelGreaterThanAlignValue(weightsInput);
}

bool vpux::VPUIP::canTilingWeightsBeCompressed(VPUIP::NCEClusterTilingOp op) {
    auto nceOp = mlir::dyn_cast_or_null<VPUIP::NCEClusterTaskOp>(op.getInnerTaskOp());
    if (nceOp == nullptr) {
        return false;
    }

    if (nceOp.task_type() != VPUIP::NCETaskType::CONV) {
        return false;
    }
    // Avoid compressing weights that are previously compressed in VPU dialect alongside input compression
    if (nceOp.input_channels_compressionAttr() != nullptr && nceOp.cm_sp_patternAttr() != nullptr) {
        return false;
    }

    // The compressed convolution feature makes use of a sparsity map for the weights internally
    // so it cannot work if a custom one is provided as well
    if (nceOp.weights_sparsity_map() != nullptr) {
        return false;
    }

    auto weights = VPUIP::getTopBufferOfNCEClusterTiling(nceOp, nceOp.weights());
    if (weights == nullptr) {
        return false;
    }
    auto weightsBufferTilingOp = weights.getDefiningOp<VPUIP::NCEClusterTilingOp>();
    if (weightsBufferTilingOp == nullptr) {
        return false;
    }
    auto weightsCopyOp = weightsBufferTilingOp.getInnerTaskOpOfType<VPUIP::CopyOp>();
    if (weightsCopyOp == nullptr) {
        return false;
    }
    auto weightsInput = VPUIP::getTopBufferOfNCEClusterTiling(weightsCopyOp, weightsCopyOp.input())
                                .getDefiningOp<Const::DeclareOp>();
    if (weightsInput == nullptr) {
        return false;
    }

    // Temporary solution until [E#57202] implementation
    if (hasTransposeAttr(weightsInput.getContentAttr())) {
        return false;
    }

    return !inChannelGreaterThanAlignValue(weightsInput);
}

//
// Copy Utilities
//

bool vpux::VPUIP::isCopyWithStaticStrides(VPUIP::CopyOp copyOp) {
    auto subview = copyOp.output_buff().getDefiningOp<VPUIP::SubViewOp>();
    if (subview == nullptr) {
        return false;
    }
    if (subview != nullptr) {
        if (subview.static_stridesAttr() == nullptr) {
            return false;
        }

        auto strides = parseIntArrayAttr<int64_t>(subview.static_stridesAttr());
        return llvm::any_of(strides, [](auto stride) {
            return stride > 1;
        });
    }

    return true;
}

bool vpux::VPUIP::isCopyToDDR(VPUIP::CopyOp copyOp) {
    auto origOp = copyOp->getParentOfType<VPUIP::NCEClusterTilingOp>() == nullptr ? copyOp.getOperation()
                                                                                  : copyOp->getParentOp();
    return origOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getMemoryKind() == VPU::MemoryKind::DDR;
}

bool vpux::VPUIP::isCopyFromDDR(VPUIP::CopyOp copyOp) {
    auto origOp = copyOp->getParentOfType<VPUIP::NCEClusterTilingOp>() == nullptr ? copyOp.getOperation()
                                                                                  : copyOp->getParentOp();
    return origOp->getOperand(0).getType().cast<vpux::NDTypeInterface>().getMemoryKind() == VPU::MemoryKind::DDR;
}

// The concept of striding levels means that tensor is not contiguous in some number of dimensions.
// For a contiguous tensor that number equals to 0.
// A tensor with the following properties has striding level 1:
// sizes: [1, 360, 1280, 18]
// strides: [235929600 Bit, 655360 Bit, 512 Bit, 16 Bit]
// Since 18 * 16 bit = 288 bit which is less than 512 bit (previous stride)
// A tensor with striding level 2 would look like that:
// sizes: [1, 360, 1280, 18]
// strides: [471859200 Bit, 1310720 Bit, 512 Bit, 16 Bit]
// 18 * 16 bit = 288 bit < 512 bit
// 1280 * 512 bit = 655360 bit < 1310720 bit
//
// Striding on current dim is useless and can be ignored in case higher dimension size is equal to one
// For example, the tensor with the following properties has striding level 1
// Even though 216 * 4 < 4320 and 360 * 4320 < 3110400
// sizes:         [1, 360, 216, 4]
// strides: [3110400, 4320, 4, 1]

bool allHigherDimsAreEqualToOne(ArrayRef<int64_t> memDimsVec, size_t curDimInd) {
    for (size_t i = 0; i < curDimInd; i++) {
        if (memDimsVec[i] != 1) {
            return false;
        }
    }
    return true;
}

int64_t getStridingLevel(const mlir::Value val) {
    const auto dims = getShape(val);
    const auto strides = getStrides(val);
    const auto order = DimsOrder::fromValue(val);
    const auto dimsMemOrder = to_small_vector(order.toMemoryOrder(dims));
    const auto stridesMemOrder = to_small_vector(order.toMemoryOrder(strides));

    int64_t stridingLevel = 0;
    for (size_t ind = 1; ind < dimsMemOrder.size() && ind < stridesMemOrder.size(); ind++) {
        // Bypass current dimension if higher dimensions have size == 1
        if (allHigherDimsAreEqualToOne(makeArrayRef(dimsMemOrder), ind)) {
            continue;
        }
        if (dimsMemOrder[ind] * stridesMemOrder[ind] != stridesMemOrder[ind - 1]) {
            stridingLevel++;
        }
    }
    return stridingLevel;
}

int64_t getFirstStridingMemDimIdxFromValue(mlir::Value val) {
    const auto dims = getShape(val);
    const auto strides = getStrides(val);
    const auto order = DimsOrder::fromValue(val);
    const auto dimsMemOrder = to_small_vector(order.toMemoryOrder(dims));
    const auto stridesMemOrder = to_small_vector(order.toMemoryOrder(strides));

    for (size_t ind = 1; ind < dimsMemOrder.size() && ind < stridesMemOrder.size(); ind++) {
        // Bypass current dimension if higher dimensions have size == 1
        if (allHigherDimsAreEqualToOne(makeArrayRef(dimsMemOrder), ind)) {
            continue;
        }
        if (dimsMemOrder[ind] * stridesMemOrder[ind] != stridesMemOrder[ind - 1]) {
            return checked_cast<int64_t>(ind);
        }
    }
    return -1;
}

int64_t getFirstStridingMemDimIdx(mlir::Operation* op) {
    VPUX_THROW_WHEN(mlir::dyn_cast<VPUIP::CopyOp>(op) == nullptr && mlir::dyn_cast<VPUIP::NNDMAOp>(op) == nullptr,
                    "getFirstStridingMemDimIdx: not a CopyOp or NNDMAOp");
    auto firstStridingDim = getFirstStridingMemDimIdxFromValue(op->getOperand(0));
    if (firstStridingDim == -1) {
        firstStridingDim = getFirstStridingMemDimIdxFromValue(op->getResult(0));
    }

    return firstStridingDim;
}

// For CopyOp or NNDMAOp whoes data size is greater than VPUIP::DMA_LIMIT, split the first non-batch dimension,
// regardless the layout
// For example: NCHW - C, NHWC - H, NWHC - W
vpux::Dim vpux::VPUIP::getCopyDMATilingDimForLargeSize(mlir::Operation* op) {
    VPUX_THROW_WHEN(mlir::dyn_cast<VPUIP::CopyOp>(op) == nullptr && mlir::dyn_cast<VPUIP::NNDMAOp>(op) == nullptr,
                    "getCopyDMATilingDimForLargeSize: not a CopyOp or NNDMAOp");
    const auto inputShape = getShape(op->getOperand(0));
    const auto inOrder = DimsOrder::fromValue(op->getOperand(0));

    size_t index = 0;
    while (inputShape[inOrder.toDim(MemDim(index))] <= 1) {
        VPUX_THROW_UNLESS(index < inputShape.size(), "Unable to find a dim to tile over it");
        index++;
    }

    return inOrder.toDim(MemDim(index));
}

// For CopyOp or NNDMAOp whoes plane number is greater than VPUIP::CMX_DMA_MAX_NUM_PLANES, the next dimension of
// firstStridingDim desribes number of planes, split the tensor on it
// For example:
// Tensor memref<1x4x360x216xf16, {order = #NHWC, strides = [6220800, 1, 8640, 8]}, @DDR>
// dimW = 216 is the firstStridingDim, dim H(360) will be split
vpux::Dim vpux::VPUIP::getCopyDMATilingDimForLargePlaneNum(mlir::Operation* op) {
    VPUX_THROW_WHEN(mlir::dyn_cast<VPUIP::CopyOp>(op) == nullptr && mlir::dyn_cast<VPUIP::NNDMAOp>(op) == nullptr,
                    "getCopyDMATilingDimForLargePlaneNum: not a CopyOp or NNDMAOp");
    VPUX_THROW_UNLESS(isSplitNeededForLargePlanesNum(op),
                      "getCopyDMATilingDimForLargePlaneNum: operation {0} does not need split for large plane number",
                      *op);
    const auto inOrder = DimsOrder::fromValue(op->getOperand(0));
    auto firstStridingDim = getFirstStridingMemDimIdx(op);
    VPUX_THROW_UNLESS(firstStridingDim != -1, "At least one of the input or output of copy has stride");
    return inOrder.toDim(MemDim(firstStridingDim - 1));
}

int64_t vpux::VPUIP::getMaxStridingLevel(const VPU::ArchKind arch) {
    int64_t maxStridingLevel = 0;
    switch (arch) {
    case VPU::ArchKind::VPUX30XX:
    case VPU::ArchKind::VPUX37XX:
        maxStridingLevel = VPUIP::CMX_DMA_MAX_STRIDING_LEVEL_30XX_37XX;
        break;
    default:
        VPUX_THROW("Unsuported architecture for getMaxStridingLevel");
    }

    return maxStridingLevel;
}

int64_t vpux::VPUIP::getMaxNumberPlanes(const VPU::ArchKind arch) {
    int64_t maxNumberPlanes = 0;
    switch (arch) {
    case VPU::ArchKind::VPUX30XX:
    case VPU::ArchKind::VPUX37XX:
        maxNumberPlanes = VPUIP::CMX_DMA_MAX_NUM_PLANES_30XX_37XX;
        break;
    default:
        VPUX_THROW("Unsuported architecture for getMaxNumberPlanes");
    }

    return maxNumberPlanes;
}

// CopyOp or NNDMAop is split needed for large plane number in one of below two conditions:
// 1.Input has level 2 stride and input plane number is larger than 255
// 2.Output has level 2 stride and output plane number is larger than 255
bool vpux::VPUIP::isSplitNeededForLargePlanesNum(mlir::Operation* op) {
    VPUX_THROW_WHEN(mlir::dyn_cast<VPUIP::CopyOp>(op) == nullptr && mlir::dyn_cast<VPUIP::NNDMAOp>(op) == nullptr,
                    "isSplitNeededForLargePlanesNum: not a CopyOp or NNDMAOp");
    auto arch = VPU::getArch(op);
    int64_t inputStridingLevel = 0;
    int64_t outputStridingLevel = 0;
    int64_t maxStridingLevel = 0;
    inputStridingLevel = getStridingLevel(op->getOperand(0));
    outputStridingLevel = getStridingLevel(op->getResult(0));
    maxStridingLevel = getMaxStridingLevel(arch);
    if (inputStridingLevel > maxStridingLevel || outputStridingLevel > maxStridingLevel) {
        return false;
    }

    const auto inputShape = getShape(op->getOperand(0));
    const auto inOrder = DimsOrder::fromValue(op->getOperand(0));
    const auto inMemShape = inOrder.toMemoryOrder(inputShape);

    int64_t inputNumPlane = 0;
    int64_t outputNumPlane = 0;
    int64_t maxNumPlane = 0;
    if (inputStridingLevel == maxStridingLevel) {
        auto inputFirstStridingDim = getFirstStridingMemDimIdxFromValue(op->getOperand(0));
        inputNumPlane = inputFirstStridingDim >= 1 ? inMemShape[MemDim(inputFirstStridingDim - 1)] : 0;
    }

    if (outputStridingLevel == maxStridingLevel) {
        auto outputFirstStridingDim = getFirstStridingMemDimIdxFromValue(op->getResult(0));
        outputNumPlane = outputFirstStridingDim >= 1 ? inMemShape[MemDim(outputFirstStridingDim - 1)] : 0;
    }

    maxNumPlane = getMaxNumberPlanes(arch);

    return inputNumPlane > maxNumPlane || outputNumPlane > maxNumPlane;
}

// CopyOp and NNDMAop with legal striding level should meet below two requirments:
// 1.Input and output striding levels are both not larger than 2
// 2.This operation is not split needed for large plane number
bool vpux::VPUIP::hasLegalStridingLevel(mlir::Operation* op) {
    VPUX_THROW_WHEN(mlir::dyn_cast<VPUIP::CopyOp>(op) == nullptr && mlir::dyn_cast<VPUIP::NNDMAOp>(op) == nullptr,
                    "hasLegalStridingLevel: not a CopyOp or NNDMAOp");
    auto arch = VPU::getArch(op);
    int64_t inputStridingLevel = 0;
    int64_t outputStridingLevel = 0;
    inputStridingLevel = getStridingLevel(op->getOperand(0));
    outputStridingLevel = getStridingLevel(op->getResult(0));
    auto maxStridingLevel = getMaxStridingLevel(arch);
    if (inputStridingLevel > maxStridingLevel || outputStridingLevel > maxStridingLevel) {
        return false;
    }

    return !isSplitNeededForLargePlanesNum(op);
}

//
// Operation utility
//

bool VPUIP::isOpOnlySplitOnDim(VPUIP::SubViewOp op, Dim dim) {
    const auto inShape = getShape(op.source()).raw();
    const auto outShape = getShape(op.result()).raw();

    VPUX_THROW_UNLESS(inShape.size() == outShape.size(),
                      "input dim size {0} is not equal to output dim size {1} at '{2}'", inShape, outShape,
                      op->getLoc());

    int64_t dimsDifference = -1;
    for (size_t i = 0; i < inShape.size(); i++) {
        if (inShape[i] != outShape[i]) {
            if (dimsDifference != -1) {
                return false;
            }
            dimsDifference = i;
        }
    }
    return dimsDifference == dim.ind();
}

Byte VPUIP::getRequiredCMXSize(mlir::Operation* op) {
    auto isCMXUsed = [](mlir::Value value) {
        if (auto type = value.getType().dyn_cast<vpux::NDTypeInterface>()) {
            return type.getMemoryKind() == VPU::MemoryKind::CMX_NN;
        }
        return false;
    };

    SmallVector<vpux::NDTypeInterface> operandTypes;
    if (auto nceTaskOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(op)) {
        for (const auto& operand : op->getOperands()) {
            if (operand != nceTaskOp.parent_input() && operand != nceTaskOp.parent_output() && isCMXUsed(operand)) {
                operandTypes.push_back(operand.getType().dyn_cast<vpux::NDTypeInterface>());
            }
        }
    } else {
        for (const auto& operand : op->getOperands()) {
            if (isCMXUsed(operand)) {
                operandTypes.push_back(operand.getType().dyn_cast<vpux::NDTypeInterface>());
            }
        }
    }
    return VPU::getRequiredCMXSize(operandTypes);
}

Shape VPUIP::backInferD2SInputShape(Shape outShape, int64_t paddedOC, int64_t paddedIC, int64_t blockSize) {
    VPUX_THROW_UNLESS(outShape.size() == 4, "outShape does not have enough dims expected 4 got {0}", outShape.size());
    outShape[Dims4D::Act::H] /= blockSize;
    outShape[Dims4D::Act::W] /= blockSize;
    outShape[Dims4D::Act::C] = (outShape[Dims4D::Act::C] - paddedOC) * (blockSize * blockSize) + paddedIC;
    return outShape;
}
