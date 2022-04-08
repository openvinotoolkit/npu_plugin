//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/dma_descriptor_generator.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
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
        return originType.changeShapeElemType(shape, newQType);
    }

    return originType.changeShape(shape);
}

vpux::NDTypeInterface getPerClusterInputType(vpux::NDTypeInterface innerInputType,
                                             vpux::NDTypeInterface innerOutputType, mlir::AffineMap memPerm,
                                             ShapeRef outShape, ShapeRef offset) {
    auto inputShape = innerInputType.getShape();

    // Back infer the input shape from output shape and mem_Perm attribution
    // For example: Input: 1x8x1x32xfp16, #NHWC -> 1x32x1x8xfp16, #NHWC, memPerm: [0, 1, 3, 2]
    // If want get right input shape from per cluster output shape. There are three step:
    //   1) Get the output real physical shape: 1x32x1x8xfp16, #NHWC -> 1x1x8x32
    //   2) Using memPerm to back infer the input real physical shape: 1x1x8x32 -> 1x1x32x8
    //   3) Got the input logic shape: 1x1x32x8 -> 1x8x1x32xfp16, #NHWC
    const auto inOrder = innerInputType.getDimsOrder();
    const auto outOrder = innerOutputType.getDimsOrder();
    auto backInferInputShape = [&](ShapeRef subOutShape) -> Shape {
        // After Expand fuse into Permute and got one PermuteDMA Op
        // The channel size of input and output are not same
        // For example: input (NCHW) 1x3x32x32, output(NHWC) 1x16x32x32
        // The channel size need align with the input
        auto inLogicShape = to_small_vector(subOutShape);
        if (innerInputType.getShape().totalSize() != innerOutputType.getShape().totalSize()) {
            VPUX_THROW_UNLESS(subOutShape[Dims4D::Act::C] != inputShape[Dims4D::Act::C],
                              "Got unexpect input {0} output {1} type of PermuteDMA", innerInputType, innerOutputType);
            inLogicShape[Dims4D::Act::C.ind()] = inputShape[Dims4D::Act::C];
        }

        Shape outPhysicalShape(inLogicShape.size());
        for (const auto idx : irange(inLogicShape.size())) {
            outPhysicalShape[Dim(idx)] = inLogicShape[outOrder.dimAt(idx).ind()];
        }

        Shape inPhysicalShape(inLogicShape.size());
        for (const auto idx : irange(inLogicShape.size())) {
            inPhysicalShape[DimsOrder::fromAffineMap(memPerm).dimAt(idx)] = outPhysicalShape[Dim(idx)];
        }

        for (const auto idx : irange(inLogicShape.size())) {
            inLogicShape[inOrder.dimAt(idx).ind()] = inPhysicalShape[Dim(idx)];
        }

        return Shape(inLogicShape);
    };

    return changeShape(innerInputType, backInferInputShape(outShape), offset);
}

//
// PermuteRewriter
//

class PermuteRewriter final : public mlir::OpRewritePattern<VPUIP::PermuteDMAOp> {
public:
    PermuteRewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount, Logger log)
            : mlir::OpRewritePattern<VPUIP::PermuteDMAOp>(ctx), _dmaPortCount(dmaPortCount), _log(log) {
        setDebugName("PermuteRewriter");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::PermuteDMAOp permuteOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::LogicalResult unrollSegmentedOrOverlapped(VPUIP::NCEClusterTilingOp clusterOp, VPUIP::PermuteDMAOp permuteOp,
                                                    VPUIP::DistributedBufferType distributedType,
                                                    mlir::AffineMap memPerm, mlir::PatternRewriter& rewriter) const;

    mlir::LogicalResult unrollDuplicated(VPUIP::NCEClusterTilingOp clusterOp, VPUIP::PermuteDMAOp permuteOp,
                                         VPUIP::DistributedBufferType distributedType, mlir::AffineMap memPerm,
                                         mlir::PatternRewriter& rewriter) const;

    int64_t _dmaPortCount;
    Logger _log;
};

mlir::LogicalResult PermuteRewriter::matchAndRewrite(VPUIP::PermuteDMAOp permuteOp,
                                                     mlir::PatternRewriter& rewriter) const {
    if (auto clusterOp = permuteOp->getParentOfType<VPUIP::NCEClusterTilingOp>()) {
        _log.trace("process permute under NCEClusterTilingOp at {0}", permuteOp);

        const auto input = *clusterOp.getInputs().begin();
        const auto output = *clusterOp.getOutputs().begin();

        const auto innerInputType = permuteOp.input().getType().cast<vpux::NDTypeInterface>();
        const auto innerOutputType = permuteOp.output_buff().getType().cast<vpux::NDTypeInterface>();

        const auto inputType = input.getType().cast<vpux::NDTypeInterface>();
        const auto outputType = output.getType().cast<vpux::NDTypeInterface>();
        VPUX_THROW_UNLESS(inputType.getMemoryKind() == VPU::MemoryKind::DDR &&
                                  outputType.getMemoryKind() == VPU::MemoryKind::CMX_NN,
                          "Unexpected memory space. Got: input {0}, output {1}", inputType.getMemoryKind(),
                          outputType.getMemoryKind());

        const auto distributedType = outputType.dyn_cast<VPUIP::DistributedBufferType>();
        VPUX_THROW_WHEN(distributedType == nullptr, "Expect distributed type for permute op output, actual: {0}",
                        outputType);

        VPUX_THROW_UNLESS(permuteOp.mem_perm().hasValue(), "Can not get memPerm attribute from PermuteDMA layer at {0}",
                          permuteOp.getLoc());
        const auto memPerm = permuteOp.mem_perm().getValue();
        VPUX_THROW_UNLESS(VPUIP::doesPermuteDMATileDimSupportWrapInCluster(innerInputType, innerOutputType, memPerm,
                                                                           distributedType, _log),
                          "Unsupported PermuteDMA under cluster tiling at '{0}'", permuteOp->getLoc());

        const auto distributionAttr = distributedType.getDistribution();
        const auto mode = distributionAttr.mode().getValue();
        if (mode == VPU::DistributionMode::SEGMENTED || mode == VPU::DistributionMode::OVERLAPPED) {
            return unrollSegmentedOrOverlapped(clusterOp, permuteOp, distributedType, memPerm, rewriter);
        } else if (VPU::bitEnumContains(mode, VPU::DistributionMode::DUPLICATED) ||
                   VPU::bitEnumContains(mode, VPU::DistributionMode::MULTICASTED)) {
            return unrollDuplicated(clusterOp, permuteOp, distributedType, memPerm, rewriter);
        } else {
            VPUX_THROW("Unsupported distributed mode");
        }
    }

    _log.trace("Permute rewriter operation '{0}' at '{1}'", permuteOp->getName(), permuteOp->getLoc());
    // Skip PermuteDMA ops which have been unrolled by checking mem_perm attribute
    if (permuteOp.mem_permAttr() == nullptr) {
        return mlir::failure();
    }

    auto vpurtTask = permuteOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_UNLESS(vpurtTask != nullptr, "Can't get VPURT task operation");
    auto cycleBeginAttr = vpurtTask->getAttr(cycleBegin);
    auto cycleEndAttr = vpurtTask->getAttr(cycleEnd);
    rewriter.setInsertionPointAfter(vpurtTask);

    auto srcDeclBuff = permuteOp.input().getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(srcDeclBuff != nullptr, "Can't get buffer for operand: {0}", permuteOp.input());

    auto dstDeclBuff = permuteOp.output_buff().getDefiningOp<VPURT::DeclareBufferOp>();

    auto inType = permuteOp.input().getType().cast<vpux::NDTypeInterface>();
    auto outType = permuteOp.output().getType().cast<vpux::NDTypeInterface>();
    Byte elemTypeSize = inType.getElemTypeSize();

    auto srcType = srcDeclBuff.getType().cast<vpux::NDTypeInterface>();
    auto dstType = dstDeclBuff.getType().cast<vpux::NDTypeInterface>();
    auto srcOffset = srcDeclBuff.byteOffset();
    auto dstOffset = dstDeclBuff.byteOffset();

    // For unrolled DMA which is inside of cluster tiling, the dma descriptor is already calculated
    auto dmaDescriptorAttr = permuteOp.dma_descriptorAttr();
    const auto memPerm = permuteOp.mem_perm().getValue();
    auto mergedMemPerm = VPUIP::getPermuteDMAMergedMemPerm(inType, memPerm);
    auto numPlaneDim = VPUIP::getPermuteDMANumPlaneDim(inType, memPerm);

    auto portIsAlreadyAssigned = true;
    if (dmaDescriptorAttr == nullptr) {
        auto ctx = permuteOp->getContext();
        auto mergedInputShape = VPUIP::getPermuteDMAInputShape(inType, outType, memPerm, _log).getValue();
        auto mergedOutputShape = VPUIP::getPermuteDMAOutputShape(inType, outType, memPerm, _log).getValue();
        auto dmaDescriptorGenerator = VPUIP::PermuteDmaDescriptorGenerator(ctx, mergedMemPerm, _log);
        dmaDescriptorAttr = dmaDescriptorGenerator.generate(mergedInputShape, mergedOutputShape, elemTypeSize);
        portIsAlreadyAssigned = false;
    }

    auto subInput = VPUIP::getPermuteDMASubInputShapes(inType, outType, memPerm, _log);
    VPUX_THROW_UNLESS(subInput.hasValue(), "Cannot get unrolled subInputShapes for PermuteDMA op {0}", permuteOp);
    auto subInputShapes = subInput.getValue();
    auto subOutputShapes = VPUIP::getPermuteDMASubOutputShapes(subInputShapes, inType, outType, memPerm);

    _log.trace("Unrolling PermuteDMAOp '{0}' at '{1}'", permuteOp->getName(), permuteOp->getLoc());

    int64_t dmaPort = 0;
    SmallVector<VPUIP::PermuteDMAOp> firstPermuteDMAsOnPorts;
    SmallVector<VPUIP::PermuteDMAOp> lastPermuteDMAsOnPorts;
    SmallVector<VPUIP::PermuteDMAOp> newPermuteDMAs;
    for (auto idx = 0; idx < checked_cast<int64_t>(subInputShapes.size()); idx++) {
        auto newDmaDescriptorAttr = VPUIP::updateNumPlanes(dmaDescriptorAttr, subInputShapes[idx][numPlaneDim]);

        const auto dimOrder = (subInputShapes[0].size() == 2) ? DimsOrder::NC : DimsOrder::CHW;
        auto newSrcStrides =
                (subInputShapes[idx].size() == 2)
                        ? SmallVector<vpux::Bit>{Bit(subInputShapes[idx].back() * Bit(elemTypeSize).count()),
                                                 Bit(Bit(elemTypeSize).count())}
                        : SmallVector<vpux::Bit>{Bit(subInputShapes[idx][Dim(1)] * subInputShapes[idx][Dim(2)] *
                                                     Bit(elemTypeSize).count()),
                                                 Bit(subInputShapes[idx].back() * Bit(elemTypeSize).count()),
                                                 Bit(Bit(elemTypeSize).count())};

        auto newSrcMemRef = vpux::getMemRefType(subInputShapes[idx], srcType.getElementType(), dimOrder,
                                                srcType.getMemSpace(), Strides(newSrcStrides));

        auto newSrcBuff = srcType.getMemSpace().getIndex().hasValue()
                                  ? VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, srcDeclBuff, vpurtTask.getLoc(),
                                                                            newSrcMemRef, srcDeclBuff.section(),
                                                                            srcType.getMemSpace().getIndex().getValue(),
                                                                            srcOffset)
                                  : srcDeclBuff.sectionIndex().hasValue()
                                            ? VPURT::createOp<VPURT::DeclareBufferOp>(
                                                      rewriter, srcDeclBuff, vpurtTask.getLoc(), newSrcMemRef,
                                                      srcDeclBuff.section(),
                                                      parseIntArrayAttr<int64_t>(srcDeclBuff.sectionIndex().getValue()),
                                                      srcOffset)
                                            : VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, srcDeclBuff,
                                                                                      vpurtTask.getLoc(), newSrcMemRef,
                                                                                      srcDeclBuff.section(), srcOffset);

        auto newDstStrides =
                (subOutputShapes[idx].size() == 2)
                        ? SmallVector<vpux::Bit>{Bit(subOutputShapes[idx].back() * Bit(elemTypeSize).count()),
                                                 Bit(Bit(elemTypeSize).count())}
                        : SmallVector<vpux::Bit>{Bit(subOutputShapes[idx][Dim(1)] * subOutputShapes[idx][Dim(2)] *
                                                     Bit(elemTypeSize).count()),
                                                 Bit(subOutputShapes[idx][Dim(2)] * Bit(elemTypeSize).count()),
                                                 Bit(Bit(elemTypeSize).count())};
        mlir::Type newDstType;
        if (auto dstDistributedType = dstType.dyn_cast<VPUIP::DistributedBufferType>()) {
            auto ctx = permuteOp->getContext();
            const auto distributionAttr = dstDistributedType.getDistribution();
            const auto layout = mlir::AffineMapAttr::get(dimOrder.toAffineMap(ctx));
            newDstType = VPUIP::DistributedBufferType::get(ctx, subOutputShapes[idx].raw(), dstType.getElementType(),
                                                           layout, dstType.getMemSpace(), distributionAttr);
        } else {
            newDstType = vpux::getMemRefType(subOutputShapes[idx], dstType.getElementType(), dimOrder,
                                             dstType.getMemSpace(), Strides(newDstStrides));
        }

        VPUX_THROW_UNLESS(dstType.getMemSpace().getIndex().hasValue() || dstDeclBuff.sectionIndex().hasValue(),
                          "No section index find at '{}'", dstDeclBuff.getLoc());
        auto newDstBuff = dstType.getMemSpace().getIndex().hasValue()
                                  ? VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, dstDeclBuff, vpurtTask.getLoc(),
                                                                            newDstType, dstDeclBuff.section(),
                                                                            dstType.getMemSpace().getIndex().getValue(),
                                                                            dstOffset)
                                  : dstDeclBuff.sectionIndex().hasValue()
                                            ? VPURT::createOp<VPURT::DeclareBufferOp>(
                                                      rewriter, dstDeclBuff, vpurtTask.getLoc(), newDstType,
                                                      dstDeclBuff.section(),
                                                      parseIntArrayAttr<int64_t>(dstDeclBuff.sectionIndex().getValue()),
                                                      dstOffset)
                                            : VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, dstDeclBuff,
                                                                                      vpurtTask.getLoc(), newDstType,
                                                                                      dstDeclBuff.section(), dstOffset);

        _log.trace("Create unrolled PermuteDMA operation with input/output shape: {0}/{1}, SrcMemory at {2}, "
                   "DstMemory at {3}",
                   subInputShapes[idx], subOutputShapes[idx], newSrcBuff.section(), newDstBuff.section());

        const auto newLoc = appendLoc(vpurtTask->getLoc(), "_unrolled_permuteDMA");
        auto newDmaPort = portIsAlreadyAssigned ? permuteOp.port() : dmaPort;
        auto newPermuteDMAOp = VPURT::wrapIntoTaskOp<VPUIP::PermuteDMAOp>(
                rewriter, vpurtTask.waitBarriers(), vpurtTask.updateBarriers(), newLoc, newSrcBuff, newDstBuff, nullptr,
                newDmaDescriptorAttr, vpux::getIntAttr(rewriter, newDmaPort));
        newPermuteDMAs.push_back(newPermuteDMAOp);

        // find the first and last DMAs on different ports
        if (firstPermuteDMAsOnPorts.size() < static_cast<size_t>(_dmaPortCount)) {
            firstPermuteDMAsOnPorts.push_back(newPermuteDMAOp);
            lastPermuteDMAsOnPorts.push_back(newPermuteDMAOp);
        } else {
            lastPermuteDMAsOnPorts[newDmaPort] = newPermuteDMAOp;
        }

        dmaPort = (dmaPort + 1) % _dmaPortCount;

        auto newVpurtTask = newPermuteDMAOp->getParentOfType<VPURT::TaskOp>();
        if (cycleBeginAttr) {
            newVpurtTask->setAttr(cycleBegin, cycleBeginAttr);
        }
        if (cycleEndAttr) {
            newVpurtTask->setAttr(cycleEnd, cycleEndAttr);
        }

        auto numPlaneValue = newDmaDescriptorAttr.numPlanes().getInt();
        auto srcPlaneStrideValue = newDmaDescriptorAttr.srcPlaneStride().getInt();
        auto dstPlaneStrideVlaue = newDmaDescriptorAttr.dstPlaneStride().getInt();
        srcOffset += numPlaneValue * srcPlaneStrideValue;
        dstOffset += numPlaneValue * dstPlaneStrideVlaue;
    }

    for (auto dmaOp : newPermuteDMAs) {
        auto vpurtTask = dmaOp->getParentOfType<VPURT::TaskOp>();

        // remove wait barrier dependency for these new permute DMA except first ones on each port
        if (std::find(firstPermuteDMAsOnPorts.begin(), firstPermuteDMAsOnPorts.end(), dmaOp) ==
            firstPermuteDMAsOnPorts.end()) {
            vpurtTask.waitBarriersMutable().clear();
        }

        // remove update barrier dependency for these new permute DMA except last ones on each port
        if (std::find(lastPermuteDMAsOnPorts.begin(), lastPermuteDMAsOnPorts.end(), dmaOp) ==
            lastPermuteDMAsOnPorts.end()) {
            vpurtTask.updateBarriersMutable().clear();
        }
    }

    rewriter.eraseOp(vpurtTask);
    return mlir::success();
}

mlir::LogicalResult PermuteRewriter::unrollSegmentedOrOverlapped(VPUIP::NCEClusterTilingOp clusterOp,
                                                                 VPUIP::PermuteDMAOp permuteOp,
                                                                 VPUIP::DistributedBufferType distributedType,
                                                                 mlir::AffineMap memPerm,
                                                                 mlir::PatternRewriter& rewriter) const {
    auto loc = permuteOp->getLoc();
    auto ctx = permuteOp->getContext();

    const auto input = *clusterOp.getInputs().begin();
    const auto output = *clusterOp.getOutputs().begin();

    const auto innerInputType = permuteOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto innerOutputType = permuteOp.output_buff().getType().cast<vpux::NDTypeInterface>();

    const auto distributionAttr = distributedType.getDistribution();
    const auto numClusters = distributionAttr.num_clusters().getInt();
    const auto mode = distributionAttr.mode().getValue();
    VPUX_THROW_UNLESS(mode == VPU::DistributionMode::SEGMENTED || mode == VPU::DistributionMode::OVERLAPPED,
                      "Unsupported distributed mode");
    const auto perClusterOutShapes = distributedType.getPerClusterComputeShapes();
    const auto perClusterShapeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    auto cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));

    auto vpurtTask = clusterOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(vpurtTask == nullptr, "Can not get VPURT.TaskOp for {0}", permuteOp);
    auto cycleBeginAttr = vpurtTask->getAttr(cycleBegin);
    auto cycleEndAttr = vpurtTask->getAttr(cycleEnd);

    const auto tileInnerInputType = [&](vpux::NDTypeInterface innerInputType, vpux::NDTypeInterface innerOutputType) {
        SmallVector<vpux::NDTypeInterface> newTypes(numClusters);
        for (size_t clusterId = 0; clusterId < perClusterOutShapes.size(); ++clusterId) {
            newTypes[clusterId] =
                    getPerClusterInputType(innerInputType, innerOutputType, memPerm, perClusterOutShapes[clusterId],
                                           perClusterShapeOffsets[clusterId]);
        }
        return newTypes;
    };

    const auto tileInnerOutputType = [&](vpux::NDTypeInterface innerType) {
        SmallVector<vpux::NDTypeInterface> newTypes(numClusters);
        for (size_t clusterId = 0; clusterId < perClusterOutShapes.size(); ++clusterId) {
            newTypes[clusterId] =
                    changeShape(innerType, perClusterOutShapes[clusterId], perClusterShapeOffsets[clusterId]);
        }
        return newTypes;
    };

    auto inTypes = tileInnerInputType(innerInputType, innerOutputType);
    const auto originStride = innerInputType.getStrides();

    for (size_t clusterId = 0; clusterId < perClusterOutShapes.size(); ++clusterId) {
        inTypes[clusterId] = inTypes[clusterId].changeStrides(originStride);
    }

    const auto outTypes = tileInnerOutputType(innerOutputType);

    rewriter.setInsertionPointAfter(vpurtTask);
    const auto getOperand = [&](int64_t clusterId, mlir::Value operand, vpux::NDTypeInterface newType,
                                mlir::Operation* insertionPoint, Byte offset) -> mlir::Value {
        if (auto cst = operand.getDefiningOp<Const::DeclareOp>()) {
            return rewriter.create<VPUIP::SubViewOp>(permuteOp->getLoc(), cst, perClusterShapeOffsets[clusterId].raw(),
                                                     perClusterOutShapes[clusterId].raw());
        }

        auto declBuff = operand.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(declBuff != nullptr, "Can't get buffer offset");

        if (newType.getMemoryKind() == VPU::MemoryKind::CMX_NN) {
            const auto symbolAttr = vpux::IndexedSymbolAttr::get(ctx, {cmxNameAttr, vpux::getIntAttr(ctx, clusterId)});
            auto newCMXType = newType.changeMemSpace(symbolAttr);

            return VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, permuteOp->getLoc(), newCMXType,
                                                           VPURT::BufferSection::CMX_NN,
                                                           getIntArrayAttr(ctx, makeArrayRef({clusterId})),
                                                           declBuff.byteOffset(), declBuff.swizzlingKeyAttr());
        }

        Byte ddrOffset{declBuff.byteOffset()};
        ddrOffset += offset;

        return VPUIP::createNewDeclareBuffer(rewriter, insertionPoint, declBuff, newType, ddrOffset);
    };

    auto mergedInputShape = VPUIP::getPermuteDMAInputShape(innerInputType, innerOutputType, memPerm, _log).getValue();
    auto mergedOutputShape = VPUIP::getPermuteDMAOutputShape(innerInputType, innerOutputType, memPerm, _log).getValue();
    auto mergedMemPerm = VPUIP::getPermuteDMAMergedMemPerm(innerInputType, memPerm);
    auto dmaDescriptorGenerator = VPUIP::PermuteDmaDescriptorGenerator(ctx, mergedMemPerm, _log);
    auto elemTypeSize = Byte(innerInputType.getElemTypeSize());

    // calculate the dma descriptors and ddr offsets
    SmallVector<VPUIP::DmaDescriptorAttr> subDmaDescriptors;
    SmallVector<Byte> ddrOffsets;
    SmallVector<Shape> subMergedOutputShapes;
    Byte ddrOffset(0);

    const auto mergedOutputDimList = VPUIP::getPermuteDMAOutputMergedDimList(innerOutputType, mergedOutputShape);
    auto tileDimForMergedOutput =
            VPUIP::getTileDimForPermuteDMA(innerInputType, innerOutputType, memPerm, distributedType, _log).getValue();

    const auto numTileSize = parseIntArrayAttr<int64_t>(distributionAttr.num_tiles());
    const auto tileDimIter = std::find_if(numTileSize.begin(), numTileSize.end(), [](const int64_t dim) {
        return dim > 1;
    });
    VPUX_THROW_UNLESS(tileDimIter != numTileSize.end(), "Can not find tile dim.");
    auto tileDim = Dim(std::distance(numTileSize.begin(), tileDimIter));

    auto getSrcOffset = [&](vpux::ShapeRef offset) -> vpux::Byte {
        auto outputShape = innerOutputType.getShape();

        const auto splitDimList = mergedOutputDimList[tileDimForMergedOutput.ind()];
        VPUX_THROW_UNLESS(std::any_of(splitDimList.begin(), splitDimList.end(),
                                      [&](vpux::Dim dim) {
                                          return dim == tileDim;
                                      }),
                          "tileDim is not exist in splitDimList.");

        const auto totalOffsetSize = mergedOutputShape[tileDimForMergedOutput];
        return Byte(totalOffsetSize / outputShape[tileDim] * offset[tileDim] * elemTypeSize.count());
    };

    for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        auto mergedSubOutputShape =
                VPUIP::getPermuteDMAOutputShape(inTypes[clusterId], outTypes[clusterId], memPerm, _log).getValue();
        ddrOffsets.push_back(getSrcOffset(perClusterShapeOffsets[clusterId]));
        subMergedOutputShapes.push_back(mergedSubOutputShape);
    }
    subDmaDescriptors = dmaDescriptorGenerator.generate(mergedInputShape, mergedOutputShape, subMergedOutputShapes,
                                                        tileDimForMergedOutput, elemTypeSize);

    int64_t dmaPort = 0;
    auto inputInsertionPoint = input.getDefiningOp();
    auto outputInsertionPoint = output.getDefiningOp();
    for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        const auto newInputType = inTypes[clusterId];
        const auto newOutType = outTypes[clusterId];

        const auto inputBuffer = getOperand(clusterId, input, newInputType, inputInsertionPoint, ddrOffsets[clusterId]);
        inputInsertionPoint = inputBuffer.getDefiningOp();
        _log.trace("Insert new input buffer declaration: '{0}'", inputBuffer);

        const auto outBuffer = getOperand(clusterId, output, newOutType, outputInsertionPoint, Byte(0));
        outputInsertionPoint = outBuffer.getDefiningOp();
        _log.trace("Insert new output buffer declaration: '{0}'", outBuffer);

        const auto newLoc = appendLoc(loc, "_cluster_{0}", clusterId);
        auto newPermuteDMAOp = VPURT::wrapIntoTaskOp<VPUIP::PermuteDMAOp>(
                rewriter, vpurtTask.waitBarriers(), vpurtTask.updateBarriers(), newLoc, inputBuffer, outBuffer,
                permuteOp.mem_permAttr(), subDmaDescriptors[clusterId], vpux::getIntAttr(rewriter, dmaPort));

        dmaPort = (dmaPort + 1) % _dmaPortCount;

        _log.trace("Insert new permute dma : '{0}'", newPermuteDMAOp);
        auto newTaskOp = newPermuteDMAOp->getParentOfType<VPURT::TaskOp>();
        newTaskOp->setAttr(cycleBegin, cycleBeginAttr);
        newTaskOp->setAttr(cycleEnd, cycleEndAttr);
    }
    rewriter.eraseOp(vpurtTask);
    return mlir::success();
}

mlir::LogicalResult PermuteRewriter::unrollDuplicated(VPUIP::NCEClusterTilingOp clusterOp,
                                                      VPUIP::PermuteDMAOp permuteOp,
                                                      VPUIP::DistributedBufferType distributedType,
                                                      mlir::AffineMap memPerm, mlir::PatternRewriter& rewriter) const {
    auto loc = permuteOp->getLoc();
    auto ctx = permuteOp->getContext();

    const auto input = *clusterOp.getInputs().begin();
    const auto output = *clusterOp.getOutputs().begin();

    const auto innerInputType = permuteOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto innerOutputType = permuteOp.output_buff().getType().cast<vpux::NDTypeInterface>();

    const auto distributionAttr = distributedType.getDistribution();
    const auto numClusters = distributionAttr.num_clusters().getInt();
    VPUX_THROW_WHEN(numClusters == 0, "Invalid number of clusters for {0}", distributedType);

    SmallVector<int64_t> clusters(numClusters);
    std::iota(clusters.begin(), clusters.end(), 0);

    auto vpurtTask = clusterOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(vpurtTask == nullptr, "Can not get VPURT.TaskOp for {0}", permuteOp);
    auto cycleBeginAttr = vpurtTask->getAttr(cycleBegin);
    auto cycleEndAttr = vpurtTask->getAttr(cycleEnd);

    const auto mode = distributionAttr.mode().getValue();
    VPUX_THROW_UNLESS(mode == VPU::DistributionMode::DUPLICATED, "Unsupported distributed mode");

    rewriter.setInsertionPointAfter(vpurtTask);

    const auto perClusterOutShape = distributedType.getPerClusterComputeShapes().front();
    const auto perClusterShapeOffset = distributedType.getPerClusterComputeShapeOffsets().front();

    const auto getOperand = [&](mlir::Value operand, vpux::NDTypeInterface newType) -> mlir::Value {
        if (auto cst = operand.getDefiningOp<Const::DeclareOp>()) {
            return rewriter.create<VPUIP::SubViewOp>(permuteOp->getLoc(), cst, perClusterShapeOffset.raw(),
                                                     perClusterOutShape.raw());
        }
        auto insertionPoint = operand.getDefiningOp();

        auto declBuff = operand.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(declBuff != nullptr, "Can't get buffer offset");

        if (newType.getMemoryKind() == VPU::MemoryKind::CMX_NN) {
            return VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, permuteOp->getLoc(), newType,
                                                           VPURT::BufferSection::CMX_NN, getIntArrayAttr(ctx, clusters),
                                                           declBuff.byteOffset(), declBuff.swizzlingKeyAttr());
        }
        return VPUIP::createNewDeclareBuffer(rewriter, insertionPoint, declBuff, newType, Byte(0));
    };

    auto mergedInputShape = VPUIP::getPermuteDMAInputShape(innerInputType, innerOutputType, memPerm, _log).getValue();
    auto mergedOutputShape = VPUIP::getPermuteDMAOutputShape(innerInputType, innerOutputType, memPerm, _log).getValue();
    auto mergedMemPerm = VPUIP::getPermuteDMAMergedMemPerm(innerInputType, memPerm);
    auto dmaDescriptorGenerator = VPUIP::PermuteDmaDescriptorGenerator(ctx, mergedMemPerm, _log);
    auto elemTypeSize = Byte(innerInputType.getElemTypeSize());

    // calculate the dma descriptor
    VPUIP::DmaDescriptorAttr subDmaDescriptor =
            dmaDescriptorGenerator.generate(mergedInputShape, mergedOutputShape, elemTypeSize);
    const auto newInputType =
            getPerClusterInputType(innerInputType, innerOutputType, memPerm, perClusterOutShape, perClusterShapeOffset);
    auto newOutType = distributedType.changeShape(perClusterOutShape);

    const auto inputBuffer = getOperand(input, newInputType);
    _log.trace("Insert new input buffer declaration: '{0}'", inputBuffer);
    const auto outBuffer = getOperand(output, newOutType);
    _log.trace("Insert new output buffer declaration: '{0}'", outBuffer);

    auto newPermuteDMAOp = VPURT::wrapIntoTaskOp<VPUIP::PermuteDMAOp>(
            rewriter, vpurtTask.waitBarriers(), vpurtTask.updateBarriers(), loc, inputBuffer, outBuffer,
            permuteOp.mem_permAttr(), subDmaDescriptor);

    _log.trace("Insert new permute dma : '{0}'", newPermuteDMAOp);
    auto newTaskOp = newPermuteDMAOp->getParentOfType<VPURT::TaskOp>();
    newTaskOp->setAttr(cycleBegin, cycleBeginAttr);
    newTaskOp->setAttr(cycleEnd, cycleEndAttr);
    rewriter.eraseOp(vpurtTask);
    return mlir::success();
}

//
// UnrollPermuteToNNDMAPass
//

class UnrollPermuteToNNDMAPass final : public VPUIP::UnrollPermuteToNNDMABase<UnrollPermuteToNNDMAPass> {
public:
    explicit UnrollPermuteToNNDMAPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollPermuteToNNDMAPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto dmaOp = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    auto dmaPortCount = dmaOp.count();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<PermuteRewriter>(&ctx, dmaPortCount, _log.nest());
    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUnrollPermuteToNNDMAPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUnrollPermuteToNNDMAPass(Logger log) {
    return std::make_unique<UnrollPermuteToNNDMAPass>(log);
}
