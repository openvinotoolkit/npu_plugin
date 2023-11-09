//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/dma_descriptor_generator.hpp"
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

//
// PerAxisTileDMARewriter
//

class PerAxisTileDMARewriter final : public mlir::OpRewritePattern<VPUIP::PerAxisTileDMAOp> {
public:
    PerAxisTileDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount, Logger log)
            : mlir::OpRewritePattern<VPUIP::PerAxisTileDMAOp>(ctx), _dmaPortCount(dmaPortCount), _log(log) {
        setDebugName("PerAxisTileDMARewriter");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::PerAxisTileDMAOp perAxisTileDMAOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    mlir::LogicalResult unrollSegmentedOrOverlapped(VPUIP::PerAxisTileDMAOp perAxisTileDMAOp,
                                                    VPUIP::DistributedBufferType distributedType,
                                                    mlir::PatternRewriter& rewriter) const;

    mlir::LogicalResult unrollDuplicated(VPUIP::PerAxisTileDMAOp perAxisTileDMAOp,
                                         VPUIP::DistributedBufferType distributedType,
                                         mlir::PatternRewriter& rewriter) const;

    mlir::LogicalResult unrollPerAxisTile(VPUIP::PerAxisTileDMAOp perAxisTileDMAOp,
                                          mlir::PatternRewriter& rewriter) const;

    int64_t _dmaPortCount;
    Logger _log;
};

vpux::NDTypeInterface changeShape(vpux::NDTypeInterface originType, ShapeRef shape, ShapeRef offset, int64_t padAxis) {
    const auto elemType = originType.getElementType();
    auto newShape = to_small_vector(shape);
    newShape[padAxis] = originType.getShape()[Dim(padAxis)];
    if (auto qType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto newQType = tileScalesAndZP(qType, ShapeRef(newShape), offset);
        return originType.changeShapeElemType(ShapeRef(newShape), newQType);
    }

    return originType.changeShape(ShapeRef(newShape));
}

mlir::LogicalResult PerAxisTileDMARewriter::matchAndRewrite(VPUIP::PerAxisTileDMAOp perAxisTileDMAOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("Process PerAxisTileDMAOp: {0}", perAxisTileDMAOp);

    if (perAxisTileDMAOp.tilesAttr() == nullptr && perAxisTileDMAOp.axisAttr() == nullptr) {
        return mlir::failure();
    }

    const auto output = perAxisTileDMAOp.output();
    const auto outputType = output.getType().cast<vpux::NDTypeInterface>();
    const auto distributedType = outputType.dyn_cast<VPUIP::DistributedBufferType>();

    // if PerAxisTileDMAOp has output of DistributedType -- unroll according to DistributedType first,
    // then unroll per axis/tile
    if (distributedType != nullptr) {
        _log.trace("PerAxisTile Op with distributed type at {0}", perAxisTileDMAOp);

        const auto distributionAttr = distributedType.getDistribution();
        const auto mode = distributionAttr.getMode().getValue();
        if (mode == VPU::DistributionMode::SEGMENTED || mode == VPU::DistributionMode::OVERLAPPED) {
            return unrollSegmentedOrOverlapped(perAxisTileDMAOp, distributedType, rewriter);
        } else if (VPU::bitEnumContains(mode, VPU::DistributionMode::DUPLICATED) ||
                   VPU::bitEnumContains(mode, VPU::DistributionMode::MULTICASTED)) {
            return unrollDuplicated(perAxisTileDMAOp, distributedType, rewriter);
        } else {
            VPUX_THROW("Unsupported distributed mode");
        }
    }
    // otherwise -- unroll only per axis/tile
    return unrollPerAxisTile(perAxisTileDMAOp, rewriter);
}

mlir::LogicalResult PerAxisTileDMARewriter::unrollPerAxisTile(VPUIP::PerAxisTileDMAOp perAxisTileDMAOp,
                                                              mlir::PatternRewriter& rewriter) const {
    auto ctx = getContext();

    auto vpurtTask = perAxisTileDMAOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_UNLESS(vpurtTask != nullptr, "Can't get VPURT task operation");
    rewriter.setInsertionPointAfter(vpurtTask);

    auto inType = perAxisTileDMAOp.input().getType().cast<vpux::NDTypeInterface>();
    auto outType = perAxisTileDMAOp.output().getType().cast<vpux::NDTypeInterface>();
    Byte elemTypeSize = inType.getElemTypeSize();

    auto srcDeclBuff = perAxisTileDMAOp.input().getDefiningOp<VPURT::DeclareBufferOp>();
    auto dstDeclBuff = perAxisTileDMAOp.output_buff().getDefiningOp<VPURT::DeclareBufferOp>();
    auto srcType = srcDeclBuff.getType().cast<vpux::NDTypeInterface>();
    auto dstType = dstDeclBuff.getType().cast<vpux::NDTypeInterface>();

    auto createSubPerAxisTileDMAOp = [&](ShapeRef subInShape, ShapeRef subOutShape, int64_t srcOffset,
                                         int64_t dstOffset, VPUIP::DMADescriptorAttr dmaDescriptor, int64_t port) {
        const auto dimOrder = DimsOrder::CHW;
        auto getStrides = [](ShapeRef shape, Byte elemTypeSize) -> Strides {
            const auto strides = SmallVector<vpux::Bit>{Bit(shape[Dim(1)] * shape[Dim(2)] * Bit(elemTypeSize).count()),
                                                        Bit(shape.back() * Bit(elemTypeSize).count()),
                                                        Bit(Bit(elemTypeSize).count())};
            return Strides(strides);
        };

        auto newSrcMemRef = vpux::getMemRefType(subInShape, srcType.getElementType(), dimOrder, srcType.getMemSpace(),
                                                getStrides(subInShape, elemTypeSize));

        auto newSrcBuff =
                srcType.getMemSpace().getIndex().has_value()
                        ? VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, srcDeclBuff, vpurtTask.getLoc(),
                                                                  newSrcMemRef, srcDeclBuff.getSection(),
                                                                  srcType.getMemSpace().getIndex().value(), srcOffset)
                        : srcDeclBuff.getSectionIndex().has_value()
                                  ? VPURT::createOp<VPURT::DeclareBufferOp>(
                                            rewriter, srcDeclBuff, vpurtTask.getLoc(), newSrcMemRef,
                                            srcDeclBuff.getSection(),
                                            parseIntArrayAttr<int64_t>(srcDeclBuff.getSectionIndex().value()),
                                            srcOffset)
                                  : VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, srcDeclBuff, vpurtTask.getLoc(),
                                                                            newSrcMemRef, srcDeclBuff.getSection(),
                                                                            srcOffset);

        mlir::Type newDstType;
        if (auto dstDistributedType = dstType.dyn_cast<VPUIP::DistributedBufferType>()) {
            const auto distributionAttr = dstDistributedType.getDistribution();
            const auto layout = mlir::AffineMapAttr::get(dimOrder.toAffineMap(ctx));
            newDstType = VPUIP::DistributedBufferType::get(ctx, subOutShape.raw(), dstType.getElementType(), layout,
                                                           dstType.getMemSpace(), distributionAttr);
        } else {
            newDstType = vpux::getMemRefType(subOutShape, dstType.getElementType(), dimOrder, dstType.getMemSpace(),
                                             getStrides(subOutShape, elemTypeSize));
        }

        auto newDstBuff =
                dstType.getMemSpace().getIndex().has_value()
                        ? VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, dstDeclBuff, vpurtTask.getLoc(), newDstType,
                                                                  dstDeclBuff.getSection(),
                                                                  dstType.getMemSpace().getIndex().value(), dstOffset)
                        : dstDeclBuff.getSectionIndex().has_value()
                                  ? VPURT::createOp<VPURT::DeclareBufferOp>(
                                            rewriter, dstDeclBuff, vpurtTask.getLoc(), newDstType,
                                            dstDeclBuff.getSection(),
                                            parseIntArrayAttr<int64_t>(dstDeclBuff.getSectionIndex().value()),
                                            dstOffset)
                                  : VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, dstDeclBuff, vpurtTask.getLoc(),
                                                                            newDstType, dstDeclBuff.getSection(),
                                                                            dstOffset);

        _log.trace("Creating Sub-PerAxisTileDMAOp with inShape: {0} outShape: {1}, SrcMemory at {2}, DstMemory at {3}",
                   subInShape, subOutShape, newSrcBuff.getSection(), newDstBuff.getSection());

        auto newPerAxisTileDmaOp = VPURT::wrapIntoTaskOp<VPUIP::PerAxisTileDMAOp>(
                rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), vpurtTask.getLoc(), newSrcBuff,
                newDstBuff, vpux::getIntAttr(rewriter, port), perAxisTileDMAOp.channelTypeAttr(), nullptr, nullptr,
                dmaDescriptor, perAxisTileDMAOp.dma_hwp_idAttr());

        auto newVpurtTask = newPerAxisTileDmaOp->getParentOfType<VPURT::TaskOp>();
        if (auto cycleBeginAttr = vpurtTask->getAttr(cycleBegin)) {
            newVpurtTask->setAttr(cycleBegin, cycleBeginAttr);
        }
        if (auto cycleEndAttr = vpurtTask->getAttr(cycleEnd)) {
            newVpurtTask->setAttr(cycleEnd, cycleEndAttr);
        }
    };

    _log.trace("Unroll PerAxisTileDMAOp {0}", perAxisTileDMAOp->getLoc());

    auto axis = perAxisTileDMAOp.axis();
    auto tiles = perAxisTileDMAOp.tiles();
    VPUX_THROW_UNLESS(axis.has_value() && tiles.has_value(), "Cannot get PerAxisTile attribution");
    auto mergedShapes = VPUIP::getPerAxisTileDMAMergedShape(inType, outType, axis.value(), tiles.value());
    auto dmaDescriptorAttr = perAxisTileDMAOp.dma_descriptorAttr();
    auto portIsAlreadyAssigned = true;
    if (dmaDescriptorAttr == nullptr) {
        auto dmaDescriptorGenerator = VPUIP::PerAxisTileDmaDescriptorGenerator(ctx, _log);
        dmaDescriptorAttr = dmaDescriptorGenerator.generate(mergedShapes.first, mergedShapes.second, tiles.value(),
                                                            elemTypeSize.count());
        portIsAlreadyAssigned = false;
    }

    auto subInputShapes = VPUIP::getPerAxisTileDMASubShapes(mergedShapes.first);
    auto subOutputShapes = VPUIP::getPerAxisTileDMASubShapes(mergedShapes.second);
    VPUX_THROW_UNLESS(subInputShapes.size() == subOutputShapes.size(),
                      "Unexpected PerAxisTileDMA subInput '{0}' and subOutput '{1}' number", subInputShapes.size(),
                      subOutputShapes.size());

    auto srcOffset = srcDeclBuff.getByteOffset();
    auto dstOffset = dstDeclBuff.getByteOffset();
    for (size_t idx = 0; idx < subInputShapes.size(); idx++) {
        auto dmaPort = idx % _dmaPortCount;

        auto newDmaPort = portIsAlreadyAssigned ? perAxisTileDMAOp.port() : dmaPort;
        auto newDMADescriptorAttr = VPUIP::updateNumPlanes(dmaDescriptorAttr, subInputShapes[idx][Dim(0)]);
        createSubPerAxisTileDMAOp(subInputShapes[idx], subOutputShapes[idx], srcOffset, dstOffset, newDMADescriptorAttr,
                                  newDmaPort);

        srcOffset += subInputShapes[idx].totalSize() * elemTypeSize.count();
        dstOffset += subOutputShapes[idx].totalSize() * elemTypeSize.count();
    }

    rewriter.eraseOp(vpurtTask);
    return mlir::success();
}

mlir::LogicalResult PerAxisTileDMARewriter::unrollSegmentedOrOverlapped(VPUIP::PerAxisTileDMAOp perAxisTileDMAOp,
                                                                        VPUIP::DistributedBufferType distributedType,
                                                                        mlir::PatternRewriter& rewriter) const {
    auto loc = perAxisTileDMAOp->getLoc();
    auto ctx = perAxisTileDMAOp->getContext();

    const auto distributionAttr = distributedType.getDistribution();
    const auto numClusters = distributionAttr.getNumClusters().getInt();

    auto vpurtTask = perAxisTileDMAOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(vpurtTask == nullptr, "Can't get VPURT.TaskOp for {0}", perAxisTileDMAOp);
    rewriter.setInsertionPointAfter(vpurtTask);

    const auto input = perAxisTileDMAOp.input();
    const auto output = perAxisTileDMAOp.output_buff();
    const auto inputType = input.getType().cast<vpux::NDTypeInterface>();
    const auto outputType = distributedType.getCompactType();
    const auto numTiles = parseIntArrayAttr<int64_t>(distributionAttr.getNumTiles());
    const auto originInShape = inputType.getShape().raw();
    VPUX_THROW_UNLESS(originInShape.size() == numTiles.size(),
                      "Input shape size '{0}' and tiles array size '{1}' don't match", originInShape.size(),
                      numTiles.size());

    const auto perClusterShapes = distributedType.getPerClusterMemoryShapes();
    VPUX_THROW_UNLESS(perClusterShapes.size() == checked_cast<size_t>(numClusters),
                      "Number of shapes '{0}' and clusters '{1}' don't match", perClusterShapes.size(), numClusters);
    const auto perClusterShapeOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    VPUX_THROW_UNLESS(perClusterShapeOffsets.size() == checked_cast<size_t>(numClusters),
                      "Number of shape offsets '{0}' and clusters '{1}' don't match", perClusterShapeOffsets.size(),
                      numClusters);

    const auto isValidTile = [](auto dim) {
        return dim > 1;
    };

    const auto tilingAxis = std::distance(numTiles.begin(), llvm::find_if(numTiles, isValidTile));

    const auto getOperand = [&](int64_t clusterId, mlir::Value operand, vpux::NDTypeInterface newType,
                                mlir::Operation* insertionPoint) -> mlir::Value {
        if (auto cst = operand.getDefiningOp<Const::DeclareOp>()) {
            return rewriter.create<VPUIP::SubViewOp>(loc, cst, perClusterShapeOffsets[clusterId].raw(),
                                                     perClusterShapes[clusterId].raw());
        }

        auto declBuff = operand.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(declBuff != nullptr, "Can't find DeclareBuffer");

        if (newType.getMemoryKind() == VPU::MemoryKind::CMX_NN) {
            const auto cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
            const auto symbolAttr = vpux::IndexedSymbolAttr::get(ctx, {cmxNameAttr, vpux::getIntAttr(ctx, clusterId)});
            auto newCMXType = newType.changeMemSpace(symbolAttr);

            return VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, loc, newCMXType,
                                                           VPURT::BufferSection::CMX_NN,
                                                           getIntArrayAttr(ctx, makeArrayRef({clusterId})),
                                                           declBuff.getByteOffset(), declBuff.getSwizzlingKeyAttr());
        }

        Byte ddrOffset{declBuff.getByteOffset()};
        ddrOffset += perClusterShapeOffsets[clusterId][Dim(tilingAxis)] *
                     static_cast<Byte>(newType.getStrides()[Dim(tilingAxis)]);

        auto section = declBuff.getSection();
        auto sectionIndex = declBuff.getSectionIndex();

        const auto symbolAttr = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPURT::getMemoryKind(section)));
        newType = newType.changeMemSpace(symbolAttr);

        if (sectionIndex.has_value()) {
            return VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, loc, newType, section,
                                                           sectionIndex.value(), ddrOffset.count(), nullptr);
        }

        return VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, loc, newType, section,
                                                       ddrOffset.count());
    };

    auto padAxis = perAxisTileDMAOp.axis();
    auto padTiles = perAxisTileDMAOp.tiles();
    VPUX_THROW_UNLESS(padAxis.has_value() && padTiles.has_value(), "Cannot get PerAxisTile attribute");
    VPUX_THROW_UNLESS(padAxis.value() != tilingAxis,
                      "TilePerAxis expand axis '{0}' should not be the same as tiling axis '{1}'", padAxis.value(),
                      tilingAxis);

    auto elemTypeSize = Byte(inputType.getElemTypeSize());
    auto mergedShapes = VPUIP::getPerAxisTileDMAMergedShape(inputType, outputType, padAxis.value(), padTiles.value());
    auto dmaDescriptorGenerator = VPUIP::PerAxisTileDmaDescriptorGenerator(ctx, _log);
    auto dmaDescriptorAttr = dmaDescriptorGenerator.generate(mergedShapes.first, mergedShapes.second, padTiles.value(),
                                                             elemTypeSize.count());

    const auto tileType = [&](vpux::NDTypeInterface type) {
        SmallVector<vpux::NDTypeInterface> newTypes(numClusters);
        for (size_t clusterId = 0; clusterId < perClusterShapes.size(); ++clusterId) {
            newTypes[clusterId] =
                    changeShape(type, perClusterShapes[clusterId], perClusterShapeOffsets[clusterId], padAxis.value());
        }

        return newTypes;
    };

    const auto inTypes = tileType(inputType);
    const auto outTypes = tileType(outputType);
    auto inputInsertionPoint = input.getDefiningOp();
    auto outputInsertionPoint = output.getDefiningOp();
    SmallVector<VPUIP::PerAxisTileDMAOp> newOps;
    for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        const auto newInputType = inTypes[clusterId];
        const auto newOutType = outTypes[clusterId];

        const auto inputBuffer = getOperand(clusterId, input, newInputType, inputInsertionPoint);
        inputInsertionPoint = inputBuffer.getDefiningOp();
        _log.trace("Insert new input buffer declaration: '{0}'", inputBuffer);

        const auto outBuffer = getOperand(clusterId, output, newOutType, outputInsertionPoint);
        outputInsertionPoint = outBuffer.getDefiningOp();
        _log.trace("Insert new output buffer declaration: '{0}'", outBuffer);

        const auto newLoc = appendLoc(loc, "_cluster_{0}", clusterId);
        auto newDMAPort = clusterId % _dmaPortCount;
        auto newPerAxisTileDMAOp = VPURT::wrapIntoTaskOp<VPUIP::PerAxisTileDMAOp>(
                rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), newLoc, inputBuffer, outBuffer,
                vpux::getIntAttr(rewriter, newDMAPort), perAxisTileDMAOp.channelTypeAttr(), perAxisTileDMAOp.axisAttr(),
                perAxisTileDMAOp.tilesAttr(), dmaDescriptorAttr, perAxisTileDMAOp.dma_hwp_idAttr());

        _log.trace("Insert new PerAxisTile dma : '{0}'", newPerAxisTileDMAOp);

        auto newTaskOp = newPerAxisTileDMAOp->getParentOfType<VPURT::TaskOp>();
        if (auto cycleBeginAttr = vpurtTask->getAttr(cycleBegin)) {
            newTaskOp->setAttr(cycleBegin, cycleBeginAttr);
        }
        if (auto cycleEndAttr = vpurtTask->getAttr(cycleEnd)) {
            newTaskOp->setAttr(cycleEnd, cycleEndAttr);
        }

        newOps.push_back(newPerAxisTileDMAOp);
    }

    rewriter.eraseOp(vpurtTask);

    // unrolling per cluster tiling is done, now unroll per axis/tile
    for (const auto& op : newOps) {
        if (unrollPerAxisTile(op, rewriter).failed()) {
            return mlir::failure();
        }
    }
    return mlir::success();
}

mlir::LogicalResult PerAxisTileDMARewriter::unrollDuplicated(VPUIP::PerAxisTileDMAOp perAxisTileDMAOp,
                                                             VPUIP::DistributedBufferType distributedType,
                                                             mlir::PatternRewriter& rewriter) const {
    auto loc = perAxisTileDMAOp->getLoc();
    auto ctx = perAxisTileDMAOp->getContext();

    const auto input = perAxisTileDMAOp.input();
    const auto output = perAxisTileDMAOp.output_buff();

    const auto inputType = input.getType().cast<vpux::NDTypeInterface>();
    const auto outputType = output.getType().cast<vpux::NDTypeInterface>();

    const auto distributionAttr = distributedType.getDistribution();
    const auto numClusters = distributionAttr.getNumClusters().getInt();
    VPUX_THROW_WHEN(numClusters == 0, "Invalid number of clusters for {0}", distributedType);

    SmallVector<int64_t> clusters(numClusters);
    std::iota(clusters.begin(), clusters.end(), 0);

    auto vpurtTask = perAxisTileDMAOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(vpurtTask == nullptr, "Can't get VPURT.TaskOp for {0}", perAxisTileDMAOp);
    rewriter.setInsertionPointAfter(vpurtTask);

    auto outDeclBuff = output.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(outDeclBuff != nullptr, "Can't get output buffer");

    auto cmxBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
            rewriter, outDeclBuff, loc, outDeclBuff.getType(), VPURT::BufferSection::CMX_NN,
            getIntArrayAttr(ctx, clusters), outDeclBuff.getByteOffset(), outDeclBuff.getSwizzlingKeyAttr());

    _log.trace("Insert new CMX buffer declaration: '{0}'", cmxBuffer);

    auto axis = perAxisTileDMAOp.axis();
    auto tiles = perAxisTileDMAOp.tiles();
    VPUX_THROW_UNLESS(axis.has_value() && tiles.has_value(), "Cannot get PerAxisTile attributes");
    auto elemTypeSize = Byte(inputType.getElemTypeSize());

    auto mergedShapes = VPUIP::getPerAxisTileDMAMergedShape(inputType, outputType, axis.value(), tiles.value());
    auto dmaDescriptorGenerator = VPUIP::PerAxisTileDmaDescriptorGenerator(ctx, _log);
    auto dmaDescriptor = dmaDescriptorGenerator.generate(mergedShapes.first, mergedShapes.second, tiles.value(),
                                                         elemTypeSize.count());

    const auto newLoc = appendLoc(loc, "_broadcast_copy_to_CMX[{0},{1}]", clusters.front(), clusters.back());
    const auto newPerAxisTileDMA = VPURT::wrapIntoTaskOp<VPUIP::PerAxisTileDMAOp>(
            rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), newLoc, input, cmxBuffer,
            vpux::getIntAttr(rewriter, 0), perAxisTileDMAOp.channelTypeAttr(), perAxisTileDMAOp.axisAttr(),
            perAxisTileDMAOp.tilesAttr(), dmaDescriptor, perAxisTileDMAOp.dma_hwp_idAttr());

    _log.trace("Insert new PerAxisTileDMA op: '{0}'", newPerAxisTileDMA);

    auto newVpurtTask = newPerAxisTileDMA->getParentOfType<VPURT::TaskOp>();
    if (auto cycleBeginAttr = vpurtTask->getAttr(cycleBegin)) {
        newVpurtTask->setAttr(cycleBegin, cycleBeginAttr);
    }
    if (auto cycleEndAttr = vpurtTask->getAttr(cycleEnd)) {
        newVpurtTask->setAttr(cycleEnd, cycleEndAttr);
    }
    rewriter.eraseOp(vpurtTask);

    // unrolling per cluster tiling is done, now unroll per axis/tile
    return unrollPerAxisTile(newPerAxisTileDMA, rewriter);
}

//
// UnrollPerAxisTileDMAPass
//

class UnrollPerAxisTileDMAPass final : public VPUIP::UnrollPerAxisTileDMABase<UnrollPerAxisTileDMAPass> {
public:
    explicit UnrollPerAxisTileDMAPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollPerAxisTileDMAPass::safeRunOnFunc() {
    auto& ctx = getContext();

    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto dmaOp = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    auto dmaPortCount = dmaOp.count();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<PerAxisTileDMARewriter>(&ctx, dmaPortCount, _log);

    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUnrollPerAxisTileDMAPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUnrollPerAxisTileDMAPass(Logger log) {
    return std::make_unique<UnrollPerAxisTileDMAPass>(log);
}
