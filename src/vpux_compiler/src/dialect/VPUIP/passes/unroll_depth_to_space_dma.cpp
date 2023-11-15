//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/dma_descriptor_generator.hpp"
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

//
// DepthToSpaceDMARewriter
//

class DepthToSpaceDMARewriter final : public mlir::OpRewritePattern<VPUIP::DepthToSpaceDMAOp> {
public:
    DepthToSpaceDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount, Logger log)
            : mlir::OpRewritePattern<VPUIP::DepthToSpaceDMAOp>(ctx), _dmaPortCount(dmaPortCount), _log(log) {
        setDebugName("DepthToSpaceDMARewriter");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::DepthToSpaceDMAOp depthToSpaceDMAOp,
                                        mlir::PatternRewriter& rewriter) const final;

    mlir::LogicalResult matchAndRewriteClusterDMA(VPUIP::DepthToSpaceDMAOp depthToSpaceDMAOp,
                                                  mlir::PatternRewriter& rewriter) const;

private:
    int64_t _dmaPortCount;
    Logger _log;
};

mlir::LogicalResult DepthToSpaceDMARewriter::matchAndRewriteClusterDMA(VPUIP::DepthToSpaceDMAOp depthToSpaceDMAOp,
                                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("Got DepthToSpaceDMA with DistributedBuffer: {0}", depthToSpaceDMAOp->getLoc());

    auto ctx = depthToSpaceDMAOp->getContext();

    const auto blockSize = depthToSpaceDMAOp.block_size();
    auto paddedIC = 0;
    auto paddedOC = 0;

    if (depthToSpaceDMAOp.padded_channels().has_value()) {
        paddedIC = depthToSpaceDMAOp.padded_channels().value().getInput()
                           ? depthToSpaceDMAOp.padded_channels().value().getInput().getInt()
                           : 0;
        paddedOC = depthToSpaceDMAOp.padded_channels().value().getOutput()
                           ? depthToSpaceDMAOp.padded_channels().value().getOutput().getInt()
                           : 0;
    }

    const auto input = depthToSpaceDMAOp.input();
    const auto output = depthToSpaceDMAOp.output_buff();

    const auto distributedInputType = input.getType().dyn_cast<VPUIP::DistributedBufferType>();
    const auto distributedOutputType = output.getType().dyn_cast<VPUIP::DistributedBufferType>();

    VPUX_THROW_WHEN(distributedInputType == nullptr && distributedOutputType == nullptr,
                    "At least one of operands must have DistributedBuffer type");

    const auto getDistModeAttr = [&](VPUIP::DistributedBufferType distType) {
        const auto distAttr = distType.getDistribution();
        VPUX_THROW_WHEN(distAttr == nullptr, "Failed to extract distributon tensor from distributed type");
        return distAttr.getMode();
    };

    if (distributedInputType != nullptr) {
        const auto inputDistModeAttr = getDistModeAttr(distributedInputType);
        VPUX_THROW_UNLESS(
                inputDistModeAttr != nullptr && inputDistModeAttr.getValue() == VPU::DistributionMode::SEGMENTED,
                "Unsupported input distributed mode: {0}", inputDistModeAttr);
    }

    if (distributedOutputType != nullptr) {
        const auto outputDistModeAttr = getDistModeAttr(distributedOutputType);
        VPUX_THROW_UNLESS(
                outputDistModeAttr != nullptr && outputDistModeAttr.getValue() == VPU::DistributionMode::SEGMENTED,
                "Unsupported output distributed mode: {0}", outputDistModeAttr);
    }

    auto vpurtTask = depthToSpaceDMAOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(vpurtTask == nullptr, "Can not get VPURT.TaskOp for {0}", depthToSpaceDMAOp);
    auto cycleBeginAttr = vpurtTask->getAttr(cycleBegin);
    auto cycleEndAttr = vpurtTask->getAttr(cycleEnd);

    const auto inferOutputShape = [&](ShapeRef inShape) {
        auto outShape = Shape(inShape.raw());
        outShape[Dims4D::Act::H] *= blockSize;
        outShape[Dims4D::Act::W] *= blockSize;
        outShape[Dims4D::Act::C] = (outShape[Dims4D::Act::C] - paddedIC) / (blockSize * blockSize) + paddedOC;
        return outShape;
    };

    const auto loc = depthToSpaceDMAOp->getLoc();

    mlir::SmallVector<mlir::Value> inputBuffers;
    mlir::SmallVector<mlir::Value> outputBuffers;

    if (distributedInputType != nullptr && distributedOutputType != nullptr) {
        _log.nest().trace("Got multi-cluster to multi-clutser case");
        const auto inputPerClusterShapes = distributedInputType.getPerClusterMemoryShapes();
        const auto outputPerClusterShapes = distributedOutputType.getPerClusterMemoryShapes();

        const auto isShapeCompatible = [&](ShapeRef inShape, ShapeRef outShape) {
            return inShape == VPUIP::backInferD2SInputShape(outShape.toValues(), paddedOC, paddedIC, blockSize);
        };

        VPUX_THROW_UNLESS(llvm::all_of_zip(inputPerClusterShapes, outputPerClusterShapes, isShapeCompatible),
                          "Shape per cluster not compatible");

        const auto numClusters = checked_cast<int64_t>(inputPerClusterShapes.size());

        inputBuffers = VPUIP::getPerClusterMemoryBuffers(ctx, loc, "input", input, distributedInputType, numClusters,
                                                         rewriter);
        outputBuffers = VPUIP::getPerClusterMemoryBuffers(ctx, loc, "output", output, distributedOutputType,
                                                          numClusters, rewriter);
    }

    if (distributedInputType != nullptr && distributedOutputType == nullptr) {
        _log.nest().trace("Got multi-cluster to single-clutser case");
        const auto outputShapes = SmallVector<vpux::Shape>(
                llvm::map_range(distributedInputType.getPerClusterMemoryShapes(), inferOutputShape));
        const auto outputShapeOffsets = SmallVector<vpux::Shape>(
                llvm::map_range(distributedInputType.getPerClusterMemoryShapeOffsets(), inferOutputShape));

        const auto numClusters = checked_cast<int64_t>(outputShapes.size());

        inputBuffers = VPUIP::getPerClusterMemoryBuffers(ctx, loc, "input", input, distributedInputType, numClusters,
                                                         rewriter);
        outputBuffers = VPUIP::getSplitBuffers(ctx, loc, "output", output, outputShapes, outputShapeOffsets,
                                               numClusters, rewriter);
    }

    if (distributedInputType == nullptr && distributedOutputType != nullptr) {
        _log.nest().trace("Got single-cluster to multi-clutser case");

        const auto inputShapes = SmallVector<Shape>(
                llvm::map_range(distributedOutputType.getPerClusterMemoryShapes(), [&](ShapeRef outShape) {
                    return VPUIP::backInferD2SInputShape(outShape.toValues(), paddedOC, paddedIC, blockSize);
                }));

        const auto inputShapeOffsets = SmallVector<Shape>(
                llvm::map_range(distributedOutputType.getPerClusterMemoryShapeOffsets(), [&](ShapeRef outShape) {
                    return VPUIP::backInferD2SInputShape(outShape.toValues(), paddedOC, paddedIC, blockSize);
                }));

        const auto numClusters = checked_cast<int64_t>(inputShapes.size());
        inputBuffers =
                VPUIP::getSplitBuffers(ctx, loc, "input", input, inputShapes, inputShapeOffsets, numClusters, rewriter);
        outputBuffers = VPUIP::getPerClusterMemoryBuffers(ctx, loc, "output", output, distributedOutputType,
                                                          numClusters, rewriter);
    }

    VPUX_THROW_WHEN(inputBuffers.size() != outputBuffers.size(), "Size of input/output buffers list must match");
    const auto numClusters = inputBuffers.size();

    rewriter.setInsertionPointAfter(vpurtTask);

    int64_t dmaPort = 0;
    for (size_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        const auto newLoc = appendLoc(depthToSpaceDMAOp->getLoc(), "_cluster_{0}", clusterId);
        auto newDepthToSpaceDMAOp = VPURT::wrapIntoTaskOp<VPUIP::DepthToSpaceDMAOp>(
                rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), newLoc, inputBuffers[clusterId],
                outputBuffers[clusterId], vpux::getIntAttr(rewriter, dmaPort), depthToSpaceDMAOp.channelTypeAttr(),
                depthToSpaceDMAOp.block_sizeAttr(), depthToSpaceDMAOp.modeAttr(), nullptr,
                depthToSpaceDMAOp.padded_channelsAttr(), depthToSpaceDMAOp.dma_hwp_idAttr());

        dmaPort = (dmaPort + 1) % _dmaPortCount;

        _log.nest().trace("Insert new DepthToSpaceDMAOp: '{0}'", newDepthToSpaceDMAOp);
        auto newTaskOp = newDepthToSpaceDMAOp->getParentOfType<VPURT::TaskOp>();
        newTaskOp->setAttr(cycleBegin, cycleBeginAttr);
        newTaskOp->setAttr(cycleEnd, cycleEndAttr);
    }
    rewriter.eraseOp(vpurtTask);

    return mlir::success();
}

mlir::LogicalResult DepthToSpaceDMARewriter::matchAndRewrite(VPUIP::DepthToSpaceDMAOp depthToSpaceDMAOp,
                                                             mlir::PatternRewriter& rewriter) const {
    auto inputType = depthToSpaceDMAOp.input().getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto outputType = depthToSpaceDMAOp.output_buff().getType().dyn_cast<VPUIP::DistributedBufferType>();

    if (inputType != nullptr || outputType != nullptr) {
        return matchAndRewriteClusterDMA(depthToSpaceDMAOp, rewriter);
    }
    _log.trace("Got DepthToSpaceDMAOp: {0}", depthToSpaceDMAOp->getLoc());

    auto ctx = getContext();
    const auto inOrder = DimsOrder::fromValue(depthToSpaceDMAOp.input());
    const auto outOrder = DimsOrder::fromValue(depthToSpaceDMAOp.output_buff());

    auto vpurtTask = depthToSpaceDMAOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_UNLESS(vpurtTask != nullptr, "Can't get VPURT task operation");
    auto cycleBeginAttr = vpurtTask->getAttr(cycleBegin);
    auto cycleEndAttr = vpurtTask->getAttr(cycleEnd);
    rewriter.setInsertionPointAfter(vpurtTask);

    auto inType = depthToSpaceDMAOp.input().getType().cast<vpux::NDTypeInterface>();
    auto outType = depthToSpaceDMAOp.output().getType().cast<vpux::NDTypeInterface>();
    Byte elemTypeSize = inType.getElemTypeSize();

    if (depthToSpaceDMAOp.dma_descriptor().has_value()) {
        _log.nest().trace("This DepthToSpaceDMAOp has already been unrolled.");
        return mlir::failure();
    }

    const auto inputShape = getShape(depthToSpaceDMAOp.input());
    const auto outputShape = getShape(depthToSpaceDMAOp.output_buff());

    const auto inputC = inputShape[Dims4D::Act::C];
    const auto inputH = inputShape[Dims4D::Act::H];
    const auto inputW = inputShape[Dims4D::Act::W];
    const auto outputC = outputShape[Dims4D::Act::C];
    const auto outputW = outputShape[Dims4D::Act::W];
    auto blockSize = depthToSpaceDMAOp.block_size();
    auto mode = depthToSpaceDMAOp.mode();
    auto paddedIC =
            depthToSpaceDMAOp.padded_channels() ? depthToSpaceDMAOp.padded_channels().value().getInput() : nullptr;
    auto paddedOC =
            depthToSpaceDMAOp.padded_channels() ? depthToSpaceDMAOp.padded_channels().value().getOutput() : nullptr;

    auto srcDeclBuff = depthToSpaceDMAOp.input().getDefiningOp<VPURT::DeclareBufferOp>();
    auto dstDeclBuff = depthToSpaceDMAOp.output_buff().getDefiningOp<VPURT::DeclareBufferOp>();
    auto srcType = srcDeclBuff.getType().cast<vpux::NDTypeInterface>();
    auto dstType = dstDeclBuff.getType().cast<vpux::NDTypeInterface>();

    auto createSubDepthToSpaceDMAOp = [&](ShapeRef subShape, DimsOrder order, int64_t srcOffset, int64_t dstOffset,
                                          VPUIP::DMADescriptorAttr dmaDescriptor, int64_t port) {
        SmallVector<vpux::Bit> newStrides;
        const auto dataBitSize = Bit(elemTypeSize).count();
        if (order == DimsOrder::NHWC) {
            newStrides = SmallVector<vpux::Bit>{
                    Bit(subShape[Dims4D::Act::H] * subShape[Dims4D::Act::W] * subShape[Dims4D::Act::C] * dataBitSize),
                    Bit(dataBitSize), Bit(subShape[Dims4D::Act::W] * subShape[Dims4D::Act::C] * dataBitSize),
                    Bit(subShape[Dims4D::Act::C] * dataBitSize)};
        }

        auto newSrcMemRef = vpux::getMemRefType(subShape, srcType.getElementType(), inOrder, srcType.getMemSpace(),
                                                Strides(newStrides));

        auto newSrcBuff = VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, srcDeclBuff, vpurtTask.getLoc(),
                                                                  newSrcMemRef, srcDeclBuff.getSection(),
                                                                  srcType.getMemSpace().getIndex().value(), srcOffset);

        auto newDstMemRef = vpux::getMemRefType(subShape, dstType.getElementType(), outOrder, dstType.getMemSpace(),
                                                Strides(newStrides));

        auto newDstBuff = VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, dstDeclBuff, vpurtTask.getLoc(),
                                                                  newDstMemRef, dstDeclBuff.getSection(),
                                                                  dstType.getMemSpace().getIndex().value(), dstOffset);

        _log.nest().trace("Create Sub-DepthToSpaceDMAOp with shape: {0}, SrcMemory at {1}, DstMemory at {2}", subShape,
                          newSrcBuff.getSection(), newDstBuff.getSection());

        auto newDepthToSpaceDmaOp = VPURT::wrapIntoTaskOp<VPUIP::DepthToSpaceDMAOp>(
                rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), vpurtTask.getLoc(), newSrcBuff,
                newDstBuff, vpux::getIntAttr(rewriter, port), depthToSpaceDMAOp.channelTypeAttr(),
                depthToSpaceDMAOp.block_sizeAttr(), depthToSpaceDMAOp.modeAttr(), dmaDescriptor, nullptr,
                depthToSpaceDMAOp.dma_hwp_idAttr());

        auto newVpurtTask = newDepthToSpaceDmaOp->getParentOfType<VPURT::TaskOp>();
        if (cycleBeginAttr) {
            newVpurtTask->setAttr(cycleBegin, cycleBeginAttr);
        }
        if (cycleEndAttr) {
            newVpurtTask->setAttr(cycleEnd, cycleEndAttr);
        }
    };

    _log.nest().trace("Unroll DepthToSpaceDMAOp {0}", depthToSpaceDMAOp->getLoc());

    auto dmaDescriptorGenerator = VPUIP::DepthToSpaceDmaDescriptorGenerator(ctx, _log);

    // inputH is the planes number, need to split if it exceed the max number.
    auto numberOfTile = divUp(inputH, VPUIP::getMaxNumberPlanes(VPU::getArch(depthToSpaceDMAOp)));
    auto tileHSize = inputH / numberOfTile;

    for (int idx = 0; idx < numberOfTile; idx++) {
        int64_t subInputH;
        if (idx == (numberOfTile - 1)) {
            subInputH = inputH - tileHSize * idx;
        } else {
            subInputH = tileHSize;
        }
        auto lineSrcOffset = idx * tileHSize * inputC * inputW * elemTypeSize.count() + srcDeclBuff.getByteOffset();
        auto lineDstOffset =
                idx * tileHSize * blockSize * outputC * outputW * elemTypeSize.count() + dstDeclBuff.getByteOffset();
        Shape inputSubShape = Shape(SmallVector<int64_t>{inputShape[Dims4D::Act::N], inputC, subInputH, inputW});
        Shape outputSubShape =
                Shape(SmallVector<int64_t>{outputShape[Dims4D::Act::N], outputC, subInputH * blockSize, outputW});

        auto dmaDescriptor = dmaDescriptorGenerator.generate(inType, outType, inputSubShape, outputSubShape, mode,
                                                             blockSize, paddedIC, paddedOC);
        auto depthToSpaceIndex = 0;

        if (inOrder == DimsOrder::NHWC && mode == IE::DepthToSpaceMode::BLOCKS_FIRST) {
            auto blockShape =
                    Shape(SmallVector<int64_t>{inputShape[Dims4D::Act::N], inputC / blockSize, subInputH, inputW});
            auto srcOffset = lineSrcOffset;
            auto dstOffset = lineDstOffset;
            for (int bs = 0; bs < blockSize; bs++) {
                auto dmaPort = depthToSpaceIndex % _dmaPortCount;
                createSubDepthToSpaceDMAOp(blockShape, inOrder, srcOffset, dstOffset, dmaDescriptor, dmaPort);

                depthToSpaceIndex++;

                auto srcOffsetSize =
                        paddedOC != nullptr ? blockSize * (outputC - paddedOC.getInt()) : inputC / blockSize;
                srcOffset += srcOffsetSize * elemTypeSize.count();
                dstOffset += outputC * outputW * elemTypeSize.count();
            }
        } else if (inOrder == DimsOrder::NHWC && mode == IE::DepthToSpaceMode::DEPTH_FIRST) {
            auto blockShape = Shape(SmallVector<int64_t>{inputShape[Dims4D::Act::N], blockSize, subInputH, inputW});
            auto dstOffset = lineDstOffset;
            auto srcOffset = lineSrcOffset;
            for (int idx = 0; idx < inputC / blockSize; idx++) {
                auto dmaPort = depthToSpaceIndex % _dmaPortCount;
                createSubDepthToSpaceDMAOp(blockShape, inOrder, srcOffset, dstOffset, dmaDescriptor, dmaPort);

                srcOffset += blockSize * elemTypeSize.count();

                depthToSpaceIndex++;
                auto idxC = depthToSpaceIndex / blockSize;
                auto idxH = depthToSpaceIndex % blockSize;
                dstOffset =
                        lineDstOffset + idxC * elemTypeSize.count() + outputC * outputW * idxH * elemTypeSize.count();
            }
        } else {
            VPUX_THROW("Unsupport depth to space paramter order = {0}, mode = {1}", inOrder, mode);
        }
    }

    rewriter.eraseOp(vpurtTask);
    return mlir::success();
}

//
// UnrollDepthToSpaceDMAPass
//

class UnrollDepthToSpaceDMAPass final : public VPUIP::UnrollDepthToSpaceDMABase<UnrollDepthToSpaceDMAPass> {
public:
    explicit UnrollDepthToSpaceDMAPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollDepthToSpaceDMAPass::safeRunOnFunc() {
    auto& ctx = getContext();

    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto dmaOp = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    auto dmaPortCount = dmaOp.count();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<DepthToSpaceDMARewriter>(&ctx, dmaPortCount, _log);

    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUnrollDepthToSpaceDMAPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUnrollDepthToSpaceDMAPass(Logger log) {
    return std::make_unique<UnrollDepthToSpaceDMAPass>(log);
}
