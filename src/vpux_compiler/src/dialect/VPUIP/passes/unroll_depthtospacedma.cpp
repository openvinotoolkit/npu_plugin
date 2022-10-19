//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
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

private:
    int64_t _dmaPortCount;
    Logger _log;
};

mlir::LogicalResult DepthToSpaceDMARewriter::matchAndRewrite(VPUIP::DepthToSpaceDMAOp depthToSpaceDMAOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("Get DepthToSpaceDMAOp : {0}", depthToSpaceDMAOp);

    const auto inOrder = DimsOrder::fromValue(depthToSpaceDMAOp.input());
    const auto outOrder = DimsOrder::fromValue(depthToSpaceDMAOp.output_buff());

    auto vpurtTask = depthToSpaceDMAOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_UNLESS(vpurtTask != nullptr, "Can't get VPURT task operation");
    auto cycleBeginAttr = vpurtTask->getAttr(cycleBegin);
    auto cycleEndAttr = vpurtTask->getAttr(cycleEnd);
    rewriter.setInsertionPointAfter(vpurtTask);

    auto inType = depthToSpaceDMAOp.input().getType().cast<vpux::NDTypeInterface>();
    Byte elemTypeSize = inType.getElemTypeSize();

    const auto inputShape = getShape(depthToSpaceDMAOp.input());
    const auto outputShape = getShape(depthToSpaceDMAOp.output_buff());

    if (inputShape == outputShape) {
        _log.trace("This DepthToSpaceDMAOp has already been unrolled.");
        return mlir::failure();
    }

    const auto inputC = inputShape[Dims4D::Act::C];
    const auto inputH = inputShape[Dims4D::Act::H];
    const auto inputW = inputShape[Dims4D::Act::W];
    const auto outputC = outputShape[Dims4D::Act::C];
    const auto outputW = outputShape[Dims4D::Act::W];
    auto blockSize = depthToSpaceDMAOp.block_size();
    auto model = depthToSpaceDMAOp.mode();

    auto srcDeclBuff = depthToSpaceDMAOp.input().getDefiningOp<VPURT::DeclareBufferOp>();
    auto dstDeclBuff = depthToSpaceDMAOp.output_buff().getDefiningOp<VPURT::DeclareBufferOp>();
    auto srcType = srcDeclBuff.getType().cast<vpux::NDTypeInterface>();
    auto dstType = dstDeclBuff.getType().cast<vpux::NDTypeInterface>();

    auto srcOffset = srcDeclBuff.byteOffset();
    auto dstOffset = dstDeclBuff.byteOffset();

    auto createSubDepthToSpaceDMAOp = [&](ShapeRef subShape, DimsOrder order, int64_t srcOffset, int64_t dstOffset,
                                          int64_t port) {
        SmallVector<vpux::Bit> newStrides;
        const auto dataBitSize = Bit(elemTypeSize).count();
        if (order == DimsOrder::NHWC) {
            newStrides = SmallVector<vpux::Bit>{
                    Bit(subShape[Dims4D::Act::H] * subShape[Dims4D::Act::W] * subShape[Dims4D::Act::C] * dataBitSize),
                    Bit(dataBitSize), Bit(subShape[Dims4D::Act::W] * subShape[Dims4D::Act::C] * dataBitSize),
                    Bit(subShape[Dims4D::Act::C] * dataBitSize)};
        }

        auto newSrcMemRef = vpux::getMemRefType(subShape, srcType.getElementType(), inOrder, Strides(newStrides),
                                                srcType.getMemSpace());

        auto newSrcBuff = VPURT::createOp<VPURT::DeclareBufferOp>(
                rewriter, srcDeclBuff, vpurtTask.getLoc(), newSrcMemRef, srcDeclBuff.section(),
                srcType.getMemSpace().getIndex().getValue(), srcOffset);

        auto newDstMemRef = vpux::getMemRefType(subShape, dstType.getElementType(), outOrder, Strides(newStrides),
                                                dstType.getMemSpace());

        auto newDstBuff = VPURT::createOp<VPURT::DeclareBufferOp>(
                rewriter, dstDeclBuff, vpurtTask.getLoc(), newDstMemRef, dstDeclBuff.section(),
                dstType.getMemSpace().getIndex().getValue(), dstOffset);

        _log.trace("Create Sub-DepthToSpaceDMAOp with shape: {0}, SrcMemory at {1}, DstMemory at {2}", subShape,
                   newSrcBuff.section(), newDstBuff.section());

        auto newDepthToSpaceDmaOp = VPURT::wrapIntoTaskOp<VPUIP::DepthToSpaceDMAOp>(
                rewriter, vpurtTask.waitBarriers(), vpurtTask.updateBarriers(), vpurtTask.getLoc(), newSrcBuff,
                newDstBuff, depthToSpaceDMAOp.block_sizeAttr(), depthToSpaceDMAOp.modeAttr(),
                depthToSpaceDMAOp.output_channelAttr(), depthToSpaceDMAOp.output_widthAttr(),
                vpux::getIntAttr(rewriter, port));
        auto newVpurtTask = newDepthToSpaceDmaOp->getParentOfType<VPURT::TaskOp>();
        if (cycleBeginAttr) {
            newVpurtTask->setAttr(cycleBegin, cycleBeginAttr);
        }
        if (cycleEndAttr) {
            newVpurtTask->setAttr(cycleEnd, cycleEndAttr);
        }
    };

    _log.trace("Unroll DepthToSpaceDMAOp {0}", depthToSpaceDMAOp->getLoc());

    auto depthToSpaceIndex = 0;
    if (inOrder == DimsOrder::NHWC && model == IE::DepthToSpaceMode::BLOCKS_FIRST) {
        auto subShape = Shape(SmallVector<int64_t>{inputShape[Dims4D::Act::N], inputC / blockSize, inputH, inputW});
        for (int bs = 0; bs < blockSize; bs++) {
            auto dmaPort = depthToSpaceIndex % _dmaPortCount;
            createSubDepthToSpaceDMAOp(subShape, inOrder, srcOffset, dstOffset, dmaPort);

            depthToSpaceIndex++;
            srcOffset += (inputC / blockSize) * elemTypeSize.count();
            dstOffset += outputC * outputW * elemTypeSize.count();
        }
    }

    if (inOrder == DimsOrder::NHWC && model == IE::DepthToSpaceMode::DEPTH_FIRST) {
        auto subShape = Shape(SmallVector<int64_t>{inputShape[Dims4D::Act::N], outputC, blockSize, 1});
        for (int ih = 0; ih < inputH; ih++) {
            for (int ow = 0; ow < outputW; ow++) {
                auto dmaPort = depthToSpaceIndex % _dmaPortCount;
                createSubDepthToSpaceDMAOp(subShape, inOrder, srcOffset, dstOffset, dmaPort);

                depthToSpaceIndex++;
                if (depthToSpaceIndex % blockSize == 0) {
                    srcOffset += (inputC - blockSize + 1) * elemTypeSize.count();
                } else {
                    srcOffset += elemTypeSize.count();
                }
                dstOffset += outputC * elemTypeSize.count();
            }
            dstOffset += outputC * outputW * (blockSize - 1) * elemTypeSize.count();
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

    auto func = getFunction();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto dmaOp = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    auto dmaPortCount = dmaOp.count();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<DepthToSpaceDMARewriter>(&ctx, dmaPortCount, _log);

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
