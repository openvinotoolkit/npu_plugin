//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/permute_as_nndma_utils.hpp"
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
    int64_t _dmaPortCount;
    Logger _log;
};

mlir::LogicalResult PermuteRewriter::matchAndRewrite(VPUIP::PermuteDMAOp permuteOp,
                                                     mlir::PatternRewriter& rewriter) const {
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
    if (srcDeclBuff == nullptr) {
        srcDeclBuff = rewriter.create<VPURT::DeclareBufferOp>(vpurtTask.getLoc(), permuteOp.input().getType(),
                                                              VPURT::BufferSection::NetworkInput, 0);
    }
    auto dstDeclBuff = permuteOp.output_buff().getDefiningOp<VPURT::DeclareBufferOp>();

    auto inType = permuteOp.input().getType().cast<vpux::NDTypeInterface>();
    Byte elemTypeSize = inType.getElemTypeSize();

    auto srcType = srcDeclBuff.getType().cast<vpux::NDTypeInterface>();
    auto dstType = dstDeclBuff.getType().cast<vpux::NDTypeInterface>();
    auto srcOffset = srcDeclBuff.byteOffset();
    auto dstOffset = dstDeclBuff.byteOffset();

    auto subShapes = VPUIP::getPermuteDMASubShapes(permuteOp, _log);
    VPUX_THROW_UNLESS(subShapes.hasValue(), "Cannot get unrolled subshapes for PermuteDMA op {0}", permuteOp);

    _log.trace("Unrolling PermuteDMAOp '{0}' at '{1}'", permuteOp->getName(), permuteOp->getLoc());
    auto dstStride = VPUIP::getDstStride(subShapes.getValue());
    auto dstStrideAttr = getIntAttr(permuteOp.getContext(), dstStride);
    int64_t dmaPort = 0;
    for (auto subShape : subShapes.getValue()) {
        auto newStrides = SmallVector<vpux::Bit>{Bit(subShape[Dims4D::Act::C] * Bit(elemTypeSize).count()),
                                                 Bit(Bit(elemTypeSize).count())};

        auto newSrcMemRef = vpux::getMemRefType(subShape, srcType.getElementType(), DimsOrder::NC, Strides(newStrides),
                                                srcType.getMemSpace());
        auto newSrcBuff = VPURT::createOp<VPURT::DeclareBufferOp>(
                rewriter, srcDeclBuff, vpurtTask.getLoc(), newSrcMemRef, srcDeclBuff.section(),
                srcType.getMemSpace().getIndex().getValue(), srcOffset);

        auto newDstMemRef = vpux::getMemRefType(subShape, dstType.getElementType(), DimsOrder::NC, Strides(newStrides),
                                                dstType.getMemSpace());
        auto newDstBuff = VPURT::createOp<VPURT::DeclareBufferOp>(
                rewriter, dstDeclBuff, vpurtTask.getLoc(), newDstMemRef, dstDeclBuff.section(),
                dstType.getMemSpace().getIndex().getValue(), dstOffset);

        _log.trace("Create unrolled PermuteDMA operation with shape: {0}, SrcMemory at {1}, DstMemory at {2}", subShape,
                   newSrcBuff.section(), newDstBuff.section());

        const auto newLoc = appendLoc(vpurtTask->getLoc(), "_unrolled_permuteDMA");
        auto newPermuteDMAOp = VPURT::wrapIntoTaskOp<VPUIP::PermuteDMAOp>(
                rewriter, vpurtTask.waitBarriers(), vpurtTask.updateBarriers(), newLoc, newSrcBuff, newDstBuff,
                dstStrideAttr, nullptr, vpux::getIntAttr(rewriter, dmaPort));
        dmaPort = (dmaPort + 1) % _dmaPortCount;

        auto newVpurtTask = newPermuteDMAOp->getParentOfType<VPURT::TaskOp>();
        if (cycleBeginAttr) {
            newVpurtTask->setAttr(cycleBegin, cycleBeginAttr);
        }
        if (cycleEndAttr) {
            newVpurtTask->setAttr(cycleEnd, cycleEndAttr);
        }

        srcOffset += subShape.totalSize() * elemTypeSize.count();
        dstOffset += subShape[Dims4D::Act::N] * elemTypeSize.count();
    }

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
    patterns.insert<PermuteRewriter>(&ctx, dmaPortCount, _log.nest());
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
