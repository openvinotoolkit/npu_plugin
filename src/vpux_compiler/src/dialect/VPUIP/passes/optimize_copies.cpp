//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"

#include "vpux/compiler/core/aliases_info.hpp"

#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/range.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// CopyOpSequence
//

class CopyOpSequence final : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    CopyOpSequence(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUIP::CopyOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult CopyOpSequence::matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const {
    /*
     Remove redundant Copy-to-Copy sequence:
         ParentCopyOp
              |
           CopyOp
     */
    auto parentCopyOp = copyOp.input().getDefiningOp<VPUIP::CopyOp>();
    if (parentCopyOp == nullptr) {
        return mlir::failure();
    }

    if (parentCopyOp.output_buff().isa<mlir::BlockArgument>() ||
        !isBufAllocOp(parentCopyOp.output_buff().getDefiningOp())) {
        return mlir::failure();
    }

    for (auto user : parentCopyOp.output().getUsers()) {
        if (mlir::isa<VPUIP::SubViewOp>(user)) {
            // if intermediate SubViewOp users, skip due to accuracy loss
            // TODO E#35612: implement support for intermediate SubViewOp users
            return mlir::failure();
        }
    }

    rewriter.replaceOpWithNewOp<VPUIP::CopyOp>(copyOp, parentCopyOp.input(), copyOp.output_buff());

    // CopyOp can have MemoryEffect so "hanging" unused parentCopyOp might not be erased by MLIR automatically
    if (parentCopyOp->use_empty()) {
        rewriter.eraseOp(parentCopyOp);
    }

    return mlir::success();
}

//
// CMXToCMXCopy
//

mlir::LogicalResult removeClusterTilingCMXToCMXCopy(VPUIP::NCEClusterTilingOp copyClusterOp,
                                                    mlir::PatternRewriter& rewriter, Logger log) {
    auto innerCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(copyClusterOp.getInnerTaskOp());
    auto parentNCEClusterOp = copyClusterOp->getOperand(0).getDefiningOp<VPUIP::NCEClusterTilingOp>();
    if (innerCopyOp == nullptr || parentNCEClusterOp == nullptr) {
        return mlir::failure();
    }
    auto innerNCEOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(parentNCEClusterOp.getInnerTaskOp());
    if (innerNCEOp == nullptr) {
        return mlir::failure();
    }

    auto inputType = copyClusterOp->getOperand(0).getType().dyn_cast<vpux::NDTypeInterface>();
    auto outputType = copyClusterOp->getResult(0).getType().dyn_cast<vpux::NDTypeInterface>();
    if (inputType == nullptr || outputType == nullptr) {
        return mlir::failure();
    }
    // Check current CopyOp source and destination
    const auto srcMemory = inputType.getMemoryKind();
    const auto dstMemory = outputType.getMemoryKind();

    // Only remove redundant CMX2CMX CopyOps
    if (srcMemory != dstMemory || srcMemory != VPU::MemoryKind::CMX_NN) {
        return mlir::failure();
    }
    // CMX Concat case with subView, update the buffers used
    if (auto copySubView = copyClusterOp.output_buffs()[0].getDefiningOp<VPUIP::SubViewOp>()) {
        // case with subView - retrieve operations to be re-linked
        auto masterBuffer = copySubView.source().getDefiningOp<VPURT::AllocDistributed>();
        if (masterBuffer == nullptr) {
            return mlir::failure();
        }
        // replace the copy with the subView
        copySubView->moveBefore(parentNCEClusterOp);
        parentNCEClusterOp->getResult(0).setType(copySubView.getType());
        parentNCEClusterOp.output_buffs()[0].replaceAllUsesWith(copySubView);

        // After changing parentNCEClusterOp.output_buffs()[0] with result from copySubView
        // type will change. This change of type from operand needs to be populated to
        // inner argument so that corresponding type with NCEClusterTiling body is correct
        const auto newInnerType = copySubView.getType().dyn_cast<VPUIP::DistributedBufferType>().getCompactType();
        // Update type of argument that corresponds to output_buffs()[0]
        parentNCEClusterOp.body().getArgument(parentNCEClusterOp.getNumOperands() - 1).setType(newInnerType);
        // If output_buffs()[0] has changed then type of result of inner task needs
        // to be updated as well
        parentNCEClusterOp.getInnerTaskOp()->getResult(0).setType(newInnerType);

        copyClusterOp->replaceAllUsesWith(parentNCEClusterOp);

        // update IR location of the master buffer
        if (copySubView->isBeforeInBlock(masterBuffer)) {
            masterBuffer->moveBefore(copySubView);
        }
    } else if (inputType == outputType) {
        // case with no subView
        copyClusterOp->replaceAllUsesWith(parentNCEClusterOp);
    } else {
        log.trace("Copy not optimized {0}", copyClusterOp->getLoc());
        return mlir::failure();
    }
    rewriter.eraseOp(copyClusterOp);
    return mlir::success();
}

mlir::LogicalResult removeCMXToCMXCopy(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter, Logger log) {
    // Check current CopyOp source and destination
    const auto srcMemory = copyOp.input().getType().cast<vpux::NDTypeInterface>().getMemoryKind();
    const auto dstMemory = copyOp.output().getType().cast<vpux::NDTypeInterface>().getMemoryKind();

    // Only remove redundant CMX2CMX CopyOps
    if (srcMemory != dstMemory || srcMemory != VPU::MemoryKind::CMX_NN) {
        return mlir::failure();
    }
    // CMX Concat case with subView, update the buffers used
    auto copyBufferOp = copyOp.output_buff().getDefiningOp();
    if (mlir::isa<VPUIP::SubViewOp>(copyBufferOp)) {
        // case with subView - retrieve operations to be re-linked
        auto parentNCE = copyOp.input().getDefiningOp<VPUIP::NCEClusterTaskOp>();
        if (parentNCE == nullptr) {
            return mlir::failure();
        }
        auto masterBuffer = copyBufferOp->getOperand(0).getDefiningOp<mlir::memref::AllocOp>();
        if (masterBuffer == nullptr) {
            return mlir::failure();
        }
        // replace the copy with the subView
        copyBufferOp->moveBefore(parentNCE);
        parentNCE->getResult(0).setType(copyBufferOp->getResult(0).getType());
        parentNCE.output_buff().replaceAllUsesWith(copyBufferOp->getResult(0));
        copyOp.output().replaceAllUsesWith(copyOp.input());

        // update IR location of the master buffer
        if (copyBufferOp->isBeforeInBlock(masterBuffer)) {
            masterBuffer->moveBefore(copyBufferOp);
        }
    } else if (copyOp.input().getType() == copyOp.output().getType()) {
        // case with no subView
        copyOp.output().replaceAllUsesWith(copyOp.input());
    } else {
        log.trace("Copy not optimized {0}", copyOp->getLoc());
        return mlir::failure();
    }
    rewriter.eraseOp(copyOp);
    return mlir::success();
}

class CMXToCMXCopy final : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    CMXToCMXCopy(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUIP::CopyOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult CMXToCMXCopy::matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const {
    /*
     Remove CMX2CMX Copy without SubView:
         Copy(DDR2CMX)                    Copy(DDR2CMX)
              |                                |
            NCEOp           =>               NCEOp
              |
         Copy(CMX2CMX)

     Remove CMX2CMX Copy with SubView:
        Copy(DDR2CMX)                Copy(DDR2CMX)  SubView
              |                                \     /
            NCEOp       SubView   =>            NCEOp
               \         /
              Copy(CMX2CMX)
     */
    if (auto clusterTilingCopy = copyOp->getParentOfType<VPUIP::NCEClusterTilingOp>()) {
        return removeClusterTilingCMXToCMXCopy(clusterTilingCopy, rewriter, _log);
    } else {
        return removeCMXToCMXCopy(copyOp, rewriter, _log);
    }
}

//
// DDRToDDRCopyOfNCECluster
//

class DDRToDDRCopyOfNCECluster final : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    DDRToDDRCopyOfNCECluster(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool isCopyToDDR(VPUIP::CopyOp copyOp) {
    auto origOp = copyOp->getParentOfType<VPUIP::NCEClusterTilingOp>() == nullptr ? copyOp.getOperation()
                                                                                  : copyOp->getParentOp();
    return origOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getMemoryKind() == VPU::MemoryKind::DDR;
}

bool isCopyFromDDR(VPUIP::CopyOp copyOp) {
    auto origOp = copyOp->getParentOfType<VPUIP::NCEClusterTilingOp>() == nullptr ? copyOp.getOperation()
                                                                                  : copyOp->getParentOp();
    return origOp->getOperand(0).getType().cast<vpux::NDTypeInterface>().getMemoryKind() == VPU::MemoryKind::DDR;
}

bool isDDR2DDROfNCEClusterInput(VPUIP::CopyOp copyOp) {
    // ParentOp should be a Subview
    // ChildOp should be a copy op wrapped in ClusterTilingOp
    auto parentOp = copyOp.input().getDefiningOp<VPUIP::SubViewOp>();
    if (parentOp == nullptr) {
        return false;
    }
    if (copyOp.output().getUsers().empty()) {
        return false;
    }
    auto childOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(*copyOp.output().getUsers().begin());
    return (childOp != nullptr) && (childOp.getInnerTaskOpOfType<VPUIP::CopyOp>() != nullptr);
}

bool hasDiffSubviewForSiblingCopy(VPUIP::CopyOp copyOp, VPUIP::NCEClusterTilingOp parentOp) {
    auto subview = copyOp.output_buff().getDefiningOp<VPUIP::SubViewOp>();
    for (auto siblingOp : parentOp.getResult(0).getUsers()) {
        auto siblingCopy = mlir::dyn_cast<VPUIP::CopyOp>(*siblingOp);
        if (siblingCopy != nullptr && siblingCopy != copyOp) {
            auto siblingSubview = siblingCopy.output_buff().getDefiningOp<VPUIP::SubViewOp>();
            if (siblingSubview.static_offsets() != subview.static_offsets() ||
                siblingSubview.static_sizes() != subview.static_sizes() ||
                siblingSubview.static_strides() != subview.static_strides()) {
                return true;
            }
        }
    }
    return false;
}

bool isDDR2DDROfNCEClusterOutput(VPUIP::CopyOp copyOp) {
    // ParentOp should be a copy op wrapped in ClusterTilingOp
    // ChildOp should be a concat
    auto parentOp = copyOp->getOperand(0).getDefiningOp<VPUIP::NCEClusterTilingOp>();
    if (parentOp == nullptr || parentOp.getInnerTaskOpOfType<VPUIP::CopyOp>() == nullptr) {
        return false;
    }
    if (copyOp.output().getUsers().empty()) {
        return false;
    }
    for (auto user : copyOp.output().getUsers()) {
        if (!mlir::isa<VPUIP::ConcatViewOp>(*user)) {
            return false;
        }
    }

    /*
     Considering below case, the two DDR2DDR copy cann't be removed directly if
     they have different subview attributes.
                 ClusterTiling_Copy(CMX2DDR)
                      /        \
            Copy(DDR2DDR)       Copy(DDR2DDR)
            /        \          /       \
        SubView           |            SubView
                          |
                        Concat
    */
    return !hasDiffSubviewForSiblingCopy(copyOp, parentOp);
}

bool isParallelDDR2DDROfNCEClusterOutput(VPUIP::CopyOp copyOp) {
    // ParentOp should be a copy op wrapped in ClusterTilingOp
    // ChildOp should be a concat
    auto parentOp = copyOp->getOperand(0).getDefiningOp<VPUIP::NCEClusterTilingOp>();
    if (parentOp == nullptr || parentOp.getInnerTaskOpOfType<VPUIP::CopyOp>() == nullptr) {
        return false;
    }

    auto clusterTilingOp = parentOp->getOperand(0).getDefiningOp<VPUIP::NCEClusterTilingOp>();
    if (clusterTilingOp == nullptr) {
        return false;
    }

    if (copyOp.output().getUsers().empty()) {
        return false;
    }
    for (auto user : copyOp.output().getUsers()) {
        if (!mlir::isa<VPUIP::ConcatViewOp>(*user)) {
            return false;
        }
    }

    for (auto user : parentOp.getResult(0).getUsers()) {
        if (!mlir::isa<VPUIP::CopyOp>(*user)) {
            return false;
        }
    }
    /*
     Optimize the parallel DDR2DDR copies as CMX2DDR copies:
                 ClusterTiling_Copy(CMX2DDR)
                      /        \
            Copy(DDR2DDR)       Copy(DDR2DDR)
            /        \          /       \
        SubView           |            SubView
                          |
                        Concat
    */
    return hasDiffSubviewForSiblingCopy(copyOp, parentOp);
}

mlir::LogicalResult removeDDR2DDRForNCEClusterInput(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter, Logger log) {
    copyOp.output().replaceAllUsesWith(copyOp.input());
    log.trace("Removed DDRToDDR input copy {0} at {1}", copyOp->getName(), copyOp->getLoc());
    rewriter.eraseOp(copyOp);
    return mlir::success();
}

mlir::LogicalResult removeDDR2DDRForNCEClusterOutput(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter,
                                                     Logger log) {
    // CMX Concat case with subView, update the buffers used
    if (auto subViewOp = copyOp.output_buff().getDefiningOp<VPUIP::SubViewOp>()) {
        // case with subView - retrieve operations to be re-linked
        auto masterBuffer = subViewOp->getOperand(0).getDefiningOp<mlir::memref::AllocOp>();
        if (masterBuffer == nullptr) {
            return mlir::failure();
        }
        auto parentOp = copyOp->getOperand(0).getDefiningOp<VPUIP::NCEClusterTilingOp>();
        // replace the copy with VPUIP subView
        rewriter.setInsertionPoint(parentOp);
        auto newSubViewOp = rewriter.create<VPUIP::SubViewOp>(
                subViewOp->getLoc(), subViewOp.source(), subViewOp.static_offsetsAttr(), subViewOp.static_sizesAttr());
        parentOp.output_buffs()[0].replaceAllUsesWith(newSubViewOp->getResult(0));
        parentOp->getResult(0).setType(newSubViewOp->getResult(0).getType());

        // update IR location of the master buffer
        if (newSubViewOp->isBeforeInBlock(masterBuffer)) {
            masterBuffer->moveBefore(newSubViewOp);
        }
    } else {
        auto parentOp = copyOp.input().getDefiningOp<VPUIP::NCEClusterTilingOp>();
        auto allocOp = parentOp.output_buffs()[0].getDefiningOp<mlir::memref::AllocOp>();
        if (allocOp == nullptr) {
            return mlir::failure();
        }

        for (auto user : copyOp.output().getUsers()) {
            auto concatOp = mlir::dyn_cast<VPUIP::ConcatViewOp>(user);
            concatOp.output_buff().replaceAllUsesWith(allocOp);
        }
    }

    copyOp.output().replaceAllUsesWith(copyOp.input());
    log.trace("Removed DDRToDDR output copy {0} at {1}", copyOp->getName(), copyOp->getLoc());
    rewriter.eraseOp(copyOp);
    return mlir::success();
}

mlir::LogicalResult removeParallelDDR2DDRForNCEClusterOutput(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter,
                                                             Logger log) {
    auto parentOp = copyOp->getOperand(0).getDefiningOp<VPUIP::NCEClusterTilingOp>();
    auto clusterTilingOp = parentOp->getOperand(0).getDefiningOp<VPUIP::NCEClusterTilingOp>();

    for (auto user : llvm::make_early_inc_range(parentOp.getResult(0).getUsers())) {
        if (auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(*user)) {
            auto subview = copyOp.output_buff().getDefiningOp<VPUIP::SubViewOp>();

            rewriter.setInsertionPointAfter(subview);
            const auto copyOutBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                                mlir::ValueRange newOperands) {
                builder.create<VPUIP::CopyOp>(loc, newOperands[0], newOperands[1]);
            };
            SmallVector<mlir::Value> inputsOutputOperands = {clusterTilingOp->getResult(0), subview.result()};
            auto newCopyInCluster = rewriter.create<VPUIP::NCEClusterTilingOp>(
                    clusterTilingOp->getLoc(), subview->getResult(0).getType(), inputsOutputOperands,
                    copyOutBodyBuilder);

            copyOp.output().replaceAllUsesWith(newCopyInCluster->getResult(0));

            log.trace("Removed Parallel DDRToDDR output copy {0} at {1}", copyOp->getName(), copyOp->getLoc());
            rewriter.eraseOp(copyOp);
        }
    }

    rewriter.eraseOp(parentOp);
    return mlir::success();
}

mlir::LogicalResult DDRToDDRCopyOfNCECluster::matchAndRewrite(VPUIP::CopyOp copyOp,
                                                              mlir::PatternRewriter& rewriter) const {
    /*
     Remove redundant DDR2DDR Copy of the NCECluster's input:
ClusterTiling_Copy                    ...        SubView
   (CMX2DDR)        SubView             \         /
          \         /              ClusterTiling_Copy(CMX2DDR)
          Copy(DDR2DDR)        =>            |
               |                           Concat
            Concat

     Remove redundant DDR2DDR Copy of the NCECluster's output:
             SubView                    SubView
              (DDR)                      (DDR)
                |                          |
          Copy(DDR2DDR)            ClusterTiling_Copy
                |                      (DDR2CMX)
        ClusterTiling_Copy    =>           |
            (DDR2CMX)              ClusterTiling_NCE
                |                          |
        ClusterTiling_NCE
                |

     Optimize the parallel DDR2DDR copies as CMX2DDR copies:
                ClusterTiling_Copy(CMX2DDR)
                      /        \
            Copy(DDR2DDR)       Copy(DDR2DDR)       =>
            /        \          /       \
        SubView           |            SubView
                          |
                        Concat

                         ...
                     /          \
ClusterTiling_Copy(CMX2DDR)   ClusterTiling_Copy(CMX2DDR)
            /        \          /       \
        SubView           |            SubView
                          |
                        Concat
     */
    _log.trace("Copy {0} at {1}", copyOp->getName(), copyOp->getLoc());
    if (!isCopyFromDDR(copyOp) || !isCopyToDDR(copyOp)) {
        return mlir::failure();
    }
    if (isDDR2DDROfNCEClusterInput(copyOp)) {
        return removeDDR2DDRForNCEClusterInput(copyOp, rewriter, _log);
    } else if (isDDR2DDROfNCEClusterOutput(copyOp)) {
        return removeDDR2DDRForNCEClusterOutput(copyOp, rewriter, _log);
    } else if (isParallelDDR2DDROfNCEClusterOutput(copyOp)) {
        // TODO: Add this optimization in single cluster case
        return removeParallelDDR2DDRForNCEClusterOutput(copyOp, rewriter, _log);
    }

    return mlir::failure();
}

//
// fuseLastCopy
//

void fuseLastCopy(VPUIP::CopyOp copyOp, const AliasesInfo& aliasesInfo, Logger log) {
    if (!copyOp.output_buff().isa<mlir::BlockArgument>()) {
        return;
    }

    auto inSourceMemory = copyOp.input().getType().cast<vpux::NDTypeInterface>().getMemoryKind();
    auto outSourceMemory = copyOp.output().getType().cast<vpux::NDTypeInterface>().getMemoryKind();
    if (inSourceMemory != outSourceMemory) {
        return;
    }

    auto sourceOp = copyOp.input().getDefiningOp();
    if (sourceOp == nullptr) {
        // input also is block argument
        return;
    }

    const auto sourceRoots = aliasesInfo.getRoots(copyOp.input());
    if (sourceRoots.size() != 1) {
        return;
    }

    const auto sourceRoot = *sourceRoots.begin();
    if (sourceRoot == nullptr || sourceRoot.isa<mlir::BlockArgument>()) {
        // input also is block argument
        return;
    }

    if (!isBufAllocOp(sourceRoot.getDefiningOp())) {
        // input is constant, for example
        return;
    }
    VPUIP::ConcatViewOp concatViewOp;
    auto newBuffer = copyOp.output_buff();
    auto newOutput = copyOp.input();

    if (sourceRoot.getType() != copyOp.output_buff().getType()) {
        // we will make a QuantizeCast over the output buffer and we will copy from CMX directly to output buffer,
        // and we will return the output buffer. After ConcatView and QuantizeCast will be redundant.
        // from CMX -> CopyOp[DDR] -> ConcatViewOp -> QuantizeCastOp -> CopyOp[block-arg] -> return CopyOp
        // Output of this step:
        //                        CMX -> CopyOp[QuantizeCastOp] -> return block-arg
        //   block-arg -> QuantizeCastOp /

        auto quantizeCastOp = mlir::dyn_cast<VPUIP::QuantizeCastOp>(copyOp.input().getDefiningOp());
        if (!quantizeCastOp) {
            return;
        }

        concatViewOp = mlir::dyn_cast<VPUIP::ConcatViewOp>(quantizeCastOp.input().getDefiningOp());
        if (!concatViewOp) {
            return;
        }

        mlir::OpBuilder builder(quantizeCastOp);
        builder.setInsertionPoint(sourceRoot.getDefiningOp());

        auto newQuantizeCast = builder.create<VPUIP::QuantizeCastOp>(concatViewOp.getLoc(), sourceRoot.getType(),
                                                                     copyOp.output_buff());

        quantizeCastOp.replaceAllUsesWith(quantizeCastOp.input());
        quantizeCastOp->erase();

        newBuffer = newQuantizeCast.output();
        newOutput = copyOp.output_buff();
    }

    // Function outputs have to be an alias of the output buffer
    log.trace("Root of the copy operation input {0}", sourceRoot);
    log.trace("Reassign outputs from {0} to {1}", sourceRoot, newBuffer);

    for (auto& use : llvm::make_early_inc_range(sourceRoot.getUses())) {
        log.nest().trace("Got user {0}", use.getOwner()->getName());
        log.nest().trace("Reassign {0} to {1}", use.get(), newBuffer);
        use.set(newBuffer);
    }

    copyOp.replaceAllUsesWith(newOutput);
    copyOp->erase();
    if (concatViewOp) {
        concatViewOp->erase();
    }
}

//
// OptimizeCopiesPass
//

class OptimizeCopiesPass final : public VPUIP::OptimizeCopiesBase<OptimizeCopiesPass> {
public:
    explicit OptimizeCopiesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void OptimizeCopiesPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<CopyOpSequence>(&ctx, _log);
    patterns.insert<CMXToCMXCopy>(&ctx, _log);
    patterns.insert<DDRToDDRCopyOfNCECluster>(&ctx, _log);

    auto func = getFunction();
    auto& aliasInfo = getAnalysis<AliasesInfo>();
    func->walk([&](VPUIP::CopyOp op) {
        fuseLastCopy(op, aliasInfo, _log);
    });

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeCopiesPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createOptimizeCopiesPass(Logger log) {
    return std::make_unique<OptimizeCopiesPass>(log);
}
