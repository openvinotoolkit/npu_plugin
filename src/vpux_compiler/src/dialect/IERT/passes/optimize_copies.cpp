//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/IERT/passes.hpp"

#include "vpux/compiler/dialect/IERT/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"

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

class CopyOpSequence final : public mlir::OpRewritePattern<IERT::CopyOp> {
public:
    CopyOpSequence(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IERT::CopyOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::CopyOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult CopyOpSequence::matchAndRewrite(IERT::CopyOp copyOp, mlir::PatternRewriter& rewriter) const {
    // Check if operation that defines the input is CopyOp to identify
    // CopyOp->CopyOp sequence
    auto parentCopyOp = copyOp.input().getDefiningOp<IERT::CopyOp>();
    if (parentCopyOp == nullptr) {
        // Check current CopyOp source and destination
        const auto srcMemory = copyOp.input().getType().cast<vpux::NDTypeInterface>().getMemoryKind();
        const auto dstMemory = copyOp.output().getType().cast<vpux::NDTypeInterface>().getMemoryKind();

        // Remove redundant CMX2CMX CopyOps
        if (srcMemory == dstMemory && srcMemory == VPU::MemoryKind::CMX_NN) {
            // CMX Concat case with subView, update the buffers used
            auto copySubView = copyOp.output_buff().getDefiningOp<IERT::SubViewOp>();
            if (copySubView != nullptr) {
                // retrieve operations to be re-linked
                auto parentNCE = copyOp.input().getDefiningOp<VPUIP::NCEClusterTaskOp>();
                if (parentNCE == nullptr) {
                    return mlir::failure();
                }
                auto masterBuffer = copySubView.source().getDefiningOp<mlir::memref::AllocOp>();
                if (masterBuffer == nullptr) {
                    return mlir::failure();
                }
                // replace the copy with the subView
                copySubView->moveAfter(parentNCE.input().getDefiningOp());
                parentNCE.output().setType(copySubView.getType());
                parentNCE.output_buff().replaceAllUsesWith(copySubView);
                copyOp.output().replaceAllUsesWith(copyOp.input());

                // update IR location of the master buffer
                if (copySubView->isBeforeInBlock(masterBuffer)) {
                    masterBuffer->moveBefore(copySubView);
                }
            } else {
                // case with no subView
                copyOp.output().replaceAllUsesWith(copyOp.input());
            }
            rewriter.eraseOp(copyOp);
            return mlir::success();
        }

        return mlir::failure();
    }

    if (parentCopyOp.output_buff().isa<mlir::BlockArgument>() ||
        !isBufAllocOp(parentCopyOp.output_buff().getDefiningOp())) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IERT::CopyOp>(copyOp, parentCopyOp.input(), copyOp.output_buff());

    // CopyOp can have MemoryEffect so "hanging" unused parentCopyOp might not be erased by MLIR automatically
    if (parentCopyOp->use_empty()) {
        rewriter.eraseOp(parentCopyOp);
    }

    return mlir::success();
}

//
// fuseLastCopy
//

void fuseLastCopy(IERT::CopyOp copyOp, const AliasesInfo& aliasesInfo, Logger log) {
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

    if (sourceRoot.getType() != copyOp.output_buff().getType()) {
        // TODO: It is necessary to rearrange the operations of type casting
        return;
    }

    // Function outputs have to be an alias of the output buffer
    log.trace("Root of the copy operation input {0}", sourceRoot);
    log.trace("Reassign outputs from {0} to {1}", sourceRoot, copyOp.output_buff());

    for (auto& use : llvm::make_early_inc_range(sourceRoot.getUses())) {
        log.nest().trace("Got user {0}", use.getOwner()->getName());
        log.nest().trace("Reassign {0} to {1}", use.get(), copyOp.output_buff());
        use.set(copyOp.output_buff());
    }

    copyOp.replaceAllUsesWith(copyOp.input());
    copyOp->erase();
}

//
// OptimizeCopiesPass
//

class OptimizeCopiesPass final : public IERT::OptimizeCopiesBase<OptimizeCopiesPass> {
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

    auto func = getFunction();
    auto& aliasInfo = getAnalysis<AliasesInfo>();
    func->walk([&](IERT::CopyOp op) {
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

std::unique_ptr<mlir::Pass> vpux::IERT::createOptimizeCopiesPass(Logger log) {
    return std::make_unique<OptimizeCopiesPass>(log);
}
