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
        const auto srcMemory = VPU::getMemoryKind(copyOp.input().getType().cast<mlir::MemRefType>());
        const auto dstMemory = VPU::getMemoryKind(copyOp.output().getType().cast<mlir::MemRefType>());

        // Remove redundant CMX2CMX CopyOps
        if (srcMemory == dstMemory && srcMemory == VPU::MemoryKind::CMX_NN) {
            copyOp.output().replaceAllUsesWith(copyOp.input());
            rewriter.eraseOp(copyOp);
            return mlir::success();
        }

        return mlir::failure();
    }

    if (parentCopyOp.output_buff().isa<mlir::BlockArgument>() ||
        !isBufAllocOp(parentCopyOp.output_buff().getDefiningOp())) {
        return mlir::failure();
    }

    // retrieve weight tables using this buffer
    SmallVector<mlir::Operation*> wtUsingOutBuf;
    for (auto use : copyOp.output_buff().getUsers()) {
        if (mlir::isa<VPUIP::WeightsTableOp>(*use)) {
            wtUsingOutBuf.push_back(use);
        }
    }
    // update all weight tables using this buffer
    for (auto wt : wtUsingOutBuf) {
        wt->setOperand(0, parentCopyOp.output_buff());
    }

    rewriter.replaceOpWithNewOp<IERT::CopyOp>(copyOp, parentCopyOp.input(), copyOp.output_buff());

    return mlir::success();
}

//
// CopyToBlockArgument
//

class CopyToBlockArgument final : public mlir::OpRewritePattern<IERT::CopyOp> {
public:
    CopyToBlockArgument(mlir::MLIRContext* ctx, const AliasesInfo& aliasInfo, Logger log)
            : mlir::OpRewritePattern<IERT::CopyOp>(ctx), _aliasInfo(aliasInfo), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::CopyOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    const AliasesInfo& _aliasInfo;
    Logger _log;
};

mlir::LogicalResult CopyToBlockArgument::matchAndRewrite(IERT::CopyOp copyOp, mlir::PatternRewriter& rewriter) const {
    if (!copyOp.output_buff().isa<mlir::BlockArgument>()) {
        return mlir::failure();
    }

    auto inSourceMemory = VPU::getMemoryKind(copyOp.input().getType().cast<mlir::MemRefType>());
    auto outSourceMemory = VPU::getMemoryKind(copyOp.output().getType().cast<mlir::MemRefType>());
    if (inSourceMemory != outSourceMemory) {
        return mlir::failure();
    }

    auto sourceOp = copyOp.input().getDefiningOp();
    const auto sourceRoot = _aliasInfo.getRoot(copyOp.input());

    if (sourceOp == nullptr || sourceRoot.isa<mlir::BlockArgument>()) {
        // input also is block argument
        return mlir::failure();
    }

    if (!isBufAllocOp(sourceRoot.getDefiningOp())) {
        return mlir::failure();
    }

    if (sourceRoot.getType() != copyOp.output_buff().getType()) {
        // TODO: It is necessary to rearrange the operations of type casting
        return mlir::failure();
    }

    // Function outputs have to be an alias of the output buffer
    _log.trace("Root of the copy operation input {0}", sourceRoot);
    _log.trace("Reassign outputs from {0} to {1}", sourceRoot, copyOp.output_buff());

    for (auto& use : llvm::make_early_inc_range(sourceRoot.getUses())) {
        _log.nest().trace("Got user {0}", use.getOwner()->getName());
        _log.nest().trace("Reassign {0} to {1}", use.get(), copyOp.output_buff());
        use.set(copyOp.output_buff());
    }

    rewriter.replaceOp(copyOp, copyOp.input());
    return mlir::success();
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
    auto& aliasInfo = getAnalysis<AliasesInfo>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<CopyOpSequence>(&ctx, _log);
    patterns.insert<CopyToBlockArgument>(&ctx, aliasInfo, _log);

    auto func = getFunction();
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
