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
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "vpux/compiler/dialect/IERT/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops_interfaces.hpp"

#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"

#include "vpux/compiler/utils/logging.hpp"

#include "vpux/utils/core/range.hpp"

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
    if (copyOp.output_buff().isa<mlir::BlockArgument>()) {
        return mlir::failure();
    }

    // Check if operation that defines the input is CopyOp to identify
    // CopyOp->CopyOp sequence
    auto parentCopyOp = copyOp.input().getDefiningOp<IERT::CopyOp>();
    if (parentCopyOp == nullptr) {
        return mlir::failure();
    }

    // TODO: Below is temporary limitation
    // Check if parentCopyOp source is NNCMX and if this result has more users then skip
    // optimization for now as in case of multiple branches leaving data in NNCMX
    // can cause buffer allocation issue. This could be removed later once
    // scheduling solution is more advance and there is support for dynamic spilling
    auto parentCopyOpSource = parentCopyOp.input();
    auto sourceMemory = VPUIP::getPhysicalMemory(parentCopyOpSource.getType().cast<mlir::MemRefType>());
    if (mlir::succeeded(sourceMemory) && sourceMemory.getValue() == VPUIP::PhysicalMemory::CMX_NN) {
        // Check for branches on ->CopOp->CopyOp-> sequence by counting
        // the number of tensor users. If more than 1 then there is a branch
        if (!parentCopyOpSource.hasOneUse() || !copyOp.input().hasOneUse() || !copyOp.output().hasOneUse()) {
            return mlir::failure();
        }
    }

    rewriter.replaceOpWithNewOp<IERT::CopyOp>(copyOp, parentCopyOp.input(), copyOp.output_buff());

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

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<CopyOpSequence>(&ctx, _log);

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
