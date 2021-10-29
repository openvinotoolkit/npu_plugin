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
#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops_interfaces.hpp"
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
