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

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include "vpux/compiler/dialect/IERT/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/range.hpp"

#include <llvm/ADT/SmallPtrSet.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <algorithm>

using namespace vpux;

namespace {

//
// AsyncRegionRewriter
//

bool isOptimizableOp(mlir::async::ExecuteOp execOp) {
    auto module = execOp->getParentOfType<mlir::ModuleOp>();

    uint32_t numUnits = 0;
    const auto executor = vpux::IERT::IERTDialect::getExecutor(execOp, numUnits);

    auto resOp = IERT::RunTimeResourcesOp::getFromModule(module);
    auto executorInfo = resOp.getExecutor(executor);

    return numUnits == executorInfo.count() && executorInfo.subExecutors().empty();
}

bool isSameExecutor(mlir::async::ExecuteOp execOp1, mlir::async::ExecuteOp execOp2) {
    uint32_t numUnits = 0;
    auto executor1 = vpux::IERT::IERTDialect::getExecutor(execOp1, numUnits);
    auto executor2 = vpux::IERT::IERTDialect::getExecutor(execOp2, numUnits);

    return executor1 == executor2;
}

mlir::async::ExecuteOp mergeAsyncExecuteOps(mlir::async::ExecuteOp prevExecOp, mlir::async::ExecuteOp execOp,
                                            mlir::PatternRewriter& rewriter) {
    auto* prevBodyBlock = &prevExecOp.body().front();
    auto* bodyBlock = &execOp.body().front();

    rewriter.setInsertionPointAfter(prevExecOp);

    const auto bodyBuilder = [prevBodyBlock, bodyBlock](mlir::OpBuilder& builder, mlir::Location loc,
                                                        mlir::ValueRange blockArgs) {
        SmallVector<mlir::Value> results;
        mlir::BlockAndValueMapping mapper;
        auto prevBlockArgs = prevBodyBlock->getArguments();
        auto curBlockArgs = bodyBlock->getArguments();

        for (size_t i = 0; i < blockArgs.size(); ++i) {
            if (i < prevBlockArgs.size()) {
                mapper.map(prevBlockArgs[i], blockArgs[i]);
            }
            if (i < curBlockArgs.size()) {
                mapper.map(curBlockArgs[i], blockArgs[prevBlockArgs.size() + i]);
            }
        }

        auto copyOps = [&results, &mapper, &builder](mlir::Block* bodyBlock) {
            for (auto& op : bodyBlock->getOperations()) {
                if (!mlir::isa<mlir::async::YieldOp>(op)) {
                    builder.clone(op, mapper);
                } else {
                    for (auto operand : op.getOperands()) {
                        results.push_back(mapper.lookupOrDefault(operand));
                    }
                }
            }
        };
        copyOps(prevBodyBlock);
        copyOps(bodyBlock);

        builder.create<mlir::async::YieldOp>(loc, results);
    };

    // joined results
    SmallVector<mlir::Type> results(prevBodyBlock->getTerminator()->getOperandTypes());
    results.insert(results.end(), bodyBlock->getTerminator()->getOperandTypes().begin(),
                   bodyBlock->getTerminator()->getOperandTypes().end());

    // joined dependencies
    SmallVector<mlir::Value> dependencies(prevExecOp.dependencies());
    dependencies.insert(dependencies.end(), execOp.dependencies().begin(), execOp.dependencies().end());

    // joined operands
    SmallVector<mlir::Value> operands(prevExecOp->getOperands());
    operands.insert(operands.end(), execOp->getOperands().begin(), execOp->getOperands().end());

    uint32_t numUnits = 0;
    auto executor = vpux::IERT::IERTDialect::getExecutor(execOp, numUnits);

    auto newExecOp =
            rewriter.create<mlir::async::ExecuteOp>(prevExecOp->getLoc(), results, dependencies, operands, bodyBuilder);

    IERT::IERTDialect::setExecutor(newExecOp, executor, numUnits);

    /* Each result of old operation must be replaced
       with corresponding result from new operation. */
    SmallVector<mlir::Value> matchedResultsPrev;
    SmallVector<mlir::Value> matchedResultsCurr;
    size_t prevResSize = prevExecOp->getResults().size();
    for (auto newResult : newExecOp->getResults()) {
        if (mlir::async::TokenType::classof(newResult.getType())) {
            // newExecOp returns one token which replaces tokes from old ops
            matchedResultsPrev.push_back(newResult);
            matchedResultsCurr.push_back(newResult);
            prevResSize--;
        } else {
            if (prevResSize > 0) {
                matchedResultsPrev.push_back(newResult);
                prevResSize--;
            } else {
                matchedResultsCurr.push_back(newResult);
            }
        }
    }

    prevExecOp.replaceAllUsesWith(matchedResultsPrev);
    prevExecOp->remove();
    execOp.replaceAllUsesWith(matchedResultsCurr);
    execOp->remove();

    return newExecOp;
}

void cleanupAwaitOps(mlir::async::ExecuteOp newExecOp) {
    auto awaitOp = mlir::dyn_cast_or_null<mlir::async::AwaitOp>(newExecOp->getNextNode());
    if (awaitOp == nullptr) {
        return;
    }
    auto nextAwaitOp = mlir::dyn_cast_or_null<mlir::async::AwaitOp>(awaitOp->getNextNode());
    if (nextAwaitOp == nullptr) {
        return;
    }
    if (awaitOp.operand() != nextAwaitOp.operand()) {
        return;
    }

    awaitOp.replaceAllUsesWith(nextAwaitOp);
    awaitOp->remove();
}

class GroupAsyncExecuteOps final : public mlir::OpRewritePattern<mlir::async::ExecuteOp> {
public:
    GroupAsyncExecuteOps(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<mlir::async::ExecuteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::async::ExecuteOp execOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult GroupAsyncExecuteOps::matchAndRewrite(mlir::async::ExecuteOp execOp,
                                                          mlir::PatternRewriter& rewriter) const {
    if (!isOptimizableOp(execOp)) {
        return mlir::failure();
    }

    auto prevWaitOp = mlir::dyn_cast_or_null<mlir::async::AwaitOp>(execOp->getPrevNode());
    if (prevWaitOp == nullptr) {
        return mlir::failure();
    }

    auto prevExecOp = mlir::dyn_cast_or_null<mlir::async::ExecuteOp>(prevWaitOp->getPrevNode());
    if (prevExecOp == nullptr) {
        return mlir::failure();
    }

    if (prevWaitOp.operand().getDefiningOp() != prevExecOp) {
        return mlir::failure();
    }

    if (!isOptimizableOp(prevExecOp) || !isSameExecutor(prevExecOp, execOp)) {
        return mlir::failure();
    }

    /*  TODO: Remove check below when proper operands mapping is implemented.
        If current execute op depends on previous then it requires more complex operands/results mapping.
        Mapping path:
        1st execute op yield op operands -> 1st execute op result->
        -> 2nd execute op operand-> internal block argument-> operation operand */
    if (std::find(prevExecOp->getUsers().begin(), prevExecOp->getUsers().end(), execOp) !=
        prevExecOp->getUsers().end()) {
        return mlir::failure();
    }

    mlir::async::ExecuteOp newExecOp = mergeAsyncExecuteOps(prevExecOp, execOp, rewriter);

    cleanupAwaitOps(newExecOp);

    return mlir::success();
}

//
// GroupAsyncExecuteOpsPass
//

class GroupAsyncExecuteOpsPass final : public IERT::GroupAsyncExecuteOpsBase<GroupAsyncExecuteOpsPass> {
public:
    explicit GroupAsyncExecuteOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void GroupAsyncExecuteOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<GroupAsyncExecuteOps>(&ctx, _log);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getFunction(), std::move(patterns),
                                                        getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createGroupAsyncExecuteOpsPass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createGroupAsyncExecuteOpsPass(Logger log) {
    return std::make_unique<GroupAsyncExecuteOpsPass>(log);
}
