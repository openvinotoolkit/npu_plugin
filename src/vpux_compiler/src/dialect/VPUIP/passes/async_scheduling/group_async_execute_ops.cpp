//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/SmallPtrSet.h>

#include <algorithm>

using namespace vpux;

namespace {

//
// Helpers
//

bool isOptimizableOp(mlir::async::ExecuteOp execOp) {
    auto module = execOp->getParentOfType<mlir::ModuleOp>();

    const auto executor = vpux::VPUIP::VPUIPDialect::getExecutor(execOp);

    auto executorInfo = IE::getAvailableExecutor(module, executor.getFullReference());
    VPUX_THROW_UNLESS(executorInfo != nullptr, "Failed to get information about executor {0}", executor);

    return executorInfo.subExecutors().front().empty();
}

bool isSameExecutor(mlir::async::ExecuteOp execOp1, mlir::async::ExecuteOp execOp2) {
    auto executor1 = vpux::VPUIP::VPUIPDialect::getExecutor(execOp1);
    auto executor2 = vpux::VPUIP::VPUIPDialect::getExecutor(execOp2);
    return executor1 == executor2;
}

bool haveSameDependencies(mlir::async::ExecuteOp execOp1, mlir::async::ExecuteOp execOp2) {
    // check for prefetched data operations which have an injected dependency
    if (execOp1.dependencies().size() == 1 && execOp2.dependencies().size() == 1) {
        llvm::DenseSet<mlir::Value> dependencies;
        for (auto dep : execOp1.dependencies()) {
            dependencies.insert(dep);
        }
        for (auto dep : execOp2.dependencies()) {
            if (dependencies.find(dep) == dependencies.end()) {
                return false;
            }
        }
    }
    return true;
}

bool prevHasUniqueUsers(mlir::async::ExecuteOp prevExecOp, mlir::async::ExecuteOp execOp) {
    auto getUsers = [](mlir::async::ExecuteOp op) {
        std::set<mlir::async::ExecuteOp> users;
        for (auto res : op.results()) {
            for (auto user : res.getUsers()) {
                users.insert(mlir::dyn_cast<mlir::async::ExecuteOp>(user));
            }
        }
        return users;
    };

    std::set<mlir::async::ExecuteOp> usersPrevOp = getUsers(prevExecOp);
    std::set<mlir::async::ExecuteOp> usersOp = getUsers(execOp);
    usersPrevOp.erase(execOp);

    SmallVector<mlir::async::ExecuteOp> uniqueUsersPrevOp;
    std::set_difference(usersPrevOp.begin(), usersPrevOp.end(), usersOp.begin(), usersOp.end(),
                        std::back_inserter(uniqueUsersPrevOp));
    if (!uniqueUsersPrevOp.empty())
        return true;

    return false;
}

//
// mergeAsyncExecuteOps
//

mlir::async::ExecuteOp mergeAsyncExecuteOps(mlir::async::ExecuteOp prevExecOp, mlir::async::ExecuteOp execOp,
                                            mlir::PatternRewriter& rewriter) {
    auto* prevBodyBlock = &prevExecOp.body().front();
    auto* bodyBlock = &execOp.body().front();

    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange blockArgs) {
        mlir::BlockAndValueMapping mapper;

        const auto prevBlockArgs = prevBodyBlock->getArguments();
        const auto curBlockArgs = bodyBlock->getArguments();

        for (size_t i = 0; i < blockArgs.size(); ++i) {
            if (i < prevBlockArgs.size()) {
                mapper.map(prevBlockArgs[i], blockArgs[i]);
            } else {
                mapper.map(curBlockArgs[i - prevBlockArgs.size()], blockArgs[i]);
            }
        }

        SmallVector<mlir::Value> newResults;

        const auto copyOps = [&](mlir::Block* bodyBlock) {
            for (auto& op : bodyBlock->getOperations()) {
                if (!mlir::isa<mlir::async::YieldOp>(op)) {
                    builder.clone(op, mapper);
                } else {
                    for (auto operand : op.getOperands()) {
                        newResults.push_back(mapper.lookupOrDefault(operand));
                    }
                }
            }
        };

        copyOps(prevBodyBlock);
        copyOps(bodyBlock);

        builder.create<mlir::async::YieldOp>(loc, newResults);
    };

    const auto prevResultTypes = prevBodyBlock->getTerminator()->getOperandTypes();
    const auto resultTypes = bodyBlock->getTerminator()->getOperandTypes();

    SmallVector<mlir::Type> newResultTypes(prevResultTypes);
    newResultTypes.insert(newResultTypes.end(), resultTypes.begin(), resultTypes.end());

    SmallVector<mlir::Value> newDependencies(prevExecOp.dependencies());
    newDependencies.insert(newDependencies.end(), execOp.dependencies().begin(), execOp.dependencies().end());

    SmallVector<mlir::Value> newOperands(prevExecOp.operands());
    newOperands.insert(newOperands.end(), execOp.operands().begin(), execOp.operands().end());

    auto newExecOp = rewriter.create<mlir::async::ExecuteOp>(prevExecOp->getLoc(), newResultTypes, newDependencies,
                                                             newOperands, bodyBuilder);

    auto executor = vpux::VPUIP::VPUIPDialect::getExecutor(execOp);
    VPUIP::VPUIPDialect::setExecutor(newExecOp, executor);

    return newExecOp;
}

//
// cleanup
//

void cleanup(mlir::async::ExecuteOp prevExecOp, mlir::async::ExecuteOp execOp, mlir::async::ExecuteOp newExecOp,
             mlir::PatternRewriter& rewriter, Logger log) {
    log.trace("Redirect results of original 'async.execute' operations");

    SmallVector<mlir::Value> matchedResultsPrev;
    SmallVector<mlir::Value> matchedResultsCurr;

    // newExecOp returns one token which replaces both tokens from original ops
    matchedResultsPrev.push_back(newExecOp.token());
    matchedResultsCurr.push_back(newExecOp.token());

    for (auto p : newExecOp.results() | indexed) {
        const auto ind = p.index();
        const auto newRes = p.value();

        if (ind < prevExecOp.results().size()) {
            matchedResultsPrev.push_back(newRes);
        } else {
            matchedResultsCurr.push_back(newRes);
        }
    }

    rewriter.replaceOp(prevExecOp, matchedResultsPrev);
    rewriter.replaceOp(execOp, matchedResultsCurr);
}

//
// GroupAsyncExecuteOps
//

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
    _log.trace("Got 'async.execute' operation at '{0}'", execOp->getLoc());

    if (!isOptimizableOp(execOp)) {
        return matchFailed(_log.nest(), rewriter, execOp, "The operation is not optimizible");
    }

    auto prevExecOp = mlir::dyn_cast_or_null<mlir::async::ExecuteOp>(execOp->getPrevNode());
    if (prevExecOp == nullptr) {
        return matchFailed(_log.nest(), rewriter, execOp, "Previous operation is not 'async.execute'");
    }

    if (!isSameExecutor(prevExecOp, execOp)) {
        return matchFailed(_log.nest(), rewriter, execOp, "Previous 'async.execute' uses another executor");
    }

    if (!haveSameDependencies(prevExecOp, execOp)) {
        return matchFailed(_log.nest(), rewriter, execOp, "Previous 'async.execute' has different dependencies");
    }

    /*  TODO: Remove check below when proper operands mapping is implemented.
        If current execute op depends on previous then it requires more complex operands/results mapping.
        Mapping path:
        1st execute op yield op operands -> 1st execute op result->
        -> 2nd execute op operand-> internal block argument-> operation operand */
    if (llvm::is_contained(prevExecOp->getUsers(), execOp)) {
        return matchFailed(_log.nest(), rewriter, execOp,
                           "Current 'async.execute' depends on previous 'async.execute'");
    }

    if (prevHasUniqueUsers(prevExecOp, execOp)) {
        return matchFailed(_log.nest(), rewriter, execOp, "Previous 'async.execute' has exclusive users");
    }

    auto newExecOp = mergeAsyncExecuteOps(prevExecOp, execOp, rewriter);
    cleanup(prevExecOp, execOp, newExecOp, rewriter, _log.nest());

    return mlir::success();
}

//
// GroupAsyncExecuteOpsPass
//

class GroupAsyncExecuteOpsPass final : public VPUIP::GroupAsyncExecuteOpsBase<GroupAsyncExecuteOpsPass> {
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
    patterns.add<GroupAsyncExecuteOps>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getFunction(), std::move(patterns),
                                                        getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }

    auto& depsInfo = getAnalysis<AsyncDepsInfo>();
    depsInfo.updateTokenDependencies();
}

}  // namespace

//
// createGroupAsyncExecuteOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createGroupAsyncExecuteOpsPass(Logger log) {
    return std::make_unique<GroupAsyncExecuteOpsPass>(log);
}
