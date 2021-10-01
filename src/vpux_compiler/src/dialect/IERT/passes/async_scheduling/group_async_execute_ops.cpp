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

#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
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

bool isSameTimeExecution(mlir::async::ExecuteOp execOp1, mlir::async::ExecuteOp execOp2) {
    const auto attr1 = execOp1->getAttrOfType<mlir::IntegerAttr>("schedule-time");
    VPUX_THROW_UNLESS(attr1 != nullptr, "Attribute schedule-time was not set for '{0}' operation at '{1}'",
                      execOp1->getName(), execOp1->getLoc());
    const auto attr2 = execOp2->getAttrOfType<mlir::IntegerAttr>("schedule-time");
    VPUX_THROW_UNLESS(attr2 != nullptr, "Attribute schedule-time was not set for '{0}' operation at '{1}'",
                      execOp2->getName(), execOp2->getLoc());

    auto time1 = checked_cast<uint32_t>(attr1.getValue().getZExtValue());
    auto time2 = checked_cast<uint32_t>(attr2.getValue().getZExtValue());
    return time1 == time2;
}

bool prevHasUniqueUsers(mlir::async::ExecuteOp prevExecOp, mlir::async::ExecuteOp execOp) {
    auto getUsers = [](mlir::async::ExecuteOp op) {
        std::set<mlir::async::ExecuteOp> users;
        for (auto res : op.getResults())
            for (auto user : res.getUsers())
                users.insert(mlir::dyn_cast<mlir::async::ExecuteOp>(user));
        return users;
    };

    std::set<mlir::async::ExecuteOp> usersPrevOp = getUsers(prevExecOp);
    std::set<mlir::async::ExecuteOp> usersOp = getUsers(execOp);
    usersPrevOp.erase(execOp);

    llvm::SmallVector<mlir::async::ExecuteOp> uniqueUsersPrevOp;
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
    const auto prevScheduleTime = prevExecOp->getAttr("schedule-time");

    SmallVector<mlir::Type> newResultTypes(prevResultTypes);
    newResultTypes.insert(newResultTypes.end(), resultTypes.begin(), resultTypes.end());

    SmallVector<mlir::Value> newDependencies(prevExecOp.dependencies());
    newDependencies.insert(newDependencies.end(), execOp.dependencies().begin(), execOp.dependencies().end());

    SmallVector<mlir::Value> newOperands(prevExecOp.operands());
    newOperands.insert(newOperands.end(), execOp.operands().begin(), execOp.operands().end());

    auto newExecOp = rewriter.create<mlir::async::ExecuteOp>(prevExecOp->getLoc(), newResultTypes, newDependencies,
                                                             newOperands, bodyBuilder);

    newExecOp->setAttr("schedule-time", prevScheduleTime);

    uint32_t numUnits = 0;
    auto executor = vpux::IERT::IERTDialect::getExecutor(execOp, numUnits);
    IERT::IERTDialect::setExecutor(newExecOp, executor, numUnits);

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

    if (!isSameTimeExecution(prevExecOp, execOp)) {
        return matchFailed(_log.nest(), rewriter, execOp, "Previous 'async.execute' scheduled at different time");
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

std::unique_ptr<mlir::Pass> vpux::IERT::createGroupAsyncExecuteOpsPass(Logger log) {
    return std::make_unique<GroupAsyncExecuteOpsPass>(log);
}
