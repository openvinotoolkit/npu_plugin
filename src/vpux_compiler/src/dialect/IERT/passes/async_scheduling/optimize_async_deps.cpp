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

#include "vpux/compiler/utils/extentions.hpp"

#include "vpux/utils/core/range.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetOperations.h>
#include <llvm/ADT/SmallPtrSet.h>

#include <algorithm>

using namespace vpux;

namespace {

//
// Optimizer
//

class Optimizer final {
public:
    void buildDepsMap(mlir::FuncOp func, Logger log);
    void optimizeDepsMap(mlir::FuncOp func, Logger log);
    void addTokenDependencies(mlir::FuncOp func, Logger log);
    void optimizeWaitOps(Logger log);

private:
    using OperationSet = llvm::SmallPtrSet<mlir::Operation*, 16>;

private:
    SmallVector<mlir::async::ExecuteOp> _allExecOps;
    SmallVector<mlir::async::AwaitOp> _allWaitOps;

    // mlir::async::ExecuteOp 'depends on' [ mlir::async::ExecuteOp ]
    llvm::DenseMap<mlir::Operation*, OperationSet> _depsMap;
};

//
// Optimizer::buildDepsMap
//

void Optimizer::buildDepsMap(mlir::FuncOp func, Logger log) {
    log.trace("Collect initial dependencies maps");

    for (auto& op : func.getOps()) {
        if (auto waitOp = mlir::dyn_cast<mlir::async::AwaitOp>(op)) {
            log.nest(1).trace("Found 'async.await' Operation at '{0}'", op.getLoc());

            auto producerExecOp = mlir::dyn_cast_or_null<mlir::async::ExecuteOp>(waitOp.operand().getDefiningOp());
            VPUX_THROW_UNLESS(producerExecOp != nullptr, "'async.await' operand is produced by unsupported operation");

            log.nest(2).trace("It was produced by 'async.execute' at '{0}'", producerExecOp->getLoc());

            if (waitOp.result() != nullptr) {
                for (auto* user : waitOp.result().getUsers()) {
                    if (auto userExecOp = user->getParentOfType<mlir::async::ExecuteOp>()) {
                        log.nest(2).trace("It has a user at '{0}' located in 'async.execute' region at '{1}'",
                                          user->getLoc(), userExecOp->getLoc());

                        log.nest(2).trace("Add '{0}' to dependency list of '{1}'", producerExecOp->getLoc(),
                                          userExecOp->getLoc());
                        _depsMap[userExecOp].insert(producerExecOp);
                    }
                }
            }

            _allWaitOps.push_back(waitOp);
        } else if (auto execOp = mlir::dyn_cast<mlir::async::ExecuteOp>(op)) {
            log.nest(1).trace("Found 'async.execute' Operation at '{0}'", op.getLoc());

            log.nest(2).trace("Extend its wait deps list with all previous 'async.await' Operations");
            for (auto prevWaitOp : _allWaitOps) {
                auto producerExecOp = mlir::cast<mlir::async::ExecuteOp>(prevWaitOp.operand().getDefiningOp());

                log.nest(3).trace("Add '{0}' to dependency list of '{1}'", producerExecOp->getLoc(), execOp->getLoc());
                _depsMap[execOp].insert(producerExecOp);
            }

            for (auto arg : execOp->getOperands()) {
                auto argExecOp = mlir::dyn_cast<mlir::async::ExecuteOp>(arg.getDefiningOp());
                VPUX_THROW_UNLESS(argExecOp != nullptr,
                                  "'async.execute' at '{0}' has operand '{1}' produced by unsupported Operation",
                                  execOp->getLoc(), arg);

                log.nest(2).trace("It has a dependency from other 'async.execute' Operation at '{0}'",
                                  argExecOp->getLoc());
                _depsMap[execOp].insert(argExecOp);
            }

            _allExecOps.push_back(execOp);
        }
    }
}

//
// Optimizer::optimizeDepsMap
//

void Optimizer::optimizeDepsMap(mlir::FuncOp func, Logger log) {
    //
    // A -> B -> C
    //
    // If B depends on A and C depends on [A, B] ==> we can remove A from C deps list,
    // since it will be implicit dependency taken from B.
    //

    log.trace("Remove redundant dependencies");

    auto allExecOps = to_small_vector(func.getOps<mlir::async::ExecuteOp>());

    for (auto execOp : allExecOps | reversed) {
        log.nest(1).trace("Process 'async.execute' Operation at '{0}'", execOp->getLoc());

        auto& curDeps = _depsMap[execOp];

        for (auto* curDep : curDeps) {
            auto& depOfDeps = _depsMap[curDep];
            llvm::set_subtract(curDeps, depOfDeps);
        }
    }
}

//
// Optimizer::addTokenDependencies
//

void Optimizer::addTokenDependencies(mlir::FuncOp func, Logger log) {
    log.trace("Add explicit '!async.token' based dependencies between 'async.execute' operations");

    for (auto execOp : func.getOps<mlir::async::ExecuteOp>()) {
        log.nest(1).trace("Process 'async.execute' Operation at '{0}'", execOp->getLoc());

        const auto& execDeps = _depsMap[execOp];

        llvm::SmallPtrSet<mlir::Value, 4> depsSet;
        for (auto* depOp : execDeps) {
            const auto tokenVal = mlir::cast<mlir::async::ExecuteOp>(depOp).token();
            depsSet.insert(tokenVal);
        }

        auto depsVec = to_small_vector(depsSet);

        std::sort(depsVec.begin(), depsVec.end(), [](mlir::Value val1, mlir::Value val2) {
            return val1.getParentBlock() == val2.getParentBlock() &&
                   val1.getDefiningOp()->isBeforeInBlock(val2.getDefiningOp());
        });

        log.nest(2).trace("Use the following explicit dependencies : {0}", depsVec);
        execOp.dependenciesMutable().assign(makeArrayRef(depsVec));
    }
}

//
// Optimizer::optimizeWaitOps
//

void Optimizer::optimizeWaitOps(Logger log) {
    log.trace("Optimize 'async.await' operations");

    for (auto waitOp : _allWaitOps) {
        log.nest(1).trace("Process 'async.await' Operation at '{0}'", waitOp->getLoc());

        if (waitOp.result() == nullptr || waitOp.result().use_empty()) {
            log.nest(2).trace("Remove the operation, since it has no use");
            waitOp->erase();
            continue;
        }

        if (auto* firstUser = getFirstUser(waitOp.result())) {
            log.nest(2).trace("Move the operation close to its first user at '{0}'", firstUser->getLoc());
            waitOp->moveBefore(firstUser);
        }
    }
}

//
// OptimizeAsyncDepsPass
//

class OptimizeAsyncDepsPass final : public IERT::OptimizeAsyncDepsBase<OptimizeAsyncDepsPass> {
public:
    explicit OptimizeAsyncDepsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void OptimizeAsyncDepsPass::safeRunOnFunc() {
    auto func = getFunction();

    Optimizer optimizer;
    optimizer.buildDepsMap(func, _log);
    optimizer.optimizeDepsMap(func, _log);
    optimizer.addTokenDependencies(func, _log);
    optimizer.optimizeWaitOps(_log);
}

}  // namespace

//
// createOptimizeAsyncDepsPass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createOptimizeAsyncDepsPass(Logger log) {
    return std::make_unique<OptimizeAsyncDepsPass>(log);
}
