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

#include "vpux/compiler/utils/logging.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <llvm/ADT/DenseMap.h>

using namespace vpux;

namespace {

void moveWaitResults(mlir::async::AwaitOp waitOp, Logger log) {
    const auto futureVal = waitOp.operand();

    auto producerExecOp = mlir::dyn_cast_or_null<mlir::async::ExecuteOp>(futureVal.getDefiningOp());
    VPUX_THROW_UNLESS(producerExecOp != nullptr, "'async.await' operand is produced by unsupported operation");

    const auto futureType = futureVal.getType().dyn_cast<mlir::async::ValueType>();
    VPUX_THROW_UNLESS(futureType != nullptr, "'async.await' operand has unexpected type : '{0}'", futureVal.getType());

    log.trace("Collect all uses inside 'async.execute' regions");

    using ExecutorToAsyncUsesMap = llvm::DenseMap<mlir::Operation*, SmallVector<mlir::OpOperand*>>;
    ExecutorToAsyncUsesMap allAsyncUses;

    for (auto& use : waitOp.result().getUses()) {
        if (auto userExecOp = use.getOwner()->getParentOfType<mlir::async::ExecuteOp>()) {
            allAsyncUses[userExecOp].push_back(&use);
        }
    }

    log.trace("Update uses inside 'async.execute' regions");

    for (const auto& p : allAsyncUses) {
        auto userExecOp = mlir::cast<mlir::async::ExecuteOp>(p.first);

        userExecOp.operandsMutable().append(futureVal);
        const auto innerArg = userExecOp.getBody()->addArgument(futureType.getValueType());

        log.nest().trace("Add '{0}' as async operand for '{1}', which is mapped to '{2}'", futureVal,
                         userExecOp->getLoc(), innerArg);

        for (auto* use : p.second) {
            log.nest().trace("Redirect '{0}' to '{1}'", *use->getOwner(), innerArg);
            use->set(innerArg);
        }
    }

    if (!waitOp.result().use_empty()) {
        return;
    }

    log.trace("The operation result has no use left, replace it with no-result 'async.await'");

    OpBuilderLogger builderLog(log.nest());
    mlir::OpBuilder builder(waitOp, &builderLog);

    builder.create<mlir::async::AwaitOp>(waitOp->getLoc(), producerExecOp.token(), waitOp->getAttrs());
    waitOp->erase();
}

class MoveWaitResultToAsyncBlockArgsPass final :
        public IERT::MoveWaitResultToAsyncBlockArgsBase<MoveWaitResultToAsyncBlockArgsPass> {
public:
    explicit MoveWaitResultToAsyncBlockArgsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void MoveWaitResultToAsyncBlockArgsPass::safeRunOnFunc() {
    auto func = getFunction();

    func.walk([this](mlir::async::AwaitOp waitOp) {
        if (waitOp.result() == nullptr) {
            return;
        }

        _log.trace("Process 'async.await' Operation at '{0}'", waitOp->getLoc());
        moveWaitResults(waitOp, _log.nest());
    });
}

}  // namespace

//
// createMoveWaitResultToAsyncBlockArgsPass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createMoveWaitResultToAsyncBlockArgsPass(Logger log) {
    return std::make_unique<MoveWaitResultToAsyncBlockArgsPass>(log);
}
