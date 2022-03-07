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
#include "vpux/compiler/utils/logging.hpp"

#include "vpux/utils/core/range.hpp"

using namespace vpux;

namespace {

void warpIntoAsyncRegion(IERT::AsyncLayerOpInterface op, Logger log) {
    if (op->getParentOfType<mlir::async::ExecuteOp>() != nullptr) {
        log.trace("[SKIP] The Operation already wrapped into asynchronous region");
        return;
    }

    const auto executor = op.getExecutor();
    if (executor != nullptr) {
        log.trace("It will be executed on '{0}' Executor", executor);
    }

    log.trace("Create 'async.execute' Operation");

    const auto bodyBuilder = [op](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange) {
        auto* newOp = builder.clone(*op);
        builder.create<mlir::async::YieldOp>(loc, newOp->getResults());
    };

    OpBuilderLogger builderLog(log.nest());
    mlir::OpBuilder builder(op, &builderLog);

    auto execOp = builder.create<mlir::async::ExecuteOp>(op->getLoc(), op->getResultTypes(), None, None, bodyBuilder);
    if (executor != nullptr) {
        IERT::IERTDialect::setExecutor(execOp, executor);
    }

    log.trace("Create 'async.await' Operations per each original result");

    SmallVector<mlir::Value> newResults;
    newResults.resize(op->getNumResults());
    for (auto i : irange(op->getNumResults())) {
        auto waitOp = builder.create<mlir::async::AwaitOp>(op->getLoc(), execOp.results()[i]);
        newResults[i] = waitOp.result();
    }

    log.trace("Replace the operation with new 'async.await' results");

    op->replaceAllUsesWith(newResults);
    op->erase();
}

//
// WrapIntoAsyncRegionsPass
//

class WrapIntoAsyncRegionsPass final : public IERT::WrapIntoAsyncRegionsBase<WrapIntoAsyncRegionsPass> {
public:
    explicit WrapIntoAsyncRegionsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void WrapIntoAsyncRegionsPass::safeRunOnFunc() {
    const auto callback = [&](IERT::AsyncLayerOpInterface op) {
        if (mlir::isa<IERT::AsyncLayerOpInterface>(op->getParentOp())) {
            _log.trace("Skip for operation '{0}' at '{1}' which is wrapped in other op", op->getName(), op->getLoc());
            return;
        }
        _log.trace("Process Layer Operation '{0}' at '{1}'", op->getName(), op->getLoc());
        warpIntoAsyncRegion(op, _log.nest());
    };

    getFunction().walk(callback);
}

}  // namespace

//
// createWrapIntoAsyncRegionsPass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createWrapIntoAsyncRegionsPass(Logger log) {
    return std::make_unique<WrapIntoAsyncRegionsPass>(log);
}
