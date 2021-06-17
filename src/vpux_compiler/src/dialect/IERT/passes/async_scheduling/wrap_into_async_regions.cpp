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

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

void warpIntoAsyncRegion(mlir::Operation* op, const IERT::LayerInfoDialectInterface* layerInfo, Logger log) {
    if (op->getParentOfType<mlir::async::ExecuteOp>() != nullptr) {
        log.trace("[SKIP] The Operation already wrapped into asynchronous region");
        return;
    }

    uint32_t numExecutorUnits = 0;
    const auto executor = layerInfo->getExecutor(op, numExecutorUnits);
    if (executor != nullptr) {
        log.trace("It will be executed on '{0}' units of '{1}' Executor", numExecutorUnits, executor);
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
        IERT::IERTDialect::setExecutor(execOp, executor, numExecutorUnits);
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
    auto* iert = getContext().getLoadedDialect<IERT::IERTDialect>();
    VPUX_THROW_UNLESS(iert != nullptr, "IERT Dialect was not loaded");

    const auto* layerInfo = iert->getRegisteredInterface<IERT::LayerInfoDialectInterface>();
    VPUX_THROW_UNLESS(layerInfo != nullptr, "IERT Dialect was not initialized with LayerInfo interface");

    const auto callback = [&](mlir::Operation* op) {
        if (!op->hasTrait<RTLayer>()) {
            return;
        }

        _log.trace("Process Layer Operation '{0}' at '{1}'", op->getName(), op->getLoc());
        warpIntoAsyncRegion(op, layerInfo, _log.nest());
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
