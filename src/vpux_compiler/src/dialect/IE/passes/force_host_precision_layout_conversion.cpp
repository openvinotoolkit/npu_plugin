//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/compiler/dialect/IE/utils/operations_detection_utils.hpp"

using namespace vpux;
using namespace IE;

namespace {

//
// ForceHostPrecisionLayoutConversionPass
//

class ForceHostPrecisionLayoutConversionPass final :
        public IE::ForceHostPrecisionLayoutConversionBase<ForceHostPrecisionLayoutConversionPass> {
public:
    explicit ForceHostPrecisionLayoutConversionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void ForceHostPrecisionLayoutConversionPass::safeRunOnModule() {
    auto module = getOperation();

    IE::CNNNetworkOp netInfo;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netInfo, netFunc);

    SmallVector<mlir::Value> newArgs, newResults;
    newArgs.reserve(netFunc.getNumArguments());
    newResults.reserve(netFunc.getNumResults());

    _log.trace("Detect pre- and post- processing");

    SmallVector<mlir::Operation*> preProcOps, postProcOps;

    for (const auto arg : netFunc.getArguments()) {
        const auto newArg = detectValueAfterOperation<IE::ConvertOp, IE::ReorderOp>(arg, preProcOps);
        if (newArg != arg) {
            _log.nest().trace("Argument '{0}' is pre-processed to '{1}'", arg, newArg);
        }
        newArgs.push_back(newArg);
    }
    netFunc.walk([&](mlir::ReturnOp retOp) {
        for (const auto res : retOp->getOperands()) {
            const auto newRes = detectValueBeforeOperation<IE::ConvertOp, IE::ReorderOp>(res, postProcOps);
            if (newRes != res) {
                _log.nest().trace("Result '{0}' is post-processed to '{1}'", res, newRes);
            }
            newResults.push_back(newRes);
        }
    });

    if (preProcOps.empty() && postProcOps.empty()) {
        _log.trace("No pre- or post- processing detected");
        return;
    }

    vpux::IE::detectAndReplacePreProcessedAgrs(_log, netFunc, newArgs, ArrayRef<mlir::Value>(newResults));

    _log.trace("Redirect results");

    vpux::IE::detectAndReplacePostProcessedRes(_log, netFunc, ArrayRef<mlir::Value>(newResults));

    _log.trace("Remove obsolete pre-/post- processing operations");

    const auto removeOps = [](auto ops) {
        for (auto* op : ops) {
            VPUX_THROW_UNLESS(op->use_empty(), "Operation '{0}' at '{1}' still has users", op->getName(), op->getLoc());
            op->erase();
        }
    };

    removeOps(preProcOps | reversed);
    removeOps(postProcOps);
}

}  // namespace

//
// createForceHostPrecisionLayoutConversionPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createForceHostPrecisionLayoutConversionPass(Logger log) {
    return std::make_unique<ForceHostPrecisionLayoutConversionPass>(log);
}
