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
// ForceHostQuantizationPass
//

class ForceHostQuantizationPass final : public IE::ForceHostQuantizationBase<ForceHostQuantizationPass> {
public:
    explicit ForceHostQuantizationPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void ForceHostQuantizationPass::safeRunOnModule() {
    auto module = getOperation();

    IE::CNNNetworkOp netInfo;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netInfo, netFunc);

    SmallVector<mlir::Value> newArgs, results;
    newArgs.reserve(netFunc.getNumArguments());

    _log.trace("Detect quantize operations");

    SmallVector<mlir::Operation*> preQuantOps;

    for (const auto arg : netFunc.getArguments()) {
        const auto newArg = vpux::IE::detectValueAfterOperation<IE::QuantizeOp>(arg, preQuantOps);

        newArgs.push_back(newArg);
    }
    netFunc.walk([&](mlir::ReturnOp retOp) {
        for (const auto res : retOp->getOperands()) {
            results.push_back(res);
        }
    });

    if (preQuantOps.empty()) {
        _log.trace("No quantize operations detected");
        return;
    }

    vpux::IE::detectAndReplacePreProcessedAgrs(_log, netFunc, newArgs, ArrayRef<mlir::Value>(results));

    _log.trace("Remove obsolete quantize operations");

    const auto removeOps = [](auto ops) {
        for (auto* op : ops) {
            if (op->use_empty()) {
                op->erase();
            }
        }
    };

    removeOps(preQuantOps | reversed);
}

}  // namespace

//
// createForceHostQuantizationPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createForceHostQuantizationPass(Logger log) {
    return std::make_unique<ForceHostQuantizationPass>(log);
}
