//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

//
// UseHostPrePostProcessingPass
//

class UseHostPrePostProcessingPass final : public IE::UseHostPrePostProcessingBase<UseHostPrePostProcessingPass> {
public:
    explicit UseHostPrePostProcessingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

mlir::Value detectValueAfterPreProcessing(mlir::Value val, SmallVectorImpl<mlir::Operation*>& preProcOps) {
    if (!val.hasOneUse()) {
        return val;
    }

    if (auto convertOp = mlir::dyn_cast<IE::ConvertOp>(*val.user_begin())) {
        preProcOps.push_back(convertOp);
        return detectValueAfterPreProcessing(convertOp.output(), preProcOps);
    }
    if (auto reorderOp = mlir::dyn_cast<IE::ReorderOp>(*val.user_begin())) {
        preProcOps.push_back(reorderOp);
        return detectValueAfterPreProcessing(reorderOp.output(), preProcOps);
    }
    // TODO: support Quantize operation

    return val;
}

mlir::Value detectValueBeforePostProcessing(mlir::Value val, SmallVectorImpl<mlir::Operation*>& postProcOps) {
    auto* producer = val.getDefiningOp();
    if (producer == nullptr) {
        return val;
    }

    if (auto convertOp = mlir::dyn_cast<IE::ConvertOp>(producer)) {
        postProcOps.push_back(convertOp);
        return detectValueBeforePostProcessing(convertOp.input(), postProcOps);
    }
    if (auto reorderOp = mlir::dyn_cast<IE::ReorderOp>(producer)) {
        postProcOps.push_back(reorderOp);
        return detectValueBeforePostProcessing(reorderOp.input(), postProcOps);
    }
    // TODO: support Dequantize operation

    return val;
}

void UseHostPrePostProcessingPass::safeRunOnModule() {
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
        const auto newArg = detectValueAfterPreProcessing(arg, preProcOps);
        if (newArg != arg) {
            _log.nest().trace("Argument '{0}' is pre-processed to '{1}'", arg, newArg);
        }

        newArgs.push_back(newArg);
    }
    netFunc.walk([&](mlir::ReturnOp retOp) {
        for (const auto res : retOp->getOperands()) {
            const auto newRes = detectValueBeforePostProcessing(res, postProcOps);
            if (newRes != res) {
                _log.nest().trace("Result '{0}' is post-processed to '{1}'", newRes, res);
            }

            newResults.push_back(newRes);
        }
    });

    if (preProcOps.empty() && postProcOps.empty()) {
        _log.trace("No pre- or post- processing detected");
        return;
    }
    if (newResults.size() != netFunc.getNumResults()) {
        // It is a check for potential multiple return statements.
        // They can appear in case of control flow:
        //
        // scf.if (%some_condition) {
        //    %0 = IE.Op1(...)
        //    return %0
        // } else {
        //     %1 = IE.Op2(...)
        //     return %1
        // }
        _log.trace("Mismatch between the number of post-processing operations and results, '{0}' != '{1}'",
                   newResults.size(), netFunc.getNumResults());
        return;
    }

    const auto getType = [](mlir::Value val) {
        return val.getType();
    };
    const auto newArgTypes = to_small_vector(newArgs | transformed(getType));
    const auto newResultsTypes = to_small_vector(newResults | transformed(getType));

    if (updateFunctionSignature(netFunc, newArgTypes, newResultsTypes, _log).failed()) {
        signalPassFailure();
        return;
    }

    _log.trace("Redirect arguments");

    for (const auto& p : netFunc.getArguments() | indexed) {
        const auto ind = p.index();

        auto curArg = p.value();
        auto newArg = newArgs[ind];

        if (curArg != newArg) {
            _log.nest().trace("Switch argument at index '{0}' from '{1}' to '{2}'", ind, curArg, newArg);

            curArg.setType(newArg.getType());
            newArg.replaceAllUsesWith(curArg);
        }
    }

    _log.trace("Redirect results");

    netFunc.walk([&](mlir::ReturnOp retOp) {
        for (auto& curRes : retOp->getOpOperands()) {
            const auto newRes = newResults[curRes.getOperandNumber()];

            if (curRes.get() != newRes) {
                _log.nest().trace("Switch result from '{1}' to '{2}'", curRes.get(), newRes);
                curRes.set(newRes);
            }
        }
    });

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
// createUseHostPrePostProcessingPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createUseHostPrePostProcessingPass(Logger log) {
    return std::make_unique<UseHostPrePostProcessingPass>(log);
}
