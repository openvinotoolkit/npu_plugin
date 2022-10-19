//
// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/logging.hpp"

using namespace vpux;

namespace {

class MaximizeUPACyclesPass final : public VPUIP::MaximizeUPACyclesBase<MaximizeUPACyclesPass> {
public:
    explicit MaximizeUPACyclesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

class CycleComparator {
    std::function<size_t(mlir::async::ExecuteOp)> _cycleGetter;
    AsyncDepsInfo& _depsInfo;

public:
    CycleComparator(AsyncDepsInfo& depsInfo, std::function<size_t(mlir::async::ExecuteOp)> func): _depsInfo(depsInfo) {
        _cycleGetter = func;
    }
    bool operator()(size_t opIndex1, size_t opIndex2) const {
        auto execOp1 = _depsInfo.getExecuteOpAtIndex(opIndex1);
        auto execOp2 = _depsInfo.getExecuteOpAtIndex(opIndex2);

        return _cycleGetter(execOp1) < _cycleGetter(execOp2);
    }
};

void recalculateUPACycles(mlir::FuncOp func, AsyncDepsInfo& depsInfo, mlir::async::ExecuteOp execOp) {
    const auto execOpIdx = depsInfo.getIndex(execOp);
    const auto deps = depsInfo.getOpDeps(execOpIdx);
    const auto consumers = depsInfo.getConsumerOps(execOpIdx);

    size_t upaCycleBegin = 0;
    size_t upaCycleEnd = getAsyncExecuteCycleEnd(to_small_vector(func.getOps<mlir::async::ExecuteOp>()).back());

    if (!deps.empty()) {
        const auto element =
                *std::max_element(deps.begin(), deps.end(), CycleComparator(depsInfo, getAsyncExecuteCycleEnd));
        upaCycleBegin = getAsyncExecuteCycleEnd(depsInfo.getExecuteOpAtIndex(element));
    }

    if (!consumers.empty()) {
        const auto element = *std::min_element(consumers.begin(), consumers.end(),
                                               CycleComparator(depsInfo, getAsyncExecuteCycleBegin));
        upaCycleEnd = getAsyncExecuteCycleBegin(depsInfo.getExecuteOpAtIndex(element));
    }

    execOp->setAttr(cycleBegin, getIntAttr(execOp->getContext(), upaCycleBegin));
    execOp->setAttr(cycleEnd, getIntAttr(execOp->getContext(), upaCycleEnd));
    execOp->setAttr(cycleCostAttrName, getIntAttr(execOp->getContext(), upaCycleEnd - upaCycleBegin));
}

void MaximizeUPACyclesPass::safeRunOnFunc() {
    auto func = getFunction();
    auto& depsInfo = getAnalysis<AsyncDepsInfo>();
    depsInfo.buildConsMap();

    func->walk([&](mlir::async::ExecuteOp asyncExec) {
        // recalculate cycle for UPA according to its dependencies and consumers
        if (asyncExec->hasAttr(VPUIP::VPUIPDialect::getExecutorAttrName())) {
            const auto executor = VPUIP::VPUIPDialect::getExecutor(asyncExec);
            if (executor.getLeafNameAttr() ==
                VPU::ExecutorKindAttr::get(asyncExec->getContext(), VPU::ExecutorKind::SHAVE_UPA)) {
                // calculate UPA cycles
                recalculateUPACycles(func, depsInfo, asyncExec);
            }
        }
    });
}

}  // namespace

//
// createMaximizeUPACyclesPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createMaximizeUPACyclesPass(Logger log) {
    return std::make_unique<MaximizeUPACyclesPass>(log);
}
