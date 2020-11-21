//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/mlir/logging.hpp"

#include <mlir/IR/BuiltinOps.h>

#include <unordered_set>

using namespace vpux;

namespace {

class AddLinearSchedulingPass final
        : public VPUIP::AddLinearSchedulingBase<AddLinearSchedulingPass> {
public:
    void runOnOperation() final;

private:
    void passBody();

    void collectTrailingSwTasks(mlir::FuncOp graphFunc);

    static VPUIP::TaskOpInterface getPrevTask(mlir::Operation* op);
    static VPUIP::TaskOpInterface getNextTask(mlir::Operation* op);

private:
    std::unordered_set<mlir::Operation*> _trailingSwTasks;
};

void AddLinearSchedulingPass::runOnOperation() {
    try {
        passBody();
    } catch (const std::exception& e) {
        printTo(getOperation().emitError(),
                "AddLinearSchedulingPass failed : {0}",
                e.what());
        signalPassFailure();
    }
}

void AddLinearSchedulingPass::passBody() {
    auto module = getOperation();

    VPUIP::GraphOp graphOp;
    mlir::FuncOp graphFunc;
    if (mlir::failed(
                VPUIP::GraphOp::getFromModule(module, graphOp, graphFunc))) {
        signalPassFailure();
        return;
    }

    collectTrailingSwTasks(graphFunc);

    const auto callback = [&](mlir::Operation* op) {
        auto curTask = mlir::dyn_cast<VPUIP::TaskOpInterface>(op);
        if (curTask == nullptr) {
            return;
        }

        if (_trailingSwTasks.count(op) != 0) {
            auto upaTask = mlir::cast<VPUIP::UPATaskOpInterface>(op);
            upaTask.markAsTrailingSWLayer();
            return;
        }

        if (auto prevTask = getPrevTask(op)) {
            const auto prevBarriers = prevTask.updateBarriers();
            curTask.waitBarriersMutable().append(prevBarriers);
        }

        if (getNextTask(op) != nullptr) {
            mlir::OpBuilder builder(op);
            auto newBarrierOp =
                    builder.create<VPUIP::ConfigureBarrierOp>(op->getLoc());
            curTask.updateBarriersMutable().append(
                    mlir::ValueRange{newBarrierOp.barrier()});
        }
    };

    graphFunc.walk(callback);

    auto options = graphOp.options();
    options = options | VPUIP::ExecutionFlag::DynamicBarriers;
    graphOp.optionsAttr(
            VPUIP::ExecutionFlagAttr::get(module.getContext(), options));
}

void AddLinearSchedulingPass::collectTrailingSwTasks(mlir::FuncOp graphFunc) {
    auto tasks = to_vector<16>(graphFunc.getOps<VPUIP::TaskOpInterface>());

    for (auto curTask : tasks | reversed) {
        if (curTask.getTaskType() != VPUIP::TaskType::UPA) {
            break;
        }

        bool hasNonSwDep = false;
        for (auto updateBarrier : curTask.updateBarriers()) {
            for (auto* depOp : updateBarrier.getUsers()) {
                auto depTask = mlir::dyn_cast<VPUIP::TaskOpInterface>(depOp);

                if (depTask == nullptr) {
                    hasNonSwDep = true;
                    break;
                }

                if (depTask.getTaskType() != VPUIP::TaskType::UPA) {
                    hasNonSwDep = true;
                    break;
                }
            }

            if (hasNonSwDep) {
                break;
            }
        }

        if (hasNonSwDep) {
            break;
        }

        _trailingSwTasks.insert(curTask.getOperation());
    }
}

VPUIP::TaskOpInterface
        AddLinearSchedulingPass::getPrevTask(mlir::Operation* op) {
    auto* prevOp = op->getPrevNode();
    while (prevOp != nullptr &&
           mlir::dyn_cast<VPUIP::TaskOpInterface>(prevOp) == nullptr) {
        prevOp = prevOp->getPrevNode();
    }
    return mlir::dyn_cast_or_null<VPUIP::TaskOpInterface>(prevOp);
}

VPUIP::TaskOpInterface
        AddLinearSchedulingPass::getNextTask(mlir::Operation* op) {
    auto* nextOp = op->getNextNode();
    while (nextOp != nullptr &&
           mlir::dyn_cast<VPUIP::TaskOpInterface>(nextOp) == nullptr) {
        nextOp = nextOp->getNextNode();
    }
    return mlir::dyn_cast_or_null<VPUIP::TaskOpInterface>(nextOp);
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPUIP::createAddLinearSchedulingPass() {
    return std::make_unique<AddLinearSchedulingPass>();
}
