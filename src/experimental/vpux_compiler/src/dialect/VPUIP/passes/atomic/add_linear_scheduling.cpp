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

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"

#include <mlir/IR/BuiltinOps.h>

#include <unordered_set>

using namespace vpux;

namespace {

class AddLinearSchedulingPass final : public VPUIP::AddLinearSchedulingBase<AddLinearSchedulingPass> {
public:
    explicit AddLinearSchedulingPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    void runOnOperation() final;

private:
    void passBody();

private:
    void collectTrailingUPATasks(mlir::FuncOp graphFunc);

    static VPUIP::TaskOpInterface getPrevTask(mlir::Operation* op);
    static VPUIP::TaskOpInterface getNextTask(mlir::Operation* op);

private:
    Logger _log;
    std::unordered_set<mlir::Operation*> _trailingSwTasks;
};

void AddLinearSchedulingPass::runOnOperation() {
    try {
        passBody();
    } catch (const std::exception& e) {
        printTo(getOperation().emitError(), "{0} Pass failed : {1}", getName(), e.what());
        signalPassFailure();
    }
}

//
// passBody
//

void AddLinearSchedulingPass::passBody() {
    auto module = getOperation();

    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    auto graphOp = VPUIP::GraphOp::getFromModule(module);

    _log.trace("Try to find trailing UPA Tasks");
    collectTrailingUPATasks(netFunc);

    const auto callback = [&](mlir::Operation* op) {
        auto innerLog = _log.nest();

        innerLog.trace("Process Operation '{0}'", *op);

        auto curTask = mlir::dyn_cast<VPUIP::TaskOpInterface>(op);
        if (curTask == nullptr) {
            innerLog.trace("It is not a VPUIP Task");
            return;
        }

        if (_trailingSwTasks.count(op) != 0) {
            innerLog.trace("It is a trailing UPA Task");

            auto upaTask = mlir::cast<VPUIP::UPATaskOpInterface>(op);
            upaTask.markAsTrailingSWLayer();

            return;
        }

        if (auto prevTask = getPrevTask(op)) {
            innerLog.trace("It has dependency on previous task '{0}'", prevTask);

            const auto prevBarriers = prevTask.updateBarriers();
            curTask.waitBarriersMutable().append(prevBarriers);
        }

        if (getNextTask(op) != nullptr) {
            innerLog.trace("It has dependent task");

            OpBuilderLogger builderLog(_log.nest());
            mlir::OpBuilder builder(op, &builderLog);

            auto newBarrierOp = builder.create<VPUIP::ConfigureBarrierOp>(op->getLoc());
            curTask.updateBarriersMutable().append(mlir::ValueRange{newBarrierOp.barrier()});
        }
    };

    netFunc.walk(callback);

    _log.trace("Update VPUIP.Graph Operation 'options' attribute");
    auto options = graphOp.options();
    options = options | VPUIP::ExecutionFlag::DynamicBarriers;
    graphOp.optionsAttr(VPUIP::ExecutionFlagAttr::get(options, module.getContext()));
}

//
// collectTrailingSwTasks
//

void AddLinearSchedulingPass::collectTrailingUPATasks(mlir::FuncOp graphFunc) {
    auto tasks = to_vector<16>(graphFunc.getOps<VPUIP::TaskOpInterface>());

    for (auto curTask : tasks | reversed) {
        auto innerLog = _log.nest();

        innerLog.trace("Process Task '{0}'", curTask);

        if (curTask.getTaskType() != VPUIP::TaskType::UPA) {
            innerLog.trace("Is it not an UPA task, stop processing");
            break;
        }

        bool hasNonUPADep = false;
        for (auto updateBarrier : curTask.updateBarriers()) {
            for (auto* depOp : updateBarrier.getUsers()) {
                if (!mlir::isa<VPUIP::UPATaskOpInterface>(depOp)) {
                    innerLog.trace("Is has non UPA dependent task '{0}'", *depOp);

                    hasNonUPADep = true;
                    break;
                }
            }

            if (hasNonUPADep) {
                break;
            }
        }

        if (hasNonUPADep) {
            break;
        }

        innerLog.trace("Is is a trailing UPA task");
        _trailingSwTasks.insert(curTask.getOperation());
    }
}

//
// getPrevTask & getNextTask
//

VPUIP::TaskOpInterface AddLinearSchedulingPass::getPrevTask(mlir::Operation* op) {
    auto* prevOp = op->getPrevNode();
    while (prevOp != nullptr && mlir::dyn_cast<VPUIP::TaskOpInterface>(prevOp) == nullptr) {
        prevOp = prevOp->getPrevNode();
    }
    return mlir::dyn_cast_or_null<VPUIP::TaskOpInterface>(prevOp);
}

VPUIP::TaskOpInterface AddLinearSchedulingPass::getNextTask(mlir::Operation* op) {
    auto* nextOp = op->getNextNode();
    while (nextOp != nullptr && mlir::dyn_cast<VPUIP::TaskOpInterface>(nextOp) == nullptr) {
        nextOp = nextOp->getNextNode();
    }
    return mlir::dyn_cast_or_null<VPUIP::TaskOpInterface>(nextOp);
}

}  // namespace

//
// createAddLinearSchedulingPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createAddLinearSchedulingPass(Logger log) {
    return std::make_unique<AddLinearSchedulingPass>(log);
}
