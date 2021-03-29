//
// Copyright Intel Corporation.
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
    explicit AddLinearSchedulingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;

private:
    void collectTrailingUPATasks(mlir::FuncOp graphFunc);

    static VPUIP::TaskOpInterface getPrevTask(mlir::Operation* op);
    static VPUIP::TaskOpInterface getNextTask(mlir::Operation* op);

private:
    std::unordered_set<mlir::Operation*> _trailingSwTasks;
};

//
// safeRunOnModule
//

void AddLinearSchedulingPass::safeRunOnModule() {
    auto module = getOperation();

    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    _log.trace("Try to find trailing UPA Tasks");
    collectTrailingUPATasks(netFunc);

    uint32_t barrierID = 0;

    const auto callback = [&](mlir::Operation* op) {
        _log.trace("Process Operation '{0}'", op->getLoc());

        auto curTask = mlir::dyn_cast<VPUIP::TaskOpInterface>(op);
        if (curTask == nullptr) {
            _log.trace("It is not a VPUIP Task");
            return;
        }

        if (_trailingSwTasks.count(op) != 0) {
            _log.trace("It is a trailing UPA Task");

            auto upaTask = mlir::cast<VPUIP::UPATaskOpInterface>(op);
            upaTask.markAsTrailingSWLayer();

            return;
        }

        if (auto prevTask = getPrevTask(op)) {
            _log.trace("It has dependency on previous task '{0}'", prevTask->getLoc());

            const auto prevBarriers = prevTask.updateBarriers();
            curTask.waitBarriersMutable().append(prevBarriers);
        }

        if (getNextTask(op) != nullptr) {
            _log.trace("It has dependent task");

            OpBuilderLogger builderLog(_log);
            mlir::OpBuilder builder(op, &builderLog);

            auto newBarrierOp = builder.create<VPUIP::ConfigureBarrierOp>(op->getLoc(), barrierID);
            curTask.updateBarriersMutable().append(mlir::ValueRange{newBarrierOp.barrier()});

            barrierID = (barrierID + 1) % 2;
        }
    };

    netFunc.walk(callback);
}

//
// collectTrailingSwTasks
//

void AddLinearSchedulingPass::collectTrailingUPATasks(mlir::FuncOp graphFunc) {
    auto tasks = to_small_vector(graphFunc.getOps<VPUIP::TaskOpInterface>());

    for (auto curTask : tasks | reversed) {
        _log.trace("Process Task '{0}'", curTask->getLoc());

        if (curTask.getTaskType() != VPUIP::TaskType::UPA) {
            _log.trace("Is it not an UPA task, stop processing");
            break;
        }

        bool hasNonUPADep = false;
        for (auto updateBarrier : curTask.updateBarriers()) {
            for (auto* depOp : updateBarrier.getUsers()) {
                if (!mlir::isa<VPUIP::UPATaskOpInterface>(depOp)) {
                    _log.trace("Is has non UPA dependent task '{0}'", depOp->getLoc());

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

        _log.trace("Is is a trailing UPA task");
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
