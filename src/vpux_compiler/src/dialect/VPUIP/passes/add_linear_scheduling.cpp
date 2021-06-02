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

constexpr uint32_t MAX_BARRIERS_PER_INFERENCE = 32;
constexpr uint32_t BARRIERS_PER_CLUSTER = 8;

class AddLinearSchedulingPass final : public VPUIP::AddLinearSchedulingBase<AddLinearSchedulingPass> {
public:
    explicit AddLinearSchedulingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;

private:
    static VPUIP::TaskOpInterface getPrevTask(mlir::Operation* op);
    static VPUIP::TaskOpInterface getNextTask(mlir::Operation* op);
};

//
// safeRunOnModule
//

void AddLinearSchedulingPass::safeRunOnModule() {
    auto module = getOperation();

    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    auto resOp = IERT::RunTimeResourcesOp::getFromModule(module);
    const auto nceAttr = VPUIP::PhysicalProcessorAttr::get(module.getContext(), VPUIP::PhysicalProcessor::NCE_Cluster);
    auto nceResOp = resOp.getExecutor(nceAttr);
    VPUX_THROW_UNLESS(nceResOp != nullptr, "Failed to get NCE_Cluster infromation");
    const uint32_t numClusters = nceResOp.count();

    const uint32_t numBarriers = std::min(MAX_BARRIERS_PER_INFERENCE, BARRIERS_PER_CLUSTER * numClusters);
    uint32_t barrierID = 0;

    const auto callback = [&](mlir::Operation* op) {
        _log.trace("Process Operation '{0}'", op->getLoc());

        auto curTask = mlir::dyn_cast<VPUIP::TaskOpInterface>(op);
        if (curTask == nullptr) {
            _log.trace("It is not a VPUIP Task");
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

            // [Track number: W#6150] : use better algorithm for barriers allocation
            barrierID = (barrierID + 1) % numBarriers;
        }
    };

    netFunc.walk(callback);
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
