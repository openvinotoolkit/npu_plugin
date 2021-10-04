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

#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/dialect/IERT/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IERT/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops_interfaces.hpp"
#include "vpux/compiler/utils/logging.hpp"

using namespace vpux;

namespace {

//
// SortAsyncExecuteOpsPass
//

class SortAsyncExecuteOpsPass final : public IERT::SortAsyncExecuteOpsBase<SortAsyncExecuteOpsPass> {
public:
    explicit SortAsyncExecuteOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void SortAsyncExecuteOpsPass::safeRunOnFunc() {
    auto func = getFunction();

    auto& depsInfo = getAnalysis<AsyncDepsInfo>();

    struct SchedOp {
        size_t opIndex;
        size_t startTime;
    };

    std::list<SchedOp> schedOpList;
    uint32_t min_time = std::numeric_limits<uint32_t>::max();

    // Check all AsyncExecuteOp to build schedOpList which will consist of operation
    // index and start time assigned by list scheduler
    func.walk([&](mlir::async::ExecuteOp execOp) {
        // AsyncExecuteOp index attribute will be used as operation identifier
        auto index = depsInfo.getIndex(execOp);

        const auto timeAttr = execOp->getAttrOfType<mlir::IntegerAttr>("schedule-time");
        uint32_t time = std::numeric_limits<uint32_t>::max();
        if (timeAttr != nullptr) {
            time = checked_cast<uint32_t>(timeAttr.getValue().getZExtValue());
        }

        schedOpList.push_back({index, time});

        if (time < min_time) {
            min_time = time;
        }
    });

    // TODO:
    // Reorganization logic in this pass currently assumes that first AsyncExecuteOp in IR has
    // the lowest schedule-time so that it doesn't need to be relocated.
    // Check if this is satisfied
    VPUX_THROW_UNLESS(schedOpList.front().startTime == min_time,
                      "First async-execute op in IR doesn't have lowest schedule-time");

    // Sort AsyncExecuteOps according to assigned schedule-time
    schedOpList.sort([](SchedOp a, SchedOp b) {
        return a.startTime < b.startTime;
    });

    // Update placement of AsyncExecuteOps
    mlir::Operation* prevAsyncOp = nullptr;
    for (auto& schedOp : schedOpList) {
        mlir::Operation* asyncOp = depsInfo.getExecuteOpAtIndex(schedOp.opIndex);
        VPUX_THROW_UNLESS(asyncOp != nullptr, "AsyncOp not located based on index");
        if (prevAsyncOp != nullptr) {
            asyncOp->moveAfter(prevAsyncOp);
        }
        prevAsyncOp = asyncOp;
    }
}

}  // namespace

//
// createSortAsyncExecuteOps
//

std::unique_ptr<mlir::Pass> vpux::IERT::createSortAsyncExecuteOpsPass(Logger log) {
    return std::make_unique<SortAsyncExecuteOpsPass>(log);
}
