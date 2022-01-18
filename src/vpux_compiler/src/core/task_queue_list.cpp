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

#include "vpux/compiler/core/task_queue_list.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

vpux::TaskQueueList::TaskQueueList(mlir::FuncOp func): _log(Logger::global().nest("task-queue-list", 0)) {
    _log.trace("Filling in task queue list..");

    func.walk([&](mlir::async::ExecuteOp executeOp) {
        uint32_t numUnits = 0;
        auto executor = vpux::IERT::IERTDialect::getExecutor(executeOp, numUnits);

        _tasksQueues[executor].push_back(executeOp);
    });

    _log.trace("Task queue list is filled:");
    for (auto& taskQueuePair : _tasksQueues) {
        _log.nest().trace("- {0} executor queue size {1}", taskQueuePair.first.getName(), taskQueuePair.second.size());
    }
}

bool vpux::TaskQueueList::hasQueue(IndexedSymbolAttr executor) {
    return _tasksQueues.find(executor) != _tasksQueues.end();
}

vpux::TaskQueueList::TaskQueue& vpux::TaskQueueList::getQueue(IndexedSymbolAttr executor) {
    VPUX_THROW_UNLESS(hasQueue(executor), "No task queue for executor {0}", executor);

    return _tasksQueues[executor];
}

size_t vpux::TaskQueueList::size() {
    return _tasksQueues.size();
}
