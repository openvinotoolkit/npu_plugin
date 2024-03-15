//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/core/cycle_cost_info.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/passes.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/utils/dma.hpp"

namespace vpux {
namespace VPURT {

SmallVector<size_t> getSubTasksStartTime(const SmallVector<size_t>& subTasksCost, size_t startTime, size_t queueCount);

struct TaskConfig {
    VPURT::TaskOp taskOp;
    SmallVector<int64_t> virtBarrierWaits;
    SmallVector<int64_t> virtBarrierUpdates;
    size_t cycleCost = 0;
    size_t cycleStart = 0;
    SmallVector<size_t> subTasksCycleCost;
    SmallVector<size_t> subTasksCycleStart;

    TaskConfig() = default;
    TaskConfig(VPURT::TaskOp taskOp, SmallVector<int64_t>& virtBarrierWaitVec,
               SmallVector<int64_t>& virtBarrierUpdateVec, size_t cost, SmallVector<size_t>& subTasksCost);
};

using TaskConfigVec = SmallVector<TaskConfig, 1>;

// Class for storing information about barrier, its current
// producer count and cycle when it was last updated
// When producer count gets decremented to 0, cycle value corresponds
// to moment at which barrier was released
class BarrierConfig {
private:
    size_t _producerCount = 0;
    size_t _lastCycleUpdate = 0;

public:
    bool isReleased() {
        return _producerCount == 0;
    }

    void addProducer() {
        _producerCount++;
    }

    void decrementAtCycle(size_t cycle) {
        _lastCycleUpdate = std::max(_lastCycleUpdate, cycle);
        _producerCount--;
    }

    size_t getReleaseCycle() {
        VPUX_THROW_UNLESS(_producerCount == 0, "Barrier was not yet released");
        return _lastCycleUpdate;
    }
};

// Class for simulating inference with support for maintaining cycles of each queue type
// It allows to determine cycleBegin/End of each task
class InferenceExecutionSimulator {
public:
    InferenceExecutionSimulator(Logger log, mlir::func::FuncOp funcOp, CycleCostInfo& cycleCostInfo);

    void runSim();
    std::map<TaskQueueType, TaskConfigVec> getQueueTaskMap();
    TaskConfigVec getTaskCycleConfig();
    TaskConfigVec getTaskCycleConfig(VPU::ExecutorKind execKind);
    size_t getInferenceLatencyInCycles();
    void updateCyclesInIR();
    double getDPUTotalEnergy();
    double getSHAVETotalEnergy();

private:
    void parseFunc();

    // Store information about all tasks that are assigned to a given
    // queue of execution which is identified by executor type and its specific settings
    // like NCE and cluster number of DMA and port&channel numbers
    std::map<TaskQueueType, TaskConfigVec> _queueTasksMap;
    // Map with virtual barrier IDs and its configuration
    mlir::DenseMap<int64_t, BarrierConfig> _virtBarriers;
    // Map of executor kind which on HW side have more engines but are not
    // identifiable on IR or blob level and are dispatched only during inference
    // Such case is on VPU2 for 2 ActShave engines on single cluster. Tasks
    // are not assigned to it by compiler but are dispatched during inference
    // based on availability. Compiler needs to know this to correctly model
    // this parallelism as this is not modeled on TaskOp level
    mlir::DenseMap<VPU::ExecutorKind, int64_t> _numOfExecutorQueuesForWhichAssignmentIsAtInference;

    Logger _log;
    mlir::func::FuncOp _funcOp;
    int64_t _dpuCount = 1;
    CycleCostInfo& _cycleCostInfo;
};

}  // namespace VPURT
}  // namespace vpux
