//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/cost_model.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/passes.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/utils/dma.hpp"

namespace vpux {
namespace VPURT {

struct TaskConfig {
    VPURT::TaskOp taskOp;
    SmallVector<int64_t> virtBarrierWaits;
    SmallVector<int64_t> virtBarrierUpdates;
    int64_t cycleCost = -1;
    int64_t cycleStart = -1;

    TaskConfig(VPURT::TaskOp taskOp, SmallVector<int64_t>& virtBarrierWaitVec,
               SmallVector<int64_t>& virtBarrierUpdateVec, int64_t cost);
};

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
    InferenceExecutionSimulator(Logger log, mlir::func::FuncOp funcOp);

    void runSim();
    SmallVector<TaskConfig> getTaskCycleConfig();
    SmallVector<TaskConfig> getTaskCycleConfig(VPU::ExecutorKind execKind);
    size_t getNumberfOfTasksWithInvalidCost();
    std::set<std::string> getLayersWithInvalidCost();
    int64_t getInferenceLatencyInCycles();
    void updateCyclesInIR();
    int64_t getTaskCycleCost(VPURT::TaskOp taskOp);

private:
    void parseFunc();

    // Store information about all tasks that are assigned to a given
    // queue of execution which is identified by executor type and its specific settings
    // like NCE and cluster number of DMA and port&channel numbers
    std::map<VPURT::TaskQueueType, SmallVector<TaskConfig>> _queueTasksMap;
    // Map with virtual barrier IDs and its configuration
    mlir::DenseMap<int64_t, BarrierConfig> _virtBarriers;
    // Map of executor kind which on HW side have more engines but are not
    // identifiable on IR or blob level and are dispatched only during inference
    mlir::DenseMap<VPU::ExecutorKind, int64_t> _numOfExecutorQueuesForWhichAssignmentIsAtInference;

    Logger _log;
    mlir::func::FuncOp _funcOp;
    VPU::ArchKind _archKind;
    std::shared_ptr<VPUNN::VPUCostModel> _costModel;
    int64_t _dpuCount = 1;
    size_t _numOfTasksWithInvalidCost = 0;
    std::set<std::string> _layersWithInvalidCost;
};

}  // namespace VPURT
}  // namespace vpux
