//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPURT/inference_execution_simulator.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/cost_model.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/passes.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/utils/dma.hpp"

using namespace vpux;

namespace {

// Get an ID that will identify given execution queue
// This ID does not correspond to any value in HW. It is just used
// by compiler to differentiate queues by some abstract value that
// can be composed of cluster index in case of NCE or ActShave or
// port and channel in case of DMA
int64_t getQueueId(VPURT::TaskOp taskOp) {
    auto* op = taskOp.getInnerTaskOp();
    if (auto dmaTask = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(op)) {
        return VPURT::getDMAQueueIdEncoding(dmaTask.getPortVal(), dmaTask.getChannelType());
    } else if (auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(op)) {
        const auto& dpuTasks = nceOp.variants().getOps<VPUIP::DPUTaskOp>();
        VPUX_THROW_UNLESS(!dpuTasks.empty(), "Encountered op '{0}' with empty dpu list", op->getLoc());
        auto dpuTask = *(dpuTasks.begin());
        return dpuTask.cluster_id().value_or(0);
    } else if (auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(op)) {
        return swKernelOp.tileIndex().value_or(0);
    }

    return 0;
}

// Helper function used by logger to create clear string about executor instance
// that a given task has executed on. Examples:
// - DMA_NN[port = 0, channel = DDR]
// - NCE[cluster = 0]
// - SHAVE_ACT[cluster = 0]
std::string getTaskQueueInfoString(VPURT::TaskQueueType taskType, VPURT::TaskOp taskOp) {
    std::string infoStr = stringifyEnum(taskType.type).data();

    auto* op = taskOp.getInnerTaskOp();

    if (auto dmaTask = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(op)) {
        infoStr += "[port = " + std::to_string(dmaTask.getPortVal());
        auto channel = dmaTask.getChannelType();
        if (channel.has_value()) {
            std::string channelName = stringifyEnum(channel.value()).data();
            infoStr += ", channel = " + channelName;
        }
        infoStr += "]";
    } else if (mlir::isa<VPUIP::NCEClusterTaskOp, VPUIP::SwKernelOp>(op)) {
        infoStr += "[cluster = " + std::to_string(taskType.id) + "]";
    }

    return infoStr;
}

}  // namespace

vpux::VPURT::TaskConfig::TaskConfig(VPURT::TaskOp op, SmallVector<int64_t>& virtBarrierWaitVec,
                                    SmallVector<int64_t>& virtBarrierUpdateVec, int64_t cost)
        : taskOp(op), virtBarrierWaits(virtBarrierWaitVec), virtBarrierUpdates(virtBarrierUpdateVec), cycleCost(cost) {
}

vpux::VPURT::InferenceExecutionSimulator::InferenceExecutionSimulator(Logger log, mlir::func::FuncOp funcOp)
        : _log(log), _funcOp(funcOp) {
    auto module = funcOp->getParentOfType<mlir::ModuleOp>();

    _archKind = VPU::getArch(module);
    _costModel = VPU::createCostModel(_archKind);

    if (auto nceOp = IE::getAvailableExecutor(module, VPU::ExecutorKind::NCE)) {
        // In case of ActShave tasks on a single cluster compiler does not assign
        // it to a dedicated engine instance as it is dispatched only at inference
        // based on engine availability. Nevertheless simulator needs to know this
        // to correctly model those queues and track cycles
        if (auto shaveActExec = nceOp.getSubExecutor(VPU::ExecutorKind::SHAVE_ACT)) {
            _numOfExecutorQueuesForWhichAssignmentIsAtInference[VPU::ExecutorKind::SHAVE_ACT] = shaveActExec.count();
        }
        _dpuCount = nceOp.getSubExecutor(VPU::ExecutorKind::DPU).count();
    }
    // Parse model and gather information about barrier, tasks and their cycle cost
    parseFunc();
}

int64_t vpux::VPURT::InferenceExecutionSimulator::getTaskCycleCost(VPURT::TaskOp taskOp) {
    int64_t cost;
    auto* op = taskOp.getInnerTaskOp();
    std::string layerTypeStr = op->getName().getStringRef().str();
    if (mlir::isa<VPUIP::DMATypeOpInterface>(op)) {
        auto dmaLayer = mlir::dyn_cast<VPUIP::LayerOpInterface>(op);
        cost = getDMACost(dmaLayer.getInputs()[0], dmaLayer.getOutputs()[0], _archKind, _costModel);
    } else if (auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(op)) {
        layerTypeStr += "." + swKernelOp.kernelFunction().getLeafReference().str();
        cost = calculateShaveActCycles(swKernelOp, _costModel, _archKind);
    } else if (auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(op)) {
        cost = calculateNceCycles(nceOp, _costModel, _archKind, _log, _dpuCount);
    } else {
        // Add support for recalculating cost of tasks using VPUNN cost model
        // For now trust what is present in IR. In future cycleBegin/End will
        // be deprecated, see - E#86678
        auto cycleBeginAttr = taskOp->getAttr(cycleBegin);
        auto cycleEndAttr = taskOp->getAttr(cycleEnd);
        // For now instead of recalculating cost if layer has no cycle attributes
        // return just 1
        if (cycleBeginAttr == nullptr || cycleEndAttr == nullptr) {
            _log.warning("Layer has no cycleBegin/End attributes. Assume cycleCost = 1, '{0}'", taskOp->getLoc());
            _numOfTasksWithInvalidCost++;
            return 1;
        }

        cost = cycleEndAttr.cast<mlir::IntegerAttr>().getInt() - cycleBeginAttr.cast<mlir::IntegerAttr>().getInt();
    }

    // Use cost = 1 for layers with invalid cost as this is more user friendly when
    // visualizing schedule
    if (cost <= 1 || cost >= VPU::INVALID_COST_BASE) {
        _numOfTasksWithInvalidCost++;
        _layersWithInvalidCost.insert(layerTypeStr);
        _log.warning("Layer {0} has invalid cost - '{1}'. Assume cycleCost = 1, '{2}'", layerTypeStr, cost,
                     taskOp->getLoc());
        return 1;
    }
    return cost;
}

void vpux::VPURT::InferenceExecutionSimulator::parseFunc() {
    DenseMap<mlir::Operation*, int64_t> barrierOpToVIdMap;
    // Assign virtual IDs to each barrier operation. Whole simulation will
    // use those VIDs when tracking dependencies
    int64_t vid = 0;
    _funcOp->walk([&](mlir::Operation* op) {
        if (mlir::isa<VPURT::DeclareVirtualBarrierOp, VPURT::ConfigureBarrierOp>(op)) {
            VPUX_THROW_WHEN(vid > std::numeric_limits<unsigned short>::max(), "Barrier virtual id '{0}' is too large ",
                            vid);

            barrierOpToVIdMap[op] = vid++;
        }
    });

    auto getVirtualBarrierDeps = [&](mlir::ValueRange barriers) {
        SmallVector<int64_t> vIds;
        for (const auto bar : barriers) {
            const auto virtBarIt = barrierOpToVIdMap.find(bar.getDefiningOp());
            VPUX_THROW_WHEN(virtBarIt == barrierOpToVIdMap.end(), "Barrier at '{0}' was not assigned virtual ID",
                            bar.getLoc());

            const auto vid = virtBarIt->second;
            vIds.push_back(vid);
        }
        return vIds;
    };

    // Scan whole model and assign tasks to their queues
    _funcOp->walk([&](VPURT::TaskOp taskOp) {
        auto virtBarrierWaits = getVirtualBarrierDeps(taskOp.getWaitBarriers());
        auto virtBarrierUpdates = getVirtualBarrierDeps(taskOp.getUpdateBarriers());
        auto cost = getTaskCycleCost(taskOp);
        TaskConfig taskCfg(taskOp, virtBarrierWaits, virtBarrierUpdates, cost);

        // If a task produces any barriers update
        // barrier information to determine final value of producer count
        for (auto virtBarrierId : taskCfg.virtBarrierUpdates) {
            _virtBarriers[virtBarrierId].addProducer();
        }

        VPURT::TaskQueueType queueType;
        queueType.type = taskOp.getExecutorKind();
        queueType.id = getQueueId(taskOp);

        // Add a task to a container for a given queue type
        _queueTasksMap[queueType].push_back(taskCfg);
    });
}

// Class for maintaining information about a queue of execution
// To be used by InferenceExecutionSimulator::runSim() function to track
// the progress of execution
class QueueState {
public:
    QueueState(): _taskIdx(0) {
    }

    QueueState(size_t numOfRuntimeDispatchedExecutors): _taskIdx(0) {
        VPUX_THROW_UNLESS(numOfRuntimeDispatchedExecutors > 0,
                          "Number of executors need to be larger then 0, got '{0}'", numOfRuntimeDispatchedExecutors);
        _cycle.resize(numOfRuntimeDispatchedExecutors);
    }

    size_t getCurrentTaskIdx() {
        return _taskIdx;
    }

    size_t getCycle() {
        // Get minimal cycle value. This aligs with HW behavior which will
        // pick a task from a queue to an executor currently being free
        return *std::min_element(_cycle.begin(), _cycle.end());
    }

    void progressQueueToCycle(size_t newCycle) {
        // Once task from a queue gets executed update cycle state of a queue
        // and increment task index
        auto cycleItr = std::min_element(_cycle.begin(), _cycle.end());
        VPUX_THROW_WHEN(newCycle < *cycleItr, "New cycle '{0}' is smaller then exisitng '{1}'", newCycle, *cycleItr);
        *cycleItr = newCycle;
        _taskIdx++;
    }

private:
    // Vector maintaining cycle state of given queue
    // When task is executed cycle is being updated to cycleEnd
    // of this task. This member is a vector because there can be mutliple
    // executors of the same type that are not assigned or distinguishable by compiler
    // and are dispatched at runtime.
    SmallVector<size_t> _cycle;

    // Index of next tast to be executed on given queue
    size_t _taskIdx;
};

void vpux::VPURT::InferenceExecutionSimulator::runSim() {
    // Create a map of all encountered queue types and initialize queue state
    // based on information on how many executors there are of exactly the same type from
    // compiler point of view that are dispatched at runtime
    std::map<VPURT::TaskQueueType, QueueState> queueStateMap;
    for (auto& queue : _queueTasksMap) {
        auto numOfCycleQueuesToTrack = 1;
        if (_numOfExecutorQueuesForWhichAssignmentIsAtInference.find(queue.first.type) !=
            _numOfExecutorQueuesForWhichAssignmentIsAtInference.end()) {
            numOfCycleQueuesToTrack = _numOfExecutorQueuesForWhichAssignmentIsAtInference[queue.first.type];
        }
        queueStateMap[queue.first] = QueueState(numOfCycleQueuesToTrack);
    }

    // Run simulation to update cycleBegin/End of each task
    // Iterate over all queues and check next task on queue. If dependencies are satisfied
    // task can be executed and cycle state on this queue needs to be updated. If there are any
    // update barriers their counter will be decremented
    // If no task was executed on all queues that stop the simulation
    bool progressed = true;
    while (progressed) {
        progressed = false;
        for (auto& queueTypeTasks : _queueTasksMap) {
            auto& queueType = queueTypeTasks.first;
            auto& queueTasks = queueTypeTasks.second;

            auto index = queueStateMap[queueType].getCurrentTaskIdx();
            if (index >= queueTasks.size()) {
                continue;
            }

            auto cost = queueTasks[index].cycleCost;

            bool waitingForDependency = false;
            size_t cycleBegin = queueStateMap[queueType].getCycle();

            // Check all wait barrierss. If all are satisified task can be executed
            // CycleBegin value needs to take into account at what cycle last barrier
            // (from cycle point of view) was released
            for (auto waitVirtBarrierId : queueTasks[index].virtBarrierWaits) {
                if (!_virtBarriers[waitVirtBarrierId].isReleased()) {
                    waitingForDependency = true;
                    break;
                }
                cycleBegin = std::max(cycleBegin, _virtBarriers[waitVirtBarrierId].getReleaseCycle());
            }

            if (waitingForDependency) {
                continue;
            }

            // All dependencies satisified - task ready to execute
            size_t cycleEnd = cycleBegin + cost;

            _log.trace("Run {0}[{1}]: cost: {2} cycleBegin: {3} cycleEnd: {4}",
                       getTaskQueueInfoString(queueType, queueTasks[index].taskOp), index, cost, cycleBegin, cycleEnd);

            // Update all update barriers of this task. Decrement their counter
            // and pass information at what cycle this update has happened. This is later needed
            // to understand at what cycle barrier was released
            for (auto updateVirtBarrierId : queueTasks[index].virtBarrierUpdates) {
                VPUX_THROW_WHEN(_virtBarriers[updateVirtBarrierId].isReleased(), "Barrier {0} was already released",
                                updateVirtBarrierId);

                _virtBarriers[updateVirtBarrierId].decrementAtCycle(cycleEnd);

                _log.nest().trace("Decrement virt barrier {0}{1}", updateVirtBarrierId,
                                  _virtBarriers[updateVirtBarrierId].isReleased() ? " - barrier released" : "");
            }

            // Task has executed on this queue. Update queue state with new cycle
            queueStateMap[queueType].progressQueueToCycle(cycleEnd);
            queueTasks[index].cycleStart = cycleBegin;
            progressed = true;
        }
    }

    // Check if all operations were processed - each queue state index should correspond to
    // the number of tasks in IR that were to be processed by this queue
    // If this is not the case then simulation of execution most likely encountered incorrect
    // dependencies setting which caused a hang
    for (auto& queueTypeTasks : _queueTasksMap) {
        auto& queueType = queueTypeTasks.first;
        auto& queueTasks = queueTypeTasks.second;

        auto index = queueStateMap[queueType].getCurrentTaskIdx();
        VPUX_THROW_WHEN(index != queueTasks.size(),
                        "Not all operations were processed for {0}, index - {1}, queue size - {2}", queueType.type,
                        queueType.id, queueTasks.size());
    }
}

SmallVector<VPURT::TaskConfig> vpux::VPURT::InferenceExecutionSimulator::getTaskCycleConfig() {
    VPUX_THROW_WHEN(_queueTasksMap.empty(), "Queue task map not initialized");

    SmallVector<TaskConfig> allQueueTaskConfig;

    for (auto& queueTypeTasks : _queueTasksMap) {
        auto& queueTasks = queueTypeTasks.second;

        allQueueTaskConfig.insert(allQueueTaskConfig.end(), queueTasks.begin(), queueTasks.end());
    }

    return allQueueTaskConfig;
}

SmallVector<VPURT::TaskConfig> vpux::VPURT::InferenceExecutionSimulator::getTaskCycleConfig(
        VPU::ExecutorKind execKind) {
    VPUX_THROW_WHEN(_queueTasksMap.empty(), "Queue task map not initialized");

    SmallVector<TaskConfig> allQueueTaskConfig;

    for (auto& queueTypeTasks : _queueTasksMap) {
        if (queueTypeTasks.first.type != execKind) {
            continue;
        }

        auto& queueTasks = queueTypeTasks.second;

        allQueueTaskConfig.insert(allQueueTaskConfig.end(), queueTasks.begin(), queueTasks.end());
    }

    return allQueueTaskConfig;
}

size_t vpux::VPURT::InferenceExecutionSimulator::getNumberfOfTasksWithInvalidCost() {
    return _numOfTasksWithInvalidCost;
}

std::set<std::string> vpux::VPURT::InferenceExecutionSimulator::getLayersWithInvalidCost() {
    return _layersWithInvalidCost;
}

int64_t vpux::VPURT::InferenceExecutionSimulator::getInferenceLatencyInCycles() {
    int64_t latency = 0;
    for (auto& queueTypeTasks : _queueTasksMap) {
        auto& queueTasks = queueTypeTasks.second;

        auto lastTaskCycleEnd = queueTasks.back().cycleStart + queueTasks.back().cycleCost;
        if (lastTaskCycleEnd > latency) {
            latency = lastTaskCycleEnd;
        }
    }
    return latency;
}

void vpux::VPURT::InferenceExecutionSimulator::updateCyclesInIR() {
    for (auto& queueTypeTasks : _queueTasksMap) {
        auto& queueTasks = queueTypeTasks.second;
        for (auto& task : queueTasks) {
            task.taskOp->setAttr(cycleBegin, getIntAttr(task.taskOp->getContext(), task.cycleStart));
            task.taskOp->setAttr(cycleEnd, getIntAttr(task.taskOp->getContext(), task.cycleStart + task.cycleCost));
        }
    }
}
