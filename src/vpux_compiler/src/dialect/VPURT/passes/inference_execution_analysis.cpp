//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/cost_model.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/inference_execution_simulator.hpp"
#include "vpux/compiler/dialect/VPURT/passes.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/utils/dma.hpp"
#include "vpux/compiler/utils/strings.hpp"

#include "vpux/utils/IE/profiling.hpp"
#include "vpux/utils/plugin/profiling_parser.hpp"

using namespace vpux;

namespace {

profiling::TaskInfo makeTaskInfo(VPURT::TaskOp taskOp, double startTimeNs, double durationNs, Logger log) {
    profiling::TaskInfo taskInfo;
    auto execKind = taskOp.getExecutorKind();
    switch (execKind) {
    case VPU::ExecutorKind::DMA_NN:
        taskInfo.exec_type = profiling::TaskInfo::ExecType::DMA;
        break;
    case VPU::ExecutorKind::NCE:
        taskInfo.exec_type = profiling::TaskInfo::ExecType::DPU;
        break;
    case VPU::ExecutorKind::SHAVE_ACT:
        taskInfo.exec_type = profiling::TaskInfo::ExecType::SW;
        break;
    case VPU::ExecutorKind::SHAVE_UPA:
        taskInfo.exec_type = profiling::TaskInfo::ExecType::UPA;
        break;

    default:
        log.warning("Not supported executor type - '{0}", execKind);
        taskInfo.exec_type = profiling::TaskInfo::ExecType::NONE;
        break;
    }

    auto op = taskOp.getInnerTaskOp();
    VPUX_THROW_WHEN(op == nullptr, "TaskOp with no op inside - '{0}'", taskOp->getLoc());
    auto taskName = stringifyLocation(op->getLoc());
    auto length = taskName.copy(taskInfo.name, sizeof(taskInfo.name) - 1);
    taskInfo.name[length] = '\0';

    auto layerType = std::string(op->getName().getStringRef().data());
    length = layerType.copy(taskInfo.layer_type, sizeof(taskInfo.layer_type) - 1);
    taskInfo.layer_type[length] = '\0';

    taskInfo.start_time_ns = startTimeNs;
    taskInfo.duration_ns = durationNs;

    return taskInfo;
}

double convertCyclesToNanoSeconds(int64_t cycles, double freqInMHz) {
    return static_cast<double>(cycles * 1000) / freqInMHz;
}

void createScheduleTraceEventFile(const SmallVector<VPURT::TaskConfig>& tasksCycleConfig, double freqInMHz,
                                  StringRef fileName, Logger log) {
    std::ofstream out_stream(fileName.str());
    VPUX_THROW_UNLESS(out_stream.good(), "File for schedule traces not created correctly");

    VPUX_THROW_WHEN(tasksCycleConfig.empty(), "Empty cycle config array");

    std::vector<profiling::TaskInfo> tasks;
    for (auto& taskConfig : tasksCycleConfig) {
        auto taskOp = taskConfig.taskOp;
        auto cycleBegin = taskConfig.cycleStart;
        auto cycleCost = taskConfig.cycleCost;
        VPUX_THROW_UNLESS(cycleBegin >= 0 && cycleCost > 0,
                          "Invalid cycle setting (cycleBegin - '{0}', cycleCost - '{1}') for op - '{2}'", cycleBegin,
                          cycleCost, taskOp->getLoc());

        tasks.push_back(makeTaskInfo(taskOp, convertCyclesToNanoSeconds(cycleBegin, freqInMHz),
                                     convertCyclesToNanoSeconds(cycleCost, freqInMHz), log));
    }

    printProfilingAsTraceEvent(tasks, {}, out_stream);
}

class InferenceExecutionAnalysisPass final :
        public VPURT::InferenceExecutionAnalysisBase<InferenceExecutionAnalysisPass> {
public:
    explicit InferenceExecutionAnalysisPass(const std::string& compileSchedTraceFileName, Logger log)
            : _compileSchedTraceFileName(compileSchedTraceFileName) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    std::string _compileSchedTraceFileName;
};

void InferenceExecutionAnalysisPass::safeRunOnFunc() {
    auto funcOp = getOperation();
    auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();

    VPURT::InferenceExecutionSimulator infSim(_log, funcOp);

    _log.trace("Start inference schedule simulation and update cycles");

    infSim.runSim();

    auto tasksCycleConfig = infSim.getTaskCycleConfig();

    // All cycles returned from VPUNN cost model are provided with respect to DPU clock
    // Get frequency information to allow translation to time units
    double freqInMHz = 0;
    if (auto nceOp = IE::getAvailableExecutor(moduleOp, VPU::ExecutorKind::NCE)) {
        freqInMHz = nceOp.getProcessorFrequency().getValueAsDouble();
    }
    VPUX_THROW_WHEN(freqInMHz == 0, "Frequency was not configured");

    auto estimatedLatencyInUs = convertCyclesToNanoSeconds(infSim.getInferenceLatencyInCycles(), freqInMHz) / 1000;
    _log.info("Estimated inference latency: {0}us", estimatedLatencyInUs);
    if (auto tasksCountWithInvalidCost = infSim.getNumberfOfTasksWithInvalidCost()) {
        _log.warning("There are {0} tasks with invalid cost, estimation might not be valid", tasksCountWithInvalidCost);

        _log.warning("Invalid cost for:");
        for (auto& layerWithInvalidCost : infSim.getLayersWithInvalidCost()) {
            _log.nest().warning("{0}", layerWithInvalidCost);
        }
    }

    VPUX_THROW_WHEN(_compileSchedTraceFileName.empty(), "Empty compile time schedule trace file");

    createScheduleTraceEventFile(tasksCycleConfig, freqInMHz, _compileSchedTraceFileName, _log);
}

}  // namespace

//
// createInferenceExecutionAnalysisPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createInferenceExecutionAnalysisPass(std::string compileSchedTraceFileName,
                                                                              Logger log) {
    return std::make_unique<InferenceExecutionAnalysisPass>(compileSchedTraceFileName, log);
}
