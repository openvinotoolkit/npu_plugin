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

#include "vpux/compiler/core/runtime_simulator.hpp"

using namespace vpux;

RuntimeSimulator::RuntimeSimulator(mlir::MLIRContext* ctx, mlir::FuncOp func, Logger log, int64_t numDmaEngines,
                                   size_t numRealBarriers)
        : _ctx(ctx),
          _func(func),
          _log(log),
          _numDmaEngines(numDmaEngines),
          _numRealBarriers(numRealBarriers),
          _active_barrier_table(),
          _real_barrier_list() {
}

void RuntimeSimulator::init() {
    _real_barrier_list.clear();

    Logger::global().error("Populating _real_barrier_list");
    for (size_t i = 0; i < _numRealBarriers; i++) {
        _real_barrier_list.push_back(i);
    }
}

int64_t RuntimeSimulator::getVirtualId(VPURT::ConfigureBarrierOp op) {
    return checked_cast<int64_t>(op->getAttr(virtualIdAttrName).cast<mlir::IntegerAttr>().getInt());
}

void RuntimeSimulator::buildTaskLists() {
    _log.trace("Building task lists");

    auto getTaskInfo = [&](VPURT::TaskOp taskOp) {
        TaskInfo taskInfo(taskOp);
        for (auto waitBarrier : taskOp.waitBarriers()) {
            if (auto barrierOp = mlir::dyn_cast<VPURT::ConfigureBarrierOp>(waitBarrier.getDefiningOp())) {
                const auto virtualId = getVirtualId(barrierOp);
                taskInfo.waitBarriers.push_back(virtualId);
            }
        }
        for (auto updateBarrier : taskOp.updateBarriers()) {
            if (auto barrierOp = mlir::dyn_cast<VPURT::ConfigureBarrierOp>(updateBarrier.getDefiningOp())) {
                const auto virtualId = getVirtualId(barrierOp);
                taskInfo.updateBarriers.push_back(virtualId);
            }
        }
        return taskInfo;
    };

    // The task lists have to be populated in the same order as during the serialization phase
    // to ensure that the correct simulation occurs
    _func.walk([&](VPURT::DeclareVirtualBarrierOp barrierOp) {
        Logger::global().error("Adding Barrier ID {0}", barrierOp->getAttr("id").cast<mlir::IntegerAttr>().getInt());
        std::cout << barrierOp->getAttr("id").cast<mlir::IntegerAttr>().getInt() << std::endl;
        if (barrierOp->getAttr("id").cast<mlir::IntegerAttr>().getInt() > 0)
            _barrierOps.push_back(barrierOp);
    });

    _func.walk([&](VPURT::TaskOp taskOp) {
        Logger::global().error("Scheduling number {0} ", taskOp->getAttr("SchedulingNumber"));
    });

    _func.walk([&](VPURT::TaskOp taskOp) {
        auto& block = taskOp.body().getBlocks().front();
        auto wrappedTaskOp = block.begin();
        switch (taskOp.getExecutorKind()) {
        // case VPU::ExecutorKind::UPADMA:
        case VPU::ExecutorKind::DMA_NN: {
            int64_t port = 0;
            if (auto dmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(wrappedTaskOp)) {
                port = dmaOp.port();
            } else if (auto compressedDmaOp = mlir::dyn_cast<VPUIP::CompressedDMAOp>(wrappedTaskOp)) {
                port = compressedDmaOp.port();
            } else {
                VPUX_THROW("Could not cast to DMA task");
            }
            VPUX_THROW_UNLESS(port < MAX_DMA_ENGINES,
                              "NNDMAOp port value ({0}) larger than maximum number of engines ({1})", port,
                              MAX_DMA_ENGINES);
            Logger::global().error("Adding DMA scheduling number {0} ", taskOp->getAttr("SchedulingNumber"));
            _dmaTasks[port].push_back(getTaskInfo(taskOp));
            break;
        }
        case VPU::ExecutorKind::NCE: {
            auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(wrappedTaskOp);
            VPUX_THROW_UNLESS(nceOp != nullptr, "Could not cast to NCE task");
            _nceTasks.push_back(getTaskInfo(taskOp));
            break;
        }
        // TODO: should we introduce _swTask?
        // case VPU::ExecutorKind::ACTShave:
        case VPU::ExecutorKind::SHAVE_UPA: {
            Logger::global().error("Adding UPA scheduling number {0} ", taskOp->getAttr("SchedulingNumber"));
            _upaTasks.push_back(getTaskInfo(taskOp));
            break;
        }
        default:
            VPUX_THROW("Unsupported task type '{0}'", taskOp.getExecutorKind());
        }
    });
}

bool RuntimeSimulator::assignPhysicalIDs() {
    _log.trace("Running barrier simulator");

    init();
    buildTaskLists();
    getAllBarriersProducersAndConsumers();
    computeOpIndegree();
    computeOpOutdegree();

    size_t barrier = 0;
    size_t nce = 0;
    size_t upa = 0;
    std::array<size_t, MAX_DMA_ENGINES> dma = {0};

    std::cout << "barrier number is" << _barrierOps.size() << std::endl;

    while (barrier < _barrierOps.size() || dma[0] < _dmaTasks[0].size() || dma[1] < _dmaTasks[1].size() ||
           nce < _nceTasks.size() || upa < _upaTasks.size()) {
        _log.nest(2).trace("BAR: {0} / {1}; DMA: {2} / {3}, {4} / {5}; NCE: {6} / {7}; UPA: {8} / {9}", barrier,
                           _barrierOps.size(), dma[0], _dmaTasks[0].size(), dma[1], _dmaTasks[1].size(), nce,
                           _nceTasks.size(), upa, _upaTasks.size());

        std::cout << "Starting runtime simulation" << std::endl;
        bool progressed = false;
        while (!_dmaTasks[0].empty() || !_nceTasks.empty() || !_barrierOps.empty() || !_upaTasks.empty()) {
            progressed = false;
            progressed |= fillBarrierTasks(_barrierOps);
            progressed |= processTasks(_dmaTasks[0]);
            progressed |= processTasks(_nceTasks);
            progressed |= processTasks(_upaTasks);

            if (!progressed) {
                return false;
            }
        }

        std::cout << "assignPhysicalIDs" << std::endl;
        for (auto& barrier : _virtualToPhysicalBarrierMap) {
            // Logger::global().error("Virtual Barrier ID {0} has physical ID {1}", barrier.first->getAttr("id"),
            //                        barrier.second.first);
            std::cout << barrier.first->getAttr("id").cast<mlir::IntegerAttr>().getInt() << " " << barrier.second.first
                      << std::endl;
        }
    }

    return true;
}

void RuntimeSimulator::acquireRealBarrier(VPURT::DeclareVirtualBarrierOp btask) {
    assert(!_real_barrier_list.empty());
    size_t real = _real_barrier_list.front();

    _real_barrier_list.pop_front();

    assert(_active_barrier_table.size() < _numRealBarriers);

    auto in_itr = in_degree_map_.find(btask.getOperation());
    auto out_itr = out_degree_map_.find(btask.getOperation());

    assert((in_itr != in_degree_map_.end()) && (out_itr != out_degree_map_.end()));

    assert(_active_barrier_table.find(btask.getOperation()) == _active_barrier_table.end());

    Logger::global().error("Assigning Virtual Barrier ID {0} with physical ID {1}", btask->getAttr("id"), real);
    Logger::global().error("Inserting barrier with Id {0} in the _active_barrier_table ", btask->getAttr("id"));

    _virtualToPhysicalBarrierMap.insert(std::make_pair(
            btask.getOperation(),
            std::make_pair(real, checked_cast<int64_t>(btask->getAttr("id").cast<mlir::IntegerAttr>().getInt()))));
    _active_barrier_table.insert(
            std::make_pair(btask.getOperation(), active_barrier_info_t(real, in_itr->second, out_itr->second)));
}

bool RuntimeSimulator::isTaskReady(VPURT::TaskOp taskOp) {
    Logger::global().error("Is task with scheduling number {0} ready?", taskOp->getAttr("SchedulingNumber"));
    // wait barriers //
    for (const auto waitBarrier : taskOp.waitBarriers()) {
        if (auto barrierOp = waitBarrier.getDefiningOp()) {
            active_barrier_table_iterator_t aitr = _active_barrier_table.find(barrierOp);

            if ((aitr == _active_barrier_table.end()) ||
                ((aitr->second).in_degree_ > 0))  // double check this condition
            {
                return false;
            }
        }
    }
    // update barriers //
    for (const auto updateBarrier : taskOp.updateBarriers()) {
        if (auto barrierOp = updateBarrier.getDefiningOp()) {
            active_barrier_table_iterator_t aitr = _active_barrier_table.find(barrierOp);
            Logger::global().error("Looking for Barrier ID {1} in the _active_barrier_table ",
                                   barrierOp->getAttr("id"));
            if (aitr == _active_barrier_table.end()) {
                return false;
            }
        }
    }
    return true;
}

void RuntimeSimulator::processTask(VPURT::TaskOp task) {
    // assert(isTaskReady(task));
    Logger::global().error("Now processing task with scheduling number {0}", task->getAttr("SchedulingNumber"));

    active_barrier_table_iterator_t aitr;

    // wait barrier
    for (const auto waitBarrier : task.waitBarriers()) {
        if (auto barrierOp = waitBarrier.getDefiningOp()) {
            aitr = _active_barrier_table.find(barrierOp);
            assert(aitr != _active_barrier_table.end());

            Logger::global().error(
                    "Found the wait barrier for task with scheduling number {0} in the active barrier table",
                    task->getAttr("SchedulingNumber"));

            active_barrier_info_t& barrier_info = aitr->second;
            assert(barrier_info.in_degree_ == 0UL);
            assert(barrier_info.out_degree_ > 0UL);

            Logger::global().error("Decrmenting the out degree of the physical barrier {0}",
                                   barrier_info.real_barrier_);
            barrier_info.out_degree_--;

            if (barrier_info.out_degree_ == 0UL) {
                // return the barrier //
                Logger::global().error("The out degree of the physical barrier {0} is 0", barrier_info.real_barrier_);
                returnRealBarrier(barrierOp);
            }
        }
    }

    // update barriers //
    for (const auto updateBarrier : task.updateBarriers()) {
        if (auto barrierOp = updateBarrier.getDefiningOp()) {
            aitr = _active_barrier_table.find(barrierOp);
            assert(aitr != _active_barrier_table.end());

            active_barrier_info_t& barrier_info = aitr->second;
            assert(barrier_info.in_degree_ > 0UL);
            Logger::global().error("Decrmenting the out degree of the physical barrier {0}",
                                   barrier_info.real_barrier_);
            barrier_info.in_degree_--;
        }
    }
}

void RuntimeSimulator::returnRealBarrier(mlir::Operation* btask) {
    active_barrier_table_iterator_t aitr = _active_barrier_table.find(btask);

    assert(aitr != _active_barrier_table.end());
    assert(((aitr->second).in_degree_ == 0UL) && ((aitr->second).out_degree_ == 0UL));

    assert(_real_barrier_list.size() < _numRealBarriers);

    active_barrier_info_t& abinfo = aitr->second;
    _real_barrier_list.push_back(abinfo.real_barrier_);

    assert(_real_barrier_list.size() <= _numRealBarriers);

    _active_barrier_table.erase(aitr);
}

void RuntimeSimulator::getAllBarriersProducersAndConsumers() {
    // Get all producers and consumers of barriers (NCE,UPA, DMA) only
    auto _barrierOps = to_small_vector(_func.getOps<VPURT::DeclareVirtualBarrierOp>());

    for (auto& barrierOp : _barrierOps) {
        Logger::global().error("Barrier ID {0}", barrierOp->getAttr("id"));

        SmallVector<mlir::Operation*> producers;
        SmallVector<mlir::Operation*> consumers;

        for (auto* userOp : barrierOp->getUsers()) {
            auto opEffects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(userOp);

            VPUX_THROW_WHEN(opEffects == nullptr,
                            "Barrier '{0}' is used by Operation '{1}' without MemoryEffects interface",
                            barrierOp->getLoc(), userOp->getName());

            using MemEffect = mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>;

            SmallVector<MemEffect> valEffects;

            opEffects.getEffectsOnValue(barrierOp.barrier(), valEffects);

            VPUX_THROW_WHEN(
                    valEffects.size() != 1,
                    "Barrier '{0}' must have exactly 1 MemoryEffect per Operation, got '{1}' for Operation '{2}'",
                    barrierOp->getLoc(), valEffects.size(), userOp->getLoc());

            const auto& effect = valEffects.front();

            if (effect.getEffect() == mlir::MemoryEffects::Write::get()) {
                auto task = mlir::dyn_cast<VPURT::TaskOp>(userOp);
                Logger::global().error("Task with scheduling number {0}", task->getAttr("SchedulingNumber"));
                if (task.getExecutorKind() == VPU::ExecutorKind::NCE) {
                    producers.push_back(userOp);
                } else if (task.getExecutorKind() == VPU::ExecutorKind::DMA_NN) {
                    producers.push_back(userOp);
                } else if (task.getExecutorKind() == VPU::ExecutorKind::SHAVE_UPA) {
                    producers.push_back(userOp);
                }
            } else if (effect.getEffect() == mlir::MemoryEffects::Read::get()) {
                auto task = mlir::dyn_cast<VPURT::TaskOp>(userOp);
                Logger::global().error("Task with scheduling number {0}", task->getAttr("SchedulingNumber"));
                if (task.getExecutorKind() == VPU::ExecutorKind::NCE) {
                    consumers.push_back(userOp);
                } else if (task.getExecutorKind() == VPU::ExecutorKind::DMA_NN) {
                    consumers.push_back(userOp);
                } else if (task.getExecutorKind() == VPU::ExecutorKind::SHAVE_UPA) {
                    consumers.push_back(userOp);
                }
            } else {
                VPUX_THROW("Barrier '{0}' has unsupported Effect in Operation '{1}'", barrierOp->getLoc(),
                           userOp->getLoc());
            }
        }
        _barrierProducersMap.insert(std::make_pair(barrierOp, producers));
        _barrierConsumersMap.insert(std::make_pair(barrierOp, consumers));
    }
}

void RuntimeSimulator::computeOpIndegree() {
    in_degree_map_.clear();

    _func.walk([&](VPURT::DeclareVirtualBarrierOp barrierOp) {
        Logger::global().error("The indegree for the barrier ID {0} is {1}", barrierOp->getAttr("id"),
                               _barrierProducersMap[barrierOp].size());
        in_degree_map_.insert(std::make_pair(barrierOp.getOperation(), _barrierProducersMap[barrierOp].size()));
    });
    std::cout << "The size of indegree table is " << in_degree_map_.size() << std::endl;
}

void RuntimeSimulator::computeOpOutdegree() {
    out_degree_map_.clear();

    _func.walk([&](VPURT::DeclareVirtualBarrierOp barrierOp) {
        out_degree_map_.insert(std::make_pair(barrierOp.getOperation(), _barrierConsumersMap[barrierOp].size()));
    });
    std::cout << "The size of outdegree table is " << out_degree_map_.size() << std::endl;
}

std::pair<int64_t, int64_t> RuntimeSimulator::getID(mlir::Operation* val) const {
    const auto it = _virtualToPhysicalBarrierMap.find(val);
    VPUX_THROW_UNLESS(it != _virtualToPhysicalBarrierMap.end(), "Value '{0}' was not covered by BarrierAllocation");
    return it->second;
}

bool RuntimeSimulator::processTasks(std::vector<TaskInfo>& task_list) {
    for (auto& task : task_list)
        Logger::global().error("Task scheduling number {0} ", task.taskOp->getAttr("SchedulingNumber"));

    taskInfo_iterator_t tbegin = task_list.begin();
    bool progressed = false;

    while (tbegin != task_list.end()) {
        auto op = *tbegin;
        Logger::global().error("Task scheduling number is {0}", op.taskOp->getAttr("SchedulingNumber"));
        if (!isTaskReady(op.taskOp)) {
            Logger::global().error("Task with scheduling number {0} is NOT ready, its wait/update barriers are not in "
                                   "the active barrier table ",
                                   op.taskOp->getAttr("SchedulingNumber"));
            ++tbegin;
            break;
        }
        Logger::global().error(
                "Task with scheduling number {0} IS ready, its wait/update barriers ARE in the active barrier table ",
                op.taskOp->getAttr("SchedulingNumber"));
        processTask(op.taskOp);
        progressed = true;
        Logger::global().error("Removing task with scheduling number {0} from its repective list",
                               op.taskOp->getAttr("SchedulingNumber"));
        tbegin = task_list.erase(tbegin);
    }
    return progressed;
}

bool RuntimeSimulator::fillBarrierTasks(std::list<VPURT::DeclareVirtualBarrierOp>& barrier_task_list) {
    active_barrier_table_iterator_t aitr;
    bool progressed = false;

    barrier_list_iterator_t bcurr = barrier_task_list.begin();
    barrier_list_iterator_t bend = barrier_task_list.end();
    barrier_list_iterator_t berase;

    while ((bcurr != bend) && !_real_barrier_list.empty()) {
        // atleast one barrier tasks and atleast one real barrier //
        auto bop = *bcurr;
        acquireRealBarrier(bop);
        progressed = true;
        berase = bcurr;
        ++bcurr;
        barrier_task_list.erase(berase);
    }
    return progressed;
}

bool RuntimeSimulator::isTaskReadyByBarrierMap(VPURT::TaskOp taskOp) {
    Logger::global().error("Is task with scheduling number {0} ready?", taskOp->getAttr("SchedulingNumber"));
    // // wait barriers //
    // for (const auto waitBarrier : taskOp.waitBarriers()) {
    //     if (auto barrierOp = waitBarrier.getDefiningOp()) {
    //         active_barrier_table_iterator_t aitr = _active_barrier_table.find(barrierOp);

    //         if ((aitr == _active_barrier_table.end()) ||
    //             ((aitr->second).in_degree_ > 0))  // double check this condition
    //         {
    //             return false;
    //         }
    //     }
    // }
    // // update barriers //
    // for (const auto updateBarrier : taskOp.updateBarriers()) {
    //     if (auto barrierOp = updateBarrier.getDefiningOp()) {
    //         active_barrier_table_iterator_t aitr = _active_barrier_table.find(barrierOp);
    //         Logger::global().error("Looking for Barrier ID {1} in the _active_barrier_table ",
    //                                barrierOp->getAttr("id"));
    //         if (aitr == _active_barrier_table.end()) {
    //             return false;
    //         }
    //     }
    // }
    // wait barriers //
    for (const auto waitBarrier : _configureTaskOpUpdateWaitMap[taskOp].first) {
        active_barrier_table_iterator_t aitr = _active_barrier_table.find(waitBarrier);

        if ((aitr == _active_barrier_table.end()) || ((aitr->second).in_degree_ > 0))  // double check this condition
        {
            return false;
        }
    }
    // update barriers //
    for (const auto updateBarrier : _configureTaskOpUpdateWaitMap[taskOp].second) {
        active_barrier_table_iterator_t aitr = _active_barrier_table.find(updateBarrier);
        Logger::global().error("Looking for Barrier ID {1} in the _active_barrier_table ",
                               updateBarrier->getAttr("id"));
        if (aitr == _active_barrier_table.end()) {
            return false;
        }
    }
    return true;
}

void RuntimeSimulator::processTaskByBarrierMap(VPURT::TaskOp task) {
    // assert(isTaskReady(task));
    Logger::global().error("Now processing task with scheduling number {0}", task->getAttr("SchedulingNumber"));

    active_barrier_table_iterator_t aitr;

    // wait barrier
    // for (const auto waitBarrier : task.waitBarriers()) {
    //     if (auto barrierOp = waitBarrier.getDefiningOp()) {
    //         aitr = _active_barrier_table.find(barrierOp);
    //         assert(aitr != _active_barrier_table.end());

    //         Logger::global().error(
    //                 "Found the wait barrier for task with scheduling number {0} in the active barrier table",
    //                 task->getAttr("SchedulingNumber"));

    //         active_barrier_info_t& barrier_info = aitr->second;
    //         assert(barrier_info.in_degree_ == 0UL);
    //         assert(barrier_info.out_degree_ > 0UL);

    //         Logger::global().error("Decrmenting the out degree of the physical barrier {0}",
    //                                barrier_info.real_barrier_);
    //         barrier_info.out_degree_--;

    //         if (barrier_info.out_degree_ == 0UL) {
    //             // return the barrier //
    //             Logger::global().error("The out degree of the physical barrier {0} is 0",
    //             barrier_info.real_barrier_); returnRealBarrier(barrierOp);
    //         }
    //     }
    // }

    // // update barriers //
    // for (const auto updateBarrier : task.updateBarriers()) {
    //     if (auto barrierOp = updateBarrier.getDefiningOp()) {
    //         aitr = _active_barrier_table.find(barrierOp);
    //         assert(aitr != _active_barrier_table.end());

    //         active_barrier_info_t& barrier_info = aitr->second;
    //         assert(barrier_info.in_degree_ > 0UL);
    //         Logger::global().error("Decrmenting the out degree of the physical barrier {0}",
    //                                barrier_info.real_barrier_);
    //         barrier_info.in_degree_--;
    //     }
    // }

    for (const auto waitBarrier : _configureTaskOpUpdateWaitMap[task].first) {
        aitr = _active_barrier_table.find(waitBarrier);
        assert(aitr != _active_barrier_table.end());

        Logger::global().error("Found the wait barrier for task with scheduling number {0} in the active barrier table",
                               task->getAttr("SchedulingNumber"));

        active_barrier_info_t& barrier_info = aitr->second;
        assert(barrier_info.in_degree_ == 0UL);
        assert(barrier_info.out_degree_ > 0UL);

        Logger::global().error("Decrmenting the out degree of the physical barrier {0}", barrier_info.real_barrier_);
        barrier_info.out_degree_--;

        if (barrier_info.out_degree_ == 0UL) {
            // return the barrier //
            Logger::global().error("The out degree of the physical barrier {0} is 0", barrier_info.real_barrier_);
            returnRealBarrier(waitBarrier);
        }
    }
    // update barriers //
    for (const auto updateBarrier : _configureTaskOpUpdateWaitMap[task].second) {
        aitr = _active_barrier_table.find(updateBarrier);
        assert(aitr != _active_barrier_table.end());

        active_barrier_info_t& barrier_info = aitr->second;
        assert(barrier_info.in_degree_ > 0UL);
        Logger::global().error("Decrmenting the out degree of the physical barrier {0}", barrier_info.real_barrier_);
        barrier_info.in_degree_--;
    }
}

bool RuntimeSimulator::processTasksByBarrierMap(std::vector<TaskInfo>& task_list) {
    for (auto& task : task_list)
        Logger::global().error("Task scheduling number {0} ", task.taskOp->getAttr("SchedulingNumber"));

    taskInfo_iterator_t tbegin = task_list.begin();
    bool progressed = false;

    while (tbegin != task_list.end()) {
        auto op = *tbegin;
        Logger::global().error("Task scheduling number is {0}", op.taskOp->getAttr("SchedulingNumber"));
        if (!isTaskReadyByBarrierMap(op.taskOp)) {
            Logger::global().error("Task with scheduling number {0} is NOT ready, its wait/update barriers are not in "
                                   "the active barrier table ",
                                   op.taskOp->getAttr("SchedulingNumber"));
            ++tbegin;
            break;
        }
        Logger::global().error(
                "Task with scheduling number {0} IS ready, its wait/update barriers ARE in the active barrier table ",
                op.taskOp->getAttr("SchedulingNumber"));
        processTaskByBarrierMap(op.taskOp);
        progressed = true;
        Logger::global().error("Removing task with scheduling number {0} from its repective list",
                               op.taskOp->getAttr("SchedulingNumber"));
        tbegin = task_list.erase(tbegin);
    }
    return progressed;
}

bool RuntimeSimulator::simulate(
        std::list<VPURT::DeclareVirtualBarrierOp>& barrierOps,
        std::unordered_map<mlir::Operation*, SmallVector<mlir::Operation*>>& barrierProducersMap,
        std::unordered_map<mlir::Operation*, SmallVector<mlir::Operation*>>& barrierConsumersMap,
        std::map<mlir::Operation*, std::pair<std::set<mlir::Operation*>, std::set<mlir::Operation*>>>&
                configureTaskOpUpdateWaitMap) {
    _log.trace("Running barrier simulator");
    // VPUX_UNUSED(barrierOps);
    // VPUX_UNUSED(barrierProducersMap);
    // VPUX_UNUSED(barrierConsumersMap);
    init();

    // build task list
    // _barrierOps = barrierOps;
    _configureTaskOpUpdateWaitMap = configureTaskOpUpdateWaitMap;

    auto getTaskInfo = [&](VPURT::TaskOp taskOp) {
        TaskInfo taskInfo(taskOp);
        for (auto waitBarrier : taskOp.waitBarriers()) {
            if (auto barrierOp = mlir::dyn_cast<VPURT::ConfigureBarrierOp>(waitBarrier.getDefiningOp())) {
                const auto virtualId = getVirtualId(barrierOp);
                taskInfo.waitBarriers.push_back(virtualId);
            }
        }
        for (auto updateBarrier : taskOp.updateBarriers()) {
            if (auto barrierOp = mlir::dyn_cast<VPURT::ConfigureBarrierOp>(updateBarrier.getDefiningOp())) {
                const auto virtualId = getVirtualId(barrierOp);
                taskInfo.updateBarriers.push_back(virtualId);
            }
        }
        return taskInfo;
    };

    // The task lists have to be populated in the same order as during the serialization phase
    // to ensure that the correct simulation occurs
    _func.walk([&](VPURT::TaskOp taskOp) {
        auto& block = taskOp.body().getBlocks().front();
        auto wrappedTaskOp = block.begin();
        switch (taskOp.getExecutorKind()) {
        // case VPU::ExecutorKind::UPADMA:
        case VPU::ExecutorKind::DMA_NN: {
            int64_t port = 0;
            if (auto dmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(wrappedTaskOp)) {
                port = dmaOp.port();
            } else if (auto compressedDmaOp = mlir::dyn_cast<VPUIP::CompressedDMAOp>(wrappedTaskOp)) {
                port = compressedDmaOp.port();
            } else {
                VPUX_THROW("Could not cast to DMA task");
            }
            VPUX_THROW_UNLESS(port < MAX_DMA_ENGINES,
                              "NNDMAOp port value ({0}) larger than maximum number of engines ({1})", port,
                              MAX_DMA_ENGINES);
            Logger::global().error("Adding DMA scheduling number {0} ", taskOp->getAttr("SchedulingNumber"));
            _dmaTasks[port].push_back(getTaskInfo(taskOp));
            break;
        }
        case VPU::ExecutorKind::NCE: {
            auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(wrappedTaskOp);
            VPUX_THROW_UNLESS(nceOp != nullptr, "Could not cast to NCE task");
            _nceTasks.push_back(getTaskInfo(taskOp));
            break;
        }
        // TODO: should we introduce _swTask?
        // case VPU::ExecutorKind::ACTShave:
        case VPU::ExecutorKind::SHAVE_UPA: {
            Logger::global().error("Adding UPA scheduling number {0} ", taskOp->getAttr("SchedulingNumber"));
            _upaTasks.push_back(getTaskInfo(taskOp));
            break;
        }
        default:
            VPUX_THROW("Unsupported task type '{0}'", taskOp.getExecutorKind());
        }
    });

    // getAllBarriersProducersAndConsumers
    _barrierProducersMap = barrierProducersMap;
    _barrierConsumersMap = barrierConsumersMap;

    computeOpIndegree();
    computeOpOutdegree();

    size_t barrier = 0;
    size_t nce = 0;
    size_t upa = 0;
    std::array<size_t, MAX_DMA_ENGINES> dma = {0};

    std::cout << "barrier number is" << barrierOps.size() << std::endl;

    while (barrier < barrierOps.size() || dma[0] < _dmaTasks[0].size() || dma[1] < _dmaTasks[1].size() ||
           nce < _nceTasks.size() || upa < _upaTasks.size()) {
        _log.nest(2).trace("BAR: {0} / {1}; DMA: {2} / {3}, {4} / {5}; NCE: {6} / {7}; UPA: {8} / {9}", barrier,
                           barrierOps.size(), dma[0], _dmaTasks[0].size(), dma[1], _dmaTasks[1].size(), nce,
                           _nceTasks.size(), upa, _upaTasks.size());

        std::cout << "Starting runtime simulation" << std::endl;
        bool progressed = false;
        while (!_dmaTasks[0].empty() || !_nceTasks.empty() || !barrierOps.empty() || !_upaTasks.empty()) {
            progressed = false;
            std::cout << "fillBarrierTasks" << std::endl;
            progressed |= fillBarrierTasks(barrierOps);
            std::cout << "processTasksByBarrierMap" << std::endl;
            progressed |= processTasksByBarrierMap(_dmaTasks[0]);
            progressed |= processTasksByBarrierMap(_nceTasks);
            progressed |= processTasksByBarrierMap(_upaTasks);

            if (!progressed) {
                return false;
            }
        }

        std::cout << "simulate" << std::endl;
        for (auto& barrier : _virtualToPhysicalBarrierMap) {
            // Logger::global().error("Virtual Barrier ID {0} has physical ID {1}", barrier.first->getAttr("id"),
            //                        barrier.second.first);
            std::cout << barrier.first->getAttr("id").cast<mlir::IntegerAttr>().getInt() << " " << barrier.second.first
                      << std::endl;
        }
    }

    return true;
}