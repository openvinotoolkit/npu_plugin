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

bool RuntimeSimulator::orderbyID(TaskInfo& a, TaskInfo& b) {
    int64_t aID = checked_cast<int64_t>(a.taskOp->getAttr("SchedulingNumber").cast<mlir::IntegerAttr>().getInt());
    int64_t bID = checked_cast<int64_t>(b.taskOp->getAttr("SchedulingNumber").cast<mlir::IntegerAttr>().getInt());
    return aID < bID;
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

    // sort barriers
    _barrierOps.sort([](mlir::Operation* a, mlir::Operation* b) -> bool {
        int64_t aID = checked_cast<int64_t>(a->getAttr("id").cast<mlir::IntegerAttr>().getInt());
        int64_t bID = checked_cast<int64_t>(b->getAttr("id").cast<mlir::IntegerAttr>().getInt());
        return aID < bID;
    });

    for (auto& barrier : _barrierOps)
        Logger::global().error("Barrier ID {0} ", barrier->getAttr("id"));

    // sort DMA
    std::sort(_dmaTasks[0].begin(), _dmaTasks[0].end(), orderbyID);

    for (auto& dma : _dmaTasks[0])
        Logger::global().error("DMA scheduling number {0} ", dma.taskOp->getAttr("SchedulingNumber"));
    
    // sort ncetasks
    std::sort(_nceTasks.begin(), _nceTasks.end(), orderbyID);

    for (auto& nce : _nceTasks)
        Logger::global().error("NCE scheduling number {0} ", nce.taskOp->getAttr("SchedulingNumber"));

    // sort upatasks
    std::sort(_upaTasks.begin(), _upaTasks.end(), orderbyID);

    for (auto& upa : _upaTasks)
        Logger::global().error("UPA scheduling number {0} ", upa.taskOp->getAttr("SchedulingNumber"));

    std::cout << "Done" << std::endl;
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

        for (auto& barrier : _virtualToPhysicalBarrierMap) {
            Logger::global().error("Virtual Barrier ID {0} has physical ID {1}", barrier.first->getAttr("id"),
                                   barrier.second.first);
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

    int64_t virtualID = checked_cast<int64_t>(btask->getAttr("id").cast<mlir::IntegerAttr>().getInt());
    _virtualToPhysicalBarrierMap.insert(std::make_pair(
            btask.getOperation(),
            std::make_pair(real, checked_cast<int64_t>(btask->getAttr("id").cast<mlir::IntegerAttr>().getInt()))));
    _active_barrier_table.insert(std::make_pair(
            btask.getOperation(), active_barrier_info_t(virtualID, real, in_itr->second, out_itr->second)));
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
                Logger::global().error("Is task with scheduling number {0} is NOT ready, its wait barrier {1} is not "
                                       "in the active barrier table or its indegree is > 0",
                                       barrierOp->getAttr("id"), taskOp->getAttr("SchedulingNumber"));
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
                Logger::global().error("Is task with scheduling number {0} is NOT ready, its update barrier {1} is not "
                                       "in the active barrier table",
                                       barrierOp->getAttr("id"), taskOp->getAttr("SchedulingNumber"));
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

            Logger::global().error(
                    "Decrmenting the out degree of the wait virtual barrier {0} which is physical barrier {1}",
                    barrier_info.virtual_id_, barrier_info.real_barrier_);
            barrier_info.out_degree_--;

            if (barrier_info.out_degree_ == 0UL) {
                // return the barrier //
                Logger::global().error(
                        "The out degree of the wait virtual barrier {0} which is physical barrier {1} is 0",
                        barrier_info.virtual_id_, barrier_info.real_barrier_);
                Logger::global().error("Returning the virtual barrier {0} which is physical barrier {1}",
                                       barrier_info.virtual_id_, barrier_info.real_barrier_);
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
            Logger::global().error(
                    "Decrmenting the in-degree of the update virtual barrier {0} which is physical barrier {1}",
                    barrier_info.virtual_id_, barrier_info.real_barrier_);
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
        barrierProducersMap.insert(std::make_pair(barrierOp, producers));
        barrierConsumersMap.insert(std::make_pair(barrierOp, consumers));
    }
}

void RuntimeSimulator::computeOpIndegree() {
    in_degree_map_.clear();

    _func.walk([&](VPURT::DeclareVirtualBarrierOp barrierOp) {
        Logger::global().error("The indegree for the barrier ID {0} is {1}", barrierOp->getAttr("id"),
                               barrierProducersMap[barrierOp].size());
        in_degree_map_.insert(std::make_pair(barrierOp.getOperation(), barrierProducersMap[barrierOp].size()));
    });
    std::cout << "The size of indegree table is " << in_degree_map_.size() << std::endl;
}

void RuntimeSimulator::computeOpOutdegree() {
    out_degree_map_.clear();

    _func.walk([&](VPURT::DeclareVirtualBarrierOp barrierOp) {
        out_degree_map_.insert(std::make_pair(barrierOp.getOperation(), barrierConsumersMap[barrierOp].size()));
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