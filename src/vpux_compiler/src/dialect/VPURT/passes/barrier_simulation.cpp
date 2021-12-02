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

#include "vpux/compiler/dialect/VPURT/passes.hpp"

#include "vpux/compiler/dialect/VPURT/ops.hpp"

using namespace vpux;

namespace {

static constexpr int64_t MAX_DMA_ENGINES = 2;
static constexpr StringLiteral virtualIdAttrName = "virtualId";

//
// BarrierSimulator
//

class BarrierSimulator final {
public:
    enum class UpdateStatus { Success, Skip, Fail };

    struct TaskInfo {
        VPURT::TaskOp taskOp;
        SmallVector<int64_t> waitBarriers;
        SmallVector<int64_t> updateBarriers;

        TaskInfo() {
        }
        TaskInfo(VPURT::TaskOp taskOp): taskOp(taskOp) {
        }
    };

    struct VirtualBarrierInfo {
        int64_t realId;
        int64_t producerCount;
        int64_t consumerCount;
        int64_t initProducerCount;
        int64_t initConsumerCount;

        VirtualBarrierInfo(): realId(), producerCount(), consumerCount(), initProducerCount(), initConsumerCount() {
        }
    };

    explicit BarrierSimulator(mlir::MLIRContext* ctx, mlir::FuncOp func, Logger log, int64_t numDmaEngines)
            : _ctx(ctx), _func(func), _log(log), _numDmaEngines(numDmaEngines), _numRealBarriers(), _active_barrier_table(), _real_barrier_list() {
    }

    void init();
    bool assignPhysicalIDs();
    void buildTaskLists();
    void assignVirtualIds(mlir::FuncOp func);
    void acquireRealBarrier(VPURT::DeclareVirtualBarrierOp btask);
    bool is_task_ready(VPURT::TaskOp taskOp);
    void getAllBarriersProducersAndConsumers();
    void compute_op_indegree();
    int64_t getVirtualId(VPURT::ConfigureBarrierOp op);
    static bool orderbyID(TaskInfo& a, TaskInfo& b);
    bool processDMAtasks(std::vector<TaskInfo> dma_task_list);
    bool fillBarrierTasks(std::list<VPURT::DeclareVirtualBarrierOp>& barrier_task_list);
    mlir::LogicalResult checkProducerCount();
    mlir::LogicalResult run();

private:
    UpdateStatus updateTaskBarriers(const TaskInfo& taskInfo, const int64_t count);
    void logDebugInfo(const size_t barrier, const std::array<size_t, MAX_DMA_ENGINES>& dma, const size_t nce,
                      const size_t upa);

private:
 
    std::vector<TaskInfo> _nceTasks;
    std::vector<TaskInfo> _upaTasks;
    
    std::array<std::vector<TaskInfo>, MAX_DMA_ENGINES> _dmaTasks;
    std::list<VPURT::DeclareVirtualBarrierOp> _barrierOps;

    std::map<int64_t, VirtualBarrierInfo> _virtualBarriers;
    SmallVector<int64_t> _barrierConfig;
    

    mlir::MLIRContext* _ctx;
    mlir::FuncOp _func;
    Logger _log;
    int64_t _numDmaEngines;
    int64_t _numRealBarriers;

    struct active_barrier_info_t {
      size_t real_barrier_;
      //size_t in_degree_;
      //size_t out_degree_;
      active_barrier_info_t(size_t real/*, size_t in, size_t out*/)
        : real_barrier_(real)/*, in_degree_(in), out_degree_(out)*/ {}
    };

    typedef std::unordered_map<mlir::Operation*, active_barrier_info_t> active_barrier_table_t;
    typedef active_barrier_table_t::iterator active_barrier_table_iterator_t;
    typedef std::list<VPURT::DeclareVirtualBarrierOp>::iterator barrier_list_iterator_t;

    typedef std::vector<TaskInfo>::iterator taskInfo_iterator_t;

    active_barrier_table_t _active_barrier_table;
    std::list<size_t> _real_barrier_list;


    typedef std::unordered_map<mlir::Operation*, size_t> in_degree_map_t;
    typedef std::unordered_map<mlir::Operation*, size_t> out_degree_map_t;
    in_degree_map_t in_degree_map_;
    out_degree_map_t out_degree_map_;

    std::unordered_map<mlir::Operation*, SmallVector<mlir::Operation*>> barrierProducersMap{};
    std::unordered_map<mlir::Operation*, SmallVector<mlir::Operation*>> barrierConsumersMap{};
};

void BarrierSimulator::getAllBarriersProducersAndConsumers() {
    // Get all producers and consumers of barriers (NCE,UPA, DMA) only
    auto _barrierOps = to_small_vector(_func.getOps<VPURT::DeclareVirtualBarrierOp>());

    for (auto& barrierOp : _barrierOps) {
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

            VPUX_THROW_WHEN(effect.getResource() != VPUIP::BarrierResource::get(),
                            "Barrier '{0}' has non Barrier Resource for Operation '{1}'", barrierOp->getLoc(),
                            userOp->getLoc());

            if (effect.getEffect() == mlir::MemoryEffects::Write::get()) {
                auto task = mlir::dyn_cast<VPURT::TaskOp>(userOp);
                if (task.getTaskType() == VPUIP::TaskType::NCE2) {
                    producers.push_back(userOp);
                } else if (task.getTaskType() == VPUIP::TaskType::NNDMA) {
                    producers.push_back(userOp);
                } else if (task.getTaskType() == VPUIP::TaskType::UPA) {
                    producers.push_back(userOp);
                }
            } else if (effect.getEffect() == mlir::MemoryEffects::Read::get()) {
                auto task = mlir::dyn_cast<VPURT::TaskOp>(userOp);
                if (task.getTaskType() == VPUIP::TaskType::NCE2) {
                    consumers.push_back(userOp);
                } else if (task.getTaskType() == VPUIP::TaskType::NNDMA) {
                    consumers.push_back(userOp);
                } else if (task.getTaskType() == VPUIP::TaskType::UPA) {
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


void BarrierSimulator::compute_op_indegree() {
    
    in_degree_map_.clear();

    _func.walk([&](VPURT::TaskOp taskOp) {
        auto& block = taskOp.op().getBlocks().front();
        auto wrappedTaskOp = block.begin();
        size_t waitBarrierIncomingEdges = 0;

        for (const auto waitBarrier : taskOp.waitBarriers()) {
            if (auto barrierOp = waitBarrier.getDefiningOp()) {
                waitBarrierIncomingEdges += barrierProducersMap[barrierOp].size();
            }
        }
        Logger::global().error("The indegree for the operation with scheduling number {0}  is {1}", taskOp->getAttr("SchedulingNumber"),
                               waitBarrierIncomingEdges);

        in_degree_map_.insert(std::make_pair(taskOp.getOperation(), waitBarrierIncomingEdges));
    });
    std::cout << "The size of indegree table is " << in_degree_map_.size() << std::endl;
}

void BarrierSimulator::init() {
     _real_barrier_list.clear();

      for (size_t i=0; i<8 ; i++) {
        _real_barrier_list.push_back(i);
      }
}

void BarrierSimulator::assignVirtualIds(mlir::FuncOp func) {
    int64_t virtualId = 0;
    func.walk([&](VPURT::ConfigureBarrierOp op) {
        op->setAttr(virtualIdAttrName, getIntAttr(_ctx, virtualId++));
    });
}

int64_t BarrierSimulator::getVirtualId(VPURT::ConfigureBarrierOp op) {
    return checked_cast<int64_t>(op->getAttr(virtualIdAttrName).cast<mlir::IntegerAttr>().getInt());
}

bool BarrierSimulator::orderbyID(TaskInfo& a, TaskInfo& b)
{
    int64_t aID = checked_cast<int64_t>(a.taskOp->getAttr("SchedulingNumber").cast<mlir::IntegerAttr>().getInt());
    int64_t bID = checked_cast<int64_t>(b.taskOp->getAttr("SchedulingNumber").cast<mlir::IntegerAttr>().getInt());
    return aID < bID;
}

void BarrierSimulator::buildTaskLists() {
    _log.trace("Building task lists");

    auto getTaskInfo = [&](VPURT::TaskOp taskOp, const int64_t count = 1) {
        TaskInfo taskInfo(taskOp);
        for (auto waitBarrier : taskOp.waitBarriers()) {
            if (auto barrierOp = mlir::dyn_cast<VPURT::ConfigureBarrierOp>(waitBarrier.getDefiningOp())) {
                const auto virtualId = getVirtualId(barrierOp);
                taskInfo.waitBarriers.push_back(virtualId);
                _virtualBarriers[virtualId].consumerCount += count;
                _virtualBarriers[virtualId].initConsumerCount += count;
            }
        }
        for (auto updateBarrier : taskOp.updateBarriers()) {
            if (auto barrierOp = mlir::dyn_cast<VPURT::ConfigureBarrierOp>(updateBarrier.getDefiningOp())) {
                const auto virtualId = getVirtualId(barrierOp);
                taskInfo.updateBarriers.push_back(virtualId);
                _virtualBarriers[virtualId].producerCount += count;
                _virtualBarriers[virtualId].initProducerCount += count;
            }
        }
        return taskInfo;
    };

    // The task lists have to be populated in the same order as during the serialization phase
    // to ensure that the correct simulation occurs
    _func.walk([&](VPURT::DeclareVirtualBarrierOp barrierOp) {
        Logger::global().error("Adding Barrier ID {0}", barrierOp->getAttr("id").cast<mlir::IntegerAttr>().getInt());
        _barrierOps.push_back(barrierOp);
        //const auto virtualId = getVirtualId(barrierOp);
       // _virtualBarriers[virtualId].realId = barrierOp.id();
        _numRealBarriers = std::max(_numRealBarriers, barrierOp.id() + 1);
    });

    _func.walk([&](VPURT::TaskOp taskOp) {
        Logger::global().error("Scheduling number {0} ", taskOp->getAttr("SchedulingNumber"));
    });

    _func.walk([&](VPURT::TaskOp taskOp) {
        auto& block = taskOp.op().getBlocks().front();
        auto wrappedTaskOp = block.begin();
        switch (taskOp.getTaskType()) {
        case VPUIP::TaskType::UPADMA:
        case VPUIP::TaskType::NNDMA: {
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
        case VPUIP::TaskType::NCE2: {
            auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(wrappedTaskOp);
            VPUX_THROW_UNLESS(nceOp != nullptr, "Could not cast to NCE task");
            _nceTasks.push_back(getTaskInfo(taskOp, nceOp.getNumVariants()));
            break;
        }
        // TODO: should we introduce _swTask?
        case VPUIP::TaskType::ACTShave:
        case VPUIP::TaskType::UPA: {
            Logger::global().error("Adding UPA scheduling number {0} ", taskOp->getAttr("SchedulingNumber"));
            _upaTasks.push_back(getTaskInfo(taskOp));
            break;
        }
        default:
            VPUX_THROW("Unsupported task type '{0}'", taskOp.getTaskType());
        }
    });

    // sort barriers
    _barrierOps.sort([](mlir::Operation* a, mlir::Operation* b) -> bool {
        int64_t aID = checked_cast<int64_t>(a->getAttr("id").cast<mlir::IntegerAttr>().getInt());
        int64_t bID = checked_cast<int64_t>(b->getAttr("id").cast<mlir::IntegerAttr>().getInt());
    return aID < bID;
    });

    for(auto& barrier: _barrierOps)
        Logger::global().error("Barrier ID {0} ", barrier->getAttr("id"));

    // sort DMA
    std::sort(_dmaTasks[0].begin(), _dmaTasks[0].end(), orderbyID);
    
    for(auto& dma: _dmaTasks[0])
        Logger::global().error("DMA scheduling number {0} ", dma.taskOp->getAttr("SchedulingNumber"));
    // sort ncetasks
    std::sort(_nceTasks.begin(), _nceTasks.end(), orderbyID);

    for(auto& nce: _nceTasks)
        Logger::global().error("NCE scheduling number {0} ", nce.taskOp->getAttr("SchedulingNumber"));

    // sort upatasks
    std::sort(_upaTasks.begin(), _upaTasks.end(), orderbyID);

    for(auto& upa: _upaTasks)
        Logger::global().error("UPA scheduling number {0} ", upa.taskOp->getAttr("SchedulingNumber"));

    std::cout << "Done" << std::endl;

}

BarrierSimulator::UpdateStatus BarrierSimulator::updateTaskBarriers(const TaskInfo& taskInfo, const int64_t count = 1) {
    for (auto waitBarrier : taskInfo.waitBarriers) {
        const auto realId = _virtualBarriers[waitBarrier].realId;
        if (_barrierConfig[realId] != waitBarrier || _virtualBarriers[waitBarrier].producerCount != 0)
            return UpdateStatus::Skip;
    }
    for (auto updateBarrier : taskInfo.updateBarriers) {
        const auto realId = _virtualBarriers[updateBarrier].realId;
        if (_barrierConfig[realId] != updateBarrier)
            return UpdateStatus::Skip;
    }

    for (auto waitBarrier : taskInfo.waitBarriers) {
        auto& virtualBarrier = _virtualBarriers[waitBarrier];
        if (virtualBarrier.consumerCount < count) {
            _log.error("Barrier {0} has fewer consumers left than currently needed (C: {1}, count: {2})", waitBarrier,
                       virtualBarrier.consumerCount, count);
            return UpdateStatus::Fail;
        }
        virtualBarrier.consumerCount -= count;
        if (virtualBarrier.consumerCount == 0)
            _barrierConfig[virtualBarrier.realId] = -1;
    }
    for (auto updateBarrier : taskInfo.updateBarriers) {
        auto& virtualBarrier = _virtualBarriers[updateBarrier];
        if (virtualBarrier.producerCount < count) {
            _log.error("Barrier {0} has fewer producers left than currently needed (P: {1}, count: {2})", updateBarrier,
                       virtualBarrier.producerCount, count);
            return UpdateStatus::Fail;
        }
        virtualBarrier.producerCount -= count;
    }

    return UpdateStatus::Success;
}

/*
 * The limitation is not related to HW capabilities or FIFO depth, but to the fact that the runtime needs to know when a
 * workload is completed, in order to replace it with another one in NN CMX.
 *
 * Since there's no other efficient feedback mechanism from DPU/SNN to LNN, LNN monitors the barrier production of DPU
 * tasks and recycles the storage when the corresponding barrier gets produced. The limitation comes from how many
 * invariants/variants can be stored in NN CMX at the same time. For single cluster inferences these counts are 64/512,
 * while for 4-cluster inferences 128/512. If too many invariants/variants contribute to the same barrier, the runtime
 * will not receive the confirmation that it may recycle the storage to bring in the next workloads, hence the deadlock.
 *
 * Since the storage area is double buffered, and the workloads in question may start at any index in the buffer, it's
 * only safe for at most <storage_size / 2 + 1> consecutive invariants/variants to produce the same barrier. So finally,
 * the limits are:
 *
 * On single cluster:
 *   32 + 1 invariants
 *   256 + 1 variants
 * On 4 clusters:
 *   64 + 1 invariants
 *   256 + 1 variants
 */
mlir::LogicalResult BarrierSimulator::checkProducerCount() {
    static constexpr int64_t MAX_PRODUCER_COUNT = 256;
    for (auto virtualBarrier : _virtualBarriers) {
        const auto virtualId = virtualBarrier.first;
        const auto producerCount = virtualBarrier.second.producerCount;
        if (producerCount > MAX_PRODUCER_COUNT) {
            _log.error("Barrier {0} has {1} barriers (max {2})", virtualId, producerCount, MAX_PRODUCER_COUNT);
            return mlir::failure();
        }
    }
    return mlir::success();
}

void BarrierSimulator::acquireRealBarrier(VPURT::DeclareVirtualBarrierOp btask) {
      assert(!_real_barrier_list.empty());
      size_t real = _real_barrier_list.front();
      _real_barrier_list.pop_front();

      assert(_active_barrier_table.size() < 8);

    //   in_degree_iterator_t in_itr = in_degree_map_.find(btask);
    //   out_degree_iterator_t out_itr = out_degree_map_.find(btask);

    //   assert((in_itr != in_degree_map_.end()) && 
    //         (out_itr != out_degree_map_.end()));

    assert(_active_barrier_table.find(btask) == _active_barrier_table.end());

     _active_barrier_table.insert(std::make_pair(btask, active_barrier_info_t(real/*, in_itr->second, out_itr->second)*/)));
}

bool BarrierSimulator::fillBarrierTasks(std::list<VPURT::DeclareVirtualBarrierOp>& barrier_task_list) {
      
    active_barrier_table_iterator_t aitr;
    bool progressed = false;

    barrier_list_iterator_t bcurr = barrier_task_list.begin(); 
    barrier_list_iterator_t bend = barrier_task_list.end(); 
    barrier_list_iterator_t berase;

    while ( (bcurr != bend) && !_real_barrier_list.empty() ) {
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

bool BarrierSimulator::is_task_ready(VPURT::TaskOp taskOp) {

    // wait barriers //

    for (const auto waitBarrier : taskOp.waitBarriers()) {

        if (auto barrierOp = waitBarrier.getDefiningOp()) {

        active_barrier_table_iterator_t aitr = _active_barrier_table.find(barrierOp);

            // if ((aitr == _active_barrier_table.end()) || ((aitr->second).in_degree_ > 0) ) 
            // { 
            //     return false;
            // }
        }
    }
    // update barriers //
    for (const auto updateBarrier : taskOp.updateBarriers()) {
        
        if (auto barrierOp = updateBarrier.getDefiningOp()) {
  
            if (_active_barrier_table.find(barrierOp) == _active_barrier_table.end())
            { 
                return false; 
            }
        }
    }
    return true;
}

bool BarrierSimulator::processDMAtasks(std::vector<TaskInfo> dma_task_list) {

    taskInfo_iterator_t tbegin = dma_task_list.begin();
    taskInfo_iterator_t tend = dma_task_list.end();
    taskInfo_iterator_t terase;
    bool progressed = false;

      while (tbegin != tend) {
        auto op = *tbegin;
        // if (!is_task_ready(op) ) { break; }
        // process_task(op);
        // filled_atleast_once = true;
        // terase = tbegin;
        // ++tbegin;
        // task_list.erase(terase);
      }
      return progressed;
}

bool BarrierSimulator::assignPhysicalIDs()
{

     _log.trace("Running barrier simulator");

    init();
    buildTaskLists();
    getAllBarriersProducersAndConsumers();
    compute_op_indegree();
    size_t barrier = 0;
    size_t nce = 0;
    size_t upa = 0;
    std::array<size_t, MAX_DMA_ENGINES> dma = {0};
    while (barrier < _barrierOps.size() || dma[0] < _dmaTasks[0].size() || dma[1] < _dmaTasks[1].size() || nce < _nceTasks.size() || upa < _upaTasks.size()) {
        
        
        _log.nest(2).trace("BAR: {0} / {1}; DMA: {2} / {3}, {4} / {5}; NCE: {6} / {7}; UPA: {8} / {9}", barrier,
                           _barrierOps.size(), dma[0], _dmaTasks[0].size(), dma[1], _dmaTasks[1].size(), nce,
                           _nceTasks.size(), upa, _upaTasks.size());

        bool progressed = false;

        while (!_dmaTasks[0].empty() || !_nceTasks.empty() || !_barrierOps.empty()) {
            progressed = false;
            progressed |= fillBarrierTasks(_barrierOps);
            progressed |= processDMAtasks(_dmaTasks[0]);
            //progressed |= process_tasks(compute_list);
        if (!progressed) { return false; }
      }
      return true;
    }


}

mlir::LogicalResult BarrierSimulator::run() {
    _log.trace("Running barrier simulator");

    _barrierConfig.resize(_numRealBarriers);
    std::fill(_barrierConfig.begin(), _barrierConfig.end(), -1);

    size_t barrier = 0;
    size_t nce = 0;
    size_t upa = 0;
    std::array<size_t, MAX_DMA_ENGINES> dma = {0};
    while (barrier < _barrierOps.size() || dma[0] < _dmaTasks[0].size() || dma[1] < _dmaTasks[1].size() ||
           nce < _nceTasks.size() || upa < _upaTasks.size()) {
        _log.nest(2).trace("BAR: {0} / {1}; DMA: {2} / {3}, {4} / {5}; NCE: {6} / {7}; UPA: {8} / {9}", barrier,
                           _barrierOps.size(), dma[0], _dmaTasks[0].size(), dma[1], _dmaTasks[1].size(), nce,
                           _nceTasks.size(), upa, _upaTasks.size());

        bool progressed = false;

        for (; barrier < _barrierOps.size(); ++barrier) {
            // const auto virtualId = getVirtualId(_barrierOps[barrier]);
            // const auto realId = _virtualBarriers[virtualId].realId;
            // if (_barrierConfig[realId] > -1)
            //     break;
            // _barrierConfig[realId] = virtualId;
            // progressed = true;
        }

        for (int64_t e = 0; e < _numDmaEngines; ++e) {
            for (; dma[e] < _dmaTasks[e].size(); ++dma[e]) {
                const auto status = updateTaskBarriers(_dmaTasks[e][dma[e]]);
                if (status == UpdateStatus::Skip)
                    break;
                else if (status == UpdateStatus::Fail)
                    return mlir::failure();
                progressed = true;
            }
        }

        for (; nce < _nceTasks.size(); ++nce) {
            auto& block = _nceTasks[nce].taskOp.op().getBlocks().front();
            auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(block.begin());
            VPUX_THROW_UNLESS(nceOp != nullptr, "Could not cast to NCE task");
            const auto status = updateTaskBarriers(_nceTasks[nce], nceOp.getNumVariants());
            if (status == UpdateStatus::Skip)
                break;
            else if (status == UpdateStatus::Fail)
                return mlir::failure();
            progressed = true;
        }

        for (; upa < _upaTasks.size(); ++upa) {
            const auto status = updateTaskBarriers(_upaTasks[upa]);
            if (status == UpdateStatus::Skip)
                break;
            else if (status == UpdateStatus::Fail)
                return mlir::failure();
            progressed = true;
        }

        if (!progressed) {
            logDebugInfo(barrier, dma, nce, upa);
            return mlir::failure();
        }
    }

    return mlir::success();
}

void BarrierSimulator::logDebugInfo(const size_t barrier, const std::array<size_t, MAX_DMA_ENGINES>& dma,
                                    const size_t nce, const size_t upa) {
    // _log.error("Barrier simulation blocked at BAR: {0} / {1} (virtual_id {2}); DMA: {3} / {4}, {5} / {6};"
    //            "NCE: {7} / {8}; UPA: {9} / {10}",
    //            barrier, _barrierOps.size(), getVirtualId(_barrierOps[barrier]), dma[0], _dmaTasks[0].size(), dma[1],
    //            _dmaTasks[1].size(), nce, _nceTasks.size(), upa, _upaTasks.size());

    _log.error("Real barriers configuration:");
    for (size_t i = 0; i < _barrierConfig.size(); ++i) {
        auto virtualBarrierId = _barrierConfig[i];
        auto virtualBarrier = _virtualBarriers[virtualBarrierId];
        _log.nest(2).error("{0,2}: virtual {1,3},   P: {2,2}, C: {3,2}, initial P: {4,2}, C: {5,2}", i,
                           virtualBarrierId, virtualBarrier.producerCount, virtualBarrier.consumerCount,
                           virtualBarrier.initProducerCount, virtualBarrier.initConsumerCount);
    }

    _log.error("Unfinished barriers status:");
    for (size_t i = 0; i < barrier; ++i) {
        auto virtualBarrier = _virtualBarriers[i];
        if (virtualBarrier.producerCount || virtualBarrier.consumerCount)
            _log.nest(2).error(
                    "Barrier {0,3} (real {1,2}) with remaining counts P: {2,2}, C: {3,2}, initial P: {4,2}, C: {5,2}",
                    i, virtualBarrier.realId, virtualBarrier.producerCount, virtualBarrier.consumerCount,
                    virtualBarrier.initProducerCount, virtualBarrier.initConsumerCount);
    }
}

//
// BarrierSimulationPass
//

class BarrierSimulationPass final : public VPURT::BarrierSimulationBase<BarrierSimulationPass> {
public:
    explicit BarrierSimulationPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void BarrierSimulationPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto resOp = IERT::RunTimeResourcesOp::getFromModule(module);

    const auto dmaAttr = VPU::ExecutorKindAttr::get(&ctx, VPU::ExecutorKind::DMA_NN);
    auto dmaResOp = resOp.getExecutor(dmaAttr);
    VPUX_THROW_UNLESS(dmaResOp != nullptr, "Failed to get DMA_NN information");

    const auto numDmaEngines = dmaResOp.count();
    VPUX_THROW_UNLESS(numDmaEngines <= MAX_DMA_ENGINES, "Found {0} DMA engines (max {1})", numDmaEngines,
                      MAX_DMA_ENGINES);

    BarrierSimulator simulator(&ctx, func, _log, numDmaEngines);
    //simulator.assignVirtualIds(func);
    //simulator.buildTaskLists(func);
    simulator.assignPhysicalIDs();
    if (mlir::failed(simulator.checkProducerCount())) {
        signalPassFailure();
        return;
    }
    if (mlir::failed(simulator.run())) {
        signalPassFailure();
    }
}

}  // namespace

//
// createBarrierSimulationPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createBarrierSimulationPass(Logger log) {
    return std::make_unique<BarrierSimulationPass>(log);
}
