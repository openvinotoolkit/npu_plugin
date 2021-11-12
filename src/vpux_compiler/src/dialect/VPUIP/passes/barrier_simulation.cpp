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
        VPUIP::TaskOpInterface taskOp;
        SmallVector<int64_t> waitBarriers;
        SmallVector<int64_t> updateBarriers;

        TaskInfo() {
        }
        TaskInfo(VPUIP::TaskOpInterface taskOp): taskOp(taskOp) {
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

    explicit BarrierSimulator(mlir::MLIRContext* ctx, Logger log, int64_t numDmaEngines)
            : _ctx(ctx), _log(log), _numDmaEngines(numDmaEngines), _numRealBarriers() {
    }

    void buildTaskLists(mlir::FuncOp func);
    void assignVirtualIds(mlir::FuncOp func);
    int64_t getVirtualId(VPUIP::ConfigureBarrierOp op);
    mlir::LogicalResult checkProducerCount();
    mlir::LogicalResult run();

private:
    UpdateStatus updateTaskBarriers(const TaskInfo& taskInfo, const int64_t count);
    void logDebugInfo(const size_t barrier, const std::array<size_t, MAX_DMA_ENGINES>& dma, const size_t nce,
                      const size_t upa);

private:
    SmallVector<TaskInfo> _nceTasks;
    SmallVector<TaskInfo> _upaTasks;
    std::array<SmallVector<TaskInfo>, MAX_DMA_ENGINES> _dmaTasks;
    SmallVector<VPUIP::ConfigureBarrierOp> _barrierOps;

    std::map<int64_t, VirtualBarrierInfo> _virtualBarriers;
    SmallVector<int64_t> _barrierConfig;

    mlir::MLIRContext* _ctx;
    Logger _log;
    int64_t _numDmaEngines;
    int64_t _numRealBarriers;
};

void BarrierSimulator::assignVirtualIds(mlir::FuncOp func) {
    int64_t virtualId = 0;
    func.walk([&](VPUIP::ConfigureBarrierOp op) {
        op->setAttr(virtualIdAttrName, getIntAttr(_ctx, virtualId++));
    });
}

int64_t BarrierSimulator::getVirtualId(VPUIP::ConfigureBarrierOp op) {
    return checked_cast<int64_t>(op->getAttr(virtualIdAttrName).cast<mlir::IntegerAttr>().getInt());
}

void BarrierSimulator::buildTaskLists(mlir::FuncOp func) {
    _log.trace("Building task lists");

    auto getTaskInfo = [&](VPUIP::TaskOpInterface taskOp, const int64_t count = 1) {
        TaskInfo taskInfo(taskOp);
        for (auto waitBarrier : taskOp.waitBarriers()) {
            if (auto barrierOp = mlir::dyn_cast<VPUIP::ConfigureBarrierOp>(waitBarrier.getDefiningOp())) {
                const auto virtualId = getVirtualId(barrierOp);
                taskInfo.waitBarriers.push_back(virtualId);
                _virtualBarriers[virtualId].consumerCount += count;
                _virtualBarriers[virtualId].initConsumerCount += count;
            }
        }
        for (auto updateBarrier : taskOp.updateBarriers()) {
            if (auto barrierOp = mlir::dyn_cast<VPUIP::ConfigureBarrierOp>(updateBarrier.getDefiningOp())) {
                const auto virtualId = getVirtualId(barrierOp);
                taskInfo.updateBarriers.push_back(virtualId);
                _virtualBarriers[virtualId].producerCount += count;
                _virtualBarriers[virtualId].initProducerCount += count;
            }
        }
        return taskInfo;
    };

    func.walk([&](VPUIP::TaskOpInterface taskOp) {
        switch (taskOp.getTaskType()) {
        case VPUIP::TaskType::UPADMA:
        case VPUIP::TaskType::NNDMA: {
            int64_t port = 0;
            if (auto dmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(taskOp.getOperation()))
                port = dmaOp.port();
            VPUX_THROW_UNLESS(port < MAX_DMA_ENGINES,
                              "NNDMAOp port value ({0}) larger than maximum number of engines ({1})", port,
                              MAX_DMA_ENGINES);
            _dmaTasks[port].push_back(getTaskInfo(taskOp));
            break;
        }
        case VPUIP::TaskType::NCE2: {
            auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(taskOp.getOperation());
            _nceTasks.push_back(getTaskInfo(taskOp, nceOp.getNumVariants()));
            break;
        }
        // TODO: should we introduce _swTask?
        case VPUIP::TaskType::ACTShave:
        case VPUIP::TaskType::UPA: {
            _upaTasks.push_back(getTaskInfo(taskOp));
            break;
        }
        case VPUIP::TaskType::Controller: {
            if (mlir::dyn_cast<VPUIP::EmptyOp>(taskOp.getOperation()))
                break;
            auto barrierOp = mlir::dyn_cast<VPUIP::ConfigureBarrierOp>(taskOp.getOperation());
            _barrierOps.push_back(barrierOp);
            const auto virtualId = getVirtualId(barrierOp);
            _virtualBarriers[virtualId].realId = barrierOp.id();
            _numRealBarriers = std::max(_numRealBarriers, barrierOp.id() + 1);
            break;
        }
        default:
            VPUX_THROW("Unsupported task type '{0}'", taskOp.getTaskType());
        }
    });
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
            const auto virtualId = getVirtualId(_barrierOps[barrier]);
            const auto realId = _virtualBarriers[virtualId].realId;
            if (_barrierConfig[realId] > -1)
                break;
            _barrierConfig[realId] = virtualId;
            progressed = true;
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
            auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(_nceTasks[nce].taskOp.getOperation());
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
    _log.error("Barrier simulation blocked at BAR: {0} / {1} (virtual_id {2}); DMA: {3} / {4}, {5} / {6};"
               "NCE: {7} / {8}; UPA: {9} / {10}",
               barrier, _barrierOps.size(), getVirtualId(_barrierOps[barrier]), dma[0], _dmaTasks[0].size(), dma[1],
               _dmaTasks[1].size(), nce, _nceTasks.size(), upa, _upaTasks.size());

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

class BarrierSimulationPass final : public VPUIP::BarrierSimulationBase<BarrierSimulationPass> {
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

    const auto dmaAttr = VPUIP::DMAEngineAttr::get(&ctx, VPUIP::DMAEngine::DMA_NN);
    auto dmaResOp = resOp.getExecutor(dmaAttr);
    VPUX_THROW_UNLESS(dmaResOp != nullptr, "Failed to get DMA_NN information");

    const auto numDmaEngines = dmaResOp.count();
    VPUX_THROW_UNLESS(numDmaEngines <= MAX_DMA_ENGINES, "Found {0} DMA engines (max {1})", numDmaEngines,
                      MAX_DMA_ENGINES);

    BarrierSimulator simulator(&ctx, _log, numDmaEngines);
    simulator.assignVirtualIds(func);
    simulator.buildTaskLists(func);
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

std::unique_ptr<mlir::Pass> vpux::VPUIP::createBarrierSimulationPass(Logger log) {
    return std::make_unique<BarrierSimulationPass>(log);
}
