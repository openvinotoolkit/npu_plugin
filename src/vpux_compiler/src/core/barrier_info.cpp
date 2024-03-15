//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/dma.hpp"
#include "vpux/utils/core/range.hpp"

#include <llvm/ADT/SetOperations.h>

using namespace vpux;

//
// Constructor
//

vpux::BarrierInfo::BarrierInfo(mlir::func::FuncOp func)
        : _log(Logger::global().nest("barrier-info", 0)),
          _func(func),
          _taskIndexAttrName(mlir::StringAttr::get(func->getContext(), "task-index")),
          _barrierIndexAttrName(mlir::StringAttr::get(func->getContext(), "barrier-index")),
          _taskQueueTypeMap() {
    buildBarrierMaps(func);
}

//
// clearAttributes
//

void vpux::BarrierInfo::clearAttributes() {
    auto removeAttributeFromRange = [](mlir::StringAttr attrName, auto range) {
        for (auto op : range) {
            VPUX_THROW_UNLESS(op->hasAttr(attrName), "Remove: attribute '{0}' was not set for '{1}' operation at '{2}'",
                              attrName, op->getName(), op->getLoc());
            op->removeAttr(attrName);
        }
    };

    removeAttributeFromRange(_taskIndexAttrName, _allTaskOps);
    removeAttributeFromRange(_barrierIndexAttrName, _allBarrierOps);
}

//
// getIndex (TaskOp)
//

uint32_t vpux::BarrierInfo::getIndex(VPURT::TaskOp taskOp) const {
    const auto attr = taskOp->getAttrOfType<mlir::IntegerAttr>(_taskIndexAttrName);
    VPUX_THROW_UNLESS(attr != nullptr, "Get: attribute '{0}' was not set for '{1}' operation at '{2}'",
                      _taskIndexAttrName, taskOp->getName(), taskOp->getLoc());

    return checked_cast<uint32_t>(attr.getValue().getZExtValue());
}

//
// getIndex (DeclareVirtualBarrierOp)
//

uint32_t vpux::BarrierInfo::getIndex(VPURT::DeclareVirtualBarrierOp barrierOp) const {
    const auto attr = barrierOp->getAttrOfType<mlir::IntegerAttr>(_barrierIndexAttrName);
    VPUX_THROW_UNLESS(attr != nullptr, "Get: attribute '{0}' was not set for '{1}' operation at '{2}'",
                      _barrierIndexAttrName, barrierOp->getName(), barrierOp->getLoc());

    return checked_cast<uint32_t>(attr.getValue().getZExtValue());
}

//
// getTaskOpAtIndex
//

VPURT::TaskOp vpux::BarrierInfo::getTaskOpAtIndex(size_t opIdx) const {
    VPUX_THROW_UNLESS(_allTaskOps.size() > opIdx, "Task: Invalid index '{0}' for _allTaskOps", opIdx);
    VPUX_THROW_UNLESS(getIndex(_allTaskOps[opIdx]) == opIdx, "Task: Index not matching '{0}'", opIdx);
    return _allTaskOps[opIdx];
}

//
// getBarrierOpAtIndex
//

VPURT::DeclareVirtualBarrierOp vpux::BarrierInfo::getBarrierOpAtIndex(size_t opIdx) const {
    VPUX_THROW_UNLESS(_allBarrierOps.size() > opIdx, "Barrier: Invalid index '{0}' for _allBarrierOps", opIdx);
    VPUX_THROW_UNLESS(getIndex(_allBarrierOps[opIdx]) == opIdx, "Barrier: Index not matching '{0}' '{1}'", opIdx,
                      _allBarrierOps[opIdx]);
    return _allBarrierOps[opIdx];
}

//
// getWaitBarriers
//

BarrierInfo::TaskSet& vpux::BarrierInfo::getWaitBarriers(size_t taskInd) {
    VPUX_THROW_UNLESS(taskInd <= _taskWaitBarriers.size(), "Task not found in _taskWaitBarriers, '{0}'", taskInd);
    return _taskWaitBarriers[taskInd];
}

//
// getUpdateBarriers (by Idn)
//

BarrierInfo::TaskSet& vpux::BarrierInfo::getUpdateBarriers(size_t taskInd) {
    VPUX_THROW_UNLESS(taskInd < _taskUpdateBarriers.size(), "Task not found in _taskUpdateBarriers, '{0}'", taskInd);
    return _taskUpdateBarriers[taskInd];
}

//
// getBarrierProducers
//

BarrierInfo::TaskSet& vpux::BarrierInfo::getBarrierProducers(size_t barrierInd) {
    VPUX_THROW_UNLESS(barrierInd <= _barrierProducerMap.size(), "Barrier not found in _barrierProducerMap, '{0}'",
                      barrierInd);
    return _barrierProducerMap[barrierInd];
}

//
// getBarrierConsumers
//

BarrierInfo::TaskSet& vpux::BarrierInfo::getBarrierConsumers(size_t barrierInd) {
    VPUX_THROW_UNLESS(barrierInd <= _barrierConsumerMap.size(), "Barrier not found in _barrierConsumerMap, '{0}'",
                      barrierInd);
    return _barrierConsumerMap[barrierInd];
}

//
// getBarrierProducers
//

BarrierInfo::TaskSet& vpux::BarrierInfo::getBarrierProducers(VPURT::DeclareVirtualBarrierOp barrierOp) {
    auto barrierInd = getIndex(barrierOp);
    return getBarrierProducers(barrierInd);
}

//
// getBarrierConsumers
//

BarrierInfo::TaskSet& vpux::BarrierInfo::getBarrierConsumers(VPURT::DeclareVirtualBarrierOp barrierOp) {
    auto barrierInd = getIndex(barrierOp);
    return getBarrierConsumers(barrierInd);
}

//
// getNumOfSlotsUsed
//

size_t vpux::BarrierInfo::getNumOfSlotsUsed(VPURT::TaskOp op) const {
    // This function returns the number of variants used by a task for a barrier.
    // On VPU H/W, a NCE task is executed (across multiple DPUs) via workloads descriptors (known as variants).
    // Each variant must update the barrier to signal that it is complete.
    // An NCE task may have multiple workloads descriptors (which are generated in the NCE DPU workloads pass).
    // Therefore, the number of variants must be verified as they will all update a barrier and
    // contribute to the architecture specific MAX VARIANT COUNT that a barrier has.
    // A DMA/UPA does not have variants, therefore they always just requires 1 producer slot to a barrier.

    if (op.getExecutorKind() == VPU::ExecutorKind::DPU) {
        auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(op.getInnerTaskOp());
        VPUX_THROW_UNLESS(nceOp != nullptr, "Could not cast to NCE task");
        return nceOp.getNumVariants();
    }

    if (op.getExecutorKind() == VPU::ExecutorKind::SHAVE_ACT) {
        if (auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(op.getInnerTaskOp())) {
            auto swKernelRun = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
            return std::distance(swKernelRun.begin(), swKernelRun.end());
        }
        return 1;
    }

    if (op.getExecutorKind() == VPU::ExecutorKind::DMA_NN || op.getExecutorKind() == VPU::ExecutorKind::SHAVE_UPA) {
        return 1;
    }

    VPUX_THROW("Unsupported executor: {0}", op.getExecutorKind());
}

//
// getProducerSlotCount
//

size_t vpux::BarrierInfo::getProducerSlotCount(VPURT::DeclareVirtualBarrierOp barrierOp) {
    size_t producerSlotCount = 0;
    for (const auto& producer : getBarrierProducers(barrierOp)) {
        producerSlotCount += getNumOfSlotsUsed(getTaskOpAtIndex(producer));
    }
    return producerSlotCount;
}

//
// getConsumerSlotCount
//

size_t vpux::BarrierInfo::getConsumerSlotCount(VPURT::DeclareVirtualBarrierOp barrierOp) {
    size_t consumerSlotCount = 0;
    for (const auto& consumer : getBarrierConsumers(barrierOp)) {
        consumerSlotCount += getNumOfSlotsUsed(getTaskOpAtIndex(consumer));
    }
    return consumerSlotCount;
}

//
// getNumOfVirtualBarriers
//

size_t vpux::BarrierInfo::getNumOfVirtualBarriers() const {
    return _allBarrierOps.size();
}

//
// getNumOfTasks
//

size_t vpux::BarrierInfo::getNumOfTasks() const {
    return _allTaskOps.size();
}

//
// resizeBitMap
//

void vpux::BarrierInfo::resizeBitMap(SmallVector<llvm::BitVector>& bitMap, size_t length, uint32_t bits) {
    bitMap.resize(length);
    for (auto& bit : bitMap) {
        bit.resize(bits);
    }
}

//
//  producersControlsAllConsumers
//

bool vpux::BarrierInfo::producersControlsAllConsumers(const TaskSet& origProducers, const TaskSet& newConsumers,
                                                      const TaskSet& origConsumers,
                                                      ArrayRef<TaskSet> origWaitBarriersMap) {
    // Get new consumers not in original consumers

    auto consumersWithoutDirectControl = llvm::set_difference(newConsumers, origConsumers);
    if (consumersWithoutDirectControl.empty()) {
        return true;
    }
    if (origProducers.empty()) {
        return false;
    }

    auto consumerHasImplicitTaskQueueType = inImplicitQueueTypeDependencyList(consumersWithoutDirectControl);
    if (!consumerHasImplicitTaskQueueType) {
        return false;
    }
    auto producerHasImplicitTaskQueueType = inImplicitQueueTypeDependencyList(origProducers);
    if (!producerHasImplicitTaskQueueType) {
        return false;
    }

    if (*std::max_element(origProducers.begin(), origProducers.end()) >=
        *std::min_element(consumersWithoutDirectControl.begin(), consumersWithoutDirectControl.end())) {
        return false;
    }

    auto anyProducerCanRunParallelWithConsumer = llvm::any_of(origProducers, [&](const auto& producer) {
        return llvm::any_of(consumersWithoutDirectControl, [&](const auto& consumer) {
            return origWaitBarriersMap[producer] == origWaitBarriersMap[consumer];
        });
    });
    return !anyProducerCanRunParallelWithConsumer;
}

//
// inImplicitQueueTypeDependencyList
//

bool vpux::BarrierInfo::inImplicitQueueTypeDependencyList(const TaskSet& taskList) {
    auto allTasksAreInImplicitQueueTypeDependencyList = llvm::all_of(taskList, [&](const auto& taskInd) {
        for (const auto& item : _taskQueueTypeMap) {
            if (item.second.test(taskInd)) {
                return true;
            }
        }
        return false;
    });
    return allTasksAreInImplicitQueueTypeDependencyList;
}

//
// buildBarrierMaps
//

void vpux::BarrierInfo::buildBarrierMaps(mlir::func::FuncOp func) {
    _log.trace("Collect initial producer maps");

    _allTaskOps = to_small_vector(func.getOps<VPURT::TaskOp>());
    _allBarrierOps = to_small_vector(func.getOps<VPURT::DeclareVirtualBarrierOp>());

    _log.nest().trace("There are '{0}' VPURT::TaskOp", _allTaskOps.size());
    _log.nest().trace("There are '{0}' VPURT::DeclareVirtualBarrierOp", _allBarrierOps.size());

    // resize bit maps
    _taskWaitBarriers = SmallVector<TaskSet>(_allTaskOps.size(), {});
    _taskUpdateBarriers = SmallVector<TaskSet>(_allTaskOps.size(), {});
    _barrierConsumerMap = SmallVector<TaskSet>(_allBarrierOps.size(), {});
    _barrierProducerMap = SmallVector<TaskSet>(_allBarrierOps.size(), {});
    resizeBitMap(_taskControlMap, _allTaskOps.size(), checked_cast<uint32_t>(_allTaskOps.size()));

    // resize implicit dependency map
    const auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto dmaPortNum = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN).getCount();

    auto dmaChannels = getDMAChannelsWithIndependentLinkAgents(VPU::getArch(module));
    for (auto dmaPortIdx : irange(dmaPortNum)) {
        for (auto dmaChannel : dmaChannels) {
            VPURT::TaskQueueType taskQueueType;
            taskQueueType.type = VPU::ExecutorKind::DMA_NN;
            taskQueueType.id = getDMAQueueIdEncoding(dmaPortIdx, dmaChannel);
            llvm::BitVector taskList(checked_cast<uint32_t>(_allTaskOps.size()));
            _taskQueueTypeMap.insert(std::make_pair(taskQueueType, taskList));
        }
    }

    // set index to task ops and barrier ops
    for (const auto& p : _allTaskOps | indexed) {
        p.value()->setAttr(_taskIndexAttrName, getIntAttr(p.value().getContext(), p.index()));
    }
    for (const auto& p : _allBarrierOps | indexed) {
        p.value()->setAttr(_barrierIndexAttrName, getIntAttr(p.value().getContext(), p.index()));
    }

    for (auto& op : func.getOps()) {
        if (auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(op)) {
            addTaskOp(taskOp);
        } else if (auto barrierOp = mlir::dyn_cast<VPURT::DeclareVirtualBarrierOp>(op)) {
            _log.nest().trace("Found 'VPURT::DeclareVirtualBarrierOp' Operation at '{0}'", op.getLoc());
            VPUX_THROW_WHEN(barrierOp.getBarrier() == nullptr, "DeclareVirtualBarrierOp '{0}' has no barrier.",
                            barrierOp);

            // ensure all barrier users are TaskOps
            for (auto* user : barrierOp.getBarrier().getUsers()) {
                VPUX_THROW_WHEN(mlir::dyn_cast<VPURT::TaskOp>(user) == nullptr, "Got non-TaskOp Operation at '{0}'",
                                op.getLoc());
            }
        }
    }
}

//
// addConsumer
//

void vpux::BarrierInfo::addConsumer(VPURT::DeclareVirtualBarrierOp barrierOp, size_t taskInd) {
    const auto barrierInd = getIndex(barrierOp);
    _log.trace("Add consumer '{0}' for barrier '{1}'", taskInd, barrierInd);
    _barrierConsumerMap[barrierInd].insert(taskInd);
    _taskWaitBarriers[taskInd].insert(barrierInd);
}

//
// addConsumers
//

void vpux::BarrierInfo::addConsumers(size_t barrierInd, const TaskSet& taskInds) {
    for (const auto& taskInd : taskInds) {
        _barrierConsumerMap[barrierInd].insert(taskInd);
        _taskWaitBarriers[taskInd].insert(barrierInd);
    }
}

//
// addProducer
//

void vpux::BarrierInfo::addProducer(VPURT::DeclareVirtualBarrierOp barrierOp, size_t taskInd) {
    const auto barrierInd = getIndex(barrierOp);
    _log.trace("Add producer '{0}' for barrier '{1}'", taskInd, barrierInd);
    _barrierProducerMap[barrierInd].insert(taskInd);
    _taskUpdateBarriers[taskInd].insert(barrierInd);
}

//
// addProducers
//

void vpux::BarrierInfo::addProducers(size_t barrierInd, const TaskSet& taskInds) {
    for (const auto& taskInd : taskInds) {
        _barrierProducerMap[barrierInd].insert(taskInd);
        _taskUpdateBarriers[taskInd].insert(barrierInd);
    }
}

//
// removeProducers
//

void vpux::BarrierInfo::removeProducers(VPURT::DeclareVirtualBarrierOp barrierOp, const TaskSet& taskInds) {
    const auto barrierInd = getIndex(barrierOp);
    for (const auto& taskInd : taskInds) {
        _barrierProducerMap[barrierInd].erase(taskInd);
        _taskUpdateBarriers[taskInd].erase(barrierInd);
    }
}

//
// removeConsumers
//

void vpux::BarrierInfo::removeConsumers(VPURT::DeclareVirtualBarrierOp barrierOp, const TaskSet& taskInds) {
    const auto barrierInd = getIndex(barrierOp);
    for (const auto& taskInd : taskInds) {
        _barrierConsumerMap[barrierInd].erase(taskInd);
        _taskWaitBarriers[taskInd].erase(barrierInd);
    }
}

//
// addTaskOp
//

void vpux::BarrierInfo::addTaskOp(VPURT::TaskOp taskOp) {
    const auto taskInd = getIndex(taskOp);
    _log.trace("Found 'TaskOp' Operation '{0}'", taskInd);

    for (const auto& bar : taskOp.getWaitBarriers()) {
        // Note: can also be VPURT::ConfigureBarrierOp
        auto barrierOp = bar.getDefiningOp<VPURT::DeclareVirtualBarrierOp>();
        VPUX_THROW_WHEN(barrierOp == nullptr, "Invalid wait barrier op type {0}", bar);
        addConsumer(barrierOp, taskInd);
    }

    for (const auto& bar : taskOp.getUpdateBarriers()) {
        // Note: can also be VPURT::ConfigureBarrierOp
        auto barrierOp = bar.getDefiningOp<VPURT::DeclareVirtualBarrierOp>();
        VPUX_THROW_WHEN(barrierOp == nullptr, "Invalid wait barrier op type {0}", bar);
        addProducer(barrierOp, taskInd);
    }
}

//
// addNewBarrier
//

void vpux::BarrierInfo::addNewBarrier(VPURT::DeclareVirtualBarrierOp barrierOp) {
    size_t barrierInd = _allBarrierOps.size();
    barrierOp->setAttr(_barrierIndexAttrName, getIntAttr(barrierOp.getContext(), barrierInd));

    _log.trace("Add new barrier '{0}', new barrier size '{1}'", barrierInd, _allBarrierOps.size());
    _allBarrierOps.push_back(barrierOp);

    _barrierConsumerMap.push_back({});
    _barrierProducerMap.push_back({});
}

//
// setWaitBarriers
//

void vpux::BarrierInfo::setWaitBarriers(size_t taskInd, const TaskSet& barriers) {
    // remove previous wait barriers
    for (auto barrierInd : _taskWaitBarriers[taskInd]) {
        _barrierConsumerMap[static_cast<size_t>(barrierInd)].erase(taskInd);
    }

    for (auto barrierInd : barriers) {
        _barrierConsumerMap[barrierInd].insert(taskInd);
    }
    _taskWaitBarriers[taskInd] = barriers;
}

//
// setUpdateBarriers
//

void vpux::BarrierInfo::setUpdateBarriers(size_t taskInd, const TaskSet& barriers) {
    // remove previous update barriers
    for (auto barrierInd : _taskUpdateBarriers[taskInd]) {
        _barrierProducerMap[static_cast<size_t>(barrierInd)].erase(taskInd);
    }

    for (auto barrierInd : barriers) {
        _barrierProducerMap[barrierInd].insert(taskInd);
    }
    _taskUpdateBarriers[taskInd] = barriers;
}

//
// removeProducer
//

void vpux::BarrierInfo::removeProducer(size_t taskInd, VPURT::DeclareVirtualBarrierOp barrierOp) {
    const auto barrierInd = getIndex(barrierOp);
    _barrierProducerMap[barrierInd].erase(taskInd);
    _taskUpdateBarriers[taskInd].erase(barrierInd);
}

//
// removeConsumer
//

void vpux::BarrierInfo::removeConsumer(size_t taskInd, VPURT::DeclareVirtualBarrierOp barrierOp) {
    const auto barrierInd = getIndex(barrierOp);
    _barrierConsumerMap[barrierInd].erase(taskInd);
    _taskWaitBarriers[taskInd].erase(barrierInd);
}

//
// resetBarrier
//

void vpux::BarrierInfo::resetBarrier(VPURT::DeclareVirtualBarrierOp barrierOp) {
    const auto barrierInd = getIndex(barrierOp);
    resetBarrier(checked_cast<size_t>(barrierInd));
}

//
// resetBarrier
//

void vpux::BarrierInfo::resetBarrier(size_t barrierInd) {
    _log.trace("Reset barrier '{0}'", barrierInd);

    for (auto taskInd : _barrierProducerMap[barrierInd]) {
        _taskUpdateBarriers[static_cast<size_t>(taskInd)].erase(barrierInd);
    }
    _barrierProducerMap[barrierInd].clear();

    for (auto taskInd : _barrierConsumerMap[barrierInd]) {
        _taskWaitBarriers[static_cast<size_t>(taskInd)].erase(barrierInd);
    }
    _barrierConsumerMap[barrierInd].clear();
}

//
// optimizeBarriers
//

void vpux::BarrierInfo::optimizeBarriers() {
    // A -> B -> C

    // If B depends on A and C depends on [A, B] ==> we can remove A from C deps list,
    // since it will be implicit dependency taken from B.
    // Note: It also will merge barriers which have the same producers but different consumers

    // Barrier are optimized based on order of task ops

    // TODO: E#79318 optimize loops

    _log.trace("Optimize producers / update barriers");

    auto setBarrierMask = [&](llvm::BitVector& mask, const BarrierInfo::TaskSet& barriers) {
        for (auto bar : barriers) {
            mask.set(bar);
        }
    };

    SmallVector<llvm::BitVector> updateBarriers;
    resizeBitMap(updateBarriers, _allTaskOps.size(), checked_cast<uint32_t>(_allBarrierOps.size()));

    for (size_t taskInd = _allTaskOps.size(); taskInd-- > 0;) {
        setBarrierMask(updateBarriers[taskInd], _taskUpdateBarriers[taskInd]);
        for (auto updateBarrierInd : _taskUpdateBarriers[taskInd]) {
            for (auto childTaskInd : _barrierConsumerMap[static_cast<size_t>(updateBarrierInd)]) {
                updateBarriers[taskInd] |= updateBarriers[static_cast<size_t>(childTaskInd)];
            }
        }
    }

    for (size_t taskInd = 0; taskInd < _allTaskOps.size(); ++taskInd) {
        for (auto updateBarrierInd : _taskUpdateBarriers[taskInd]) {
            for (auto childTaskInd : _barrierConsumerMap[static_cast<size_t>(updateBarrierInd)]) {
                updateBarriers[taskInd].reset(updateBarriers[static_cast<size_t>(childTaskInd)]);
            }
        }
        TaskSet targetUpdateBarriers;
        for (auto bar : updateBarriers[taskInd].set_bits()) {
            targetUpdateBarriers.insert(bar);
        }
        setUpdateBarriers(taskInd, targetUpdateBarriers);
    }

    // optimize barriers which have the same producers but different consumers
    for (size_t barIdn = 0; barIdn < _allBarrierOps.size(); ++barIdn) {
        for (size_t childBarIdn = barIdn + 1; childBarIdn < _allBarrierOps.size(); ++childBarIdn) {
            if (_barrierProducerMap[barIdn] == _barrierProducerMap[childBarIdn]) {
                _log.nest().trace("Same producers for barId '{0}' '{1}'", barIdn, childBarIdn);
                auto producers = _barrierProducerMap[barIdn].size();
                auto consumersBar = _barrierConsumerMap[barIdn].size();
                auto consumersChildBar = _barrierConsumerMap[childBarIdn].size();
                if (producers + consumersBar + consumersChildBar <= VPUIP::getBarrierMaxVariantSum(_func)) {
                    for (auto consumerInd : _barrierConsumerMap[childBarIdn]) {
                        // move all consumers to one barrier
                        addConsumer(getBarrierOpAtIndex(barIdn), static_cast<size_t>(consumerInd));
                    }
                    _log.trace("New consumers number - {0}", getConsumerSlotCount(getBarrierOpAtIndex(barIdn)));
                    resetBarrier(getBarrierOpAtIndex(childBarIdn));
                }
            }
        }
    }

    // optimize consumers
    _log.trace("Optimize consumers / wait barriers");
    SmallVector<llvm::BitVector> waitBarriers;
    resizeBitMap(waitBarriers, _allTaskOps.size(), checked_cast<uint32_t>(_allBarrierOps.size()));

    for (size_t taskInd = 0; taskInd < _allTaskOps.size(); ++taskInd) {
        setBarrierMask(waitBarriers[taskInd], _taskWaitBarriers[taskInd]);
        for (auto waitBarrierInd : _taskWaitBarriers[taskInd]) {
            for (auto parentTaskInd : _barrierProducerMap[waitBarrierInd]) {
                waitBarriers[taskInd] |= waitBarriers[static_cast<size_t>(parentTaskInd)];
            }
        }
    }

    for (size_t taskInd = _allTaskOps.size(); taskInd-- > 0;) {
        for (auto waitBarrierInd : _taskWaitBarriers[taskInd]) {
            for (auto parentTaskInd : _barrierProducerMap[waitBarrierInd]) {
                waitBarriers[taskInd].reset(waitBarriers[static_cast<size_t>(parentTaskInd)]);
            }
        }

        TaskSet targetWaitBarriers;
        for (auto bar : waitBarriers[taskInd].set_bits()) {
            targetWaitBarriers.insert(bar);
        }
        setWaitBarriers(taskInd, targetWaitBarriers);
    }
}

//
// buildTaskControlMap
//

void vpux::BarrierInfo::buildTaskControlMap(bool considerTaskFifoDependency) {
    if (considerTaskFifoDependency) {
        for (const auto& taskOp : _allTaskOps | reversed) {
            auto taskInd = getIndex(taskOp);
            auto taskQueueType = VPURT::getTaskQueueType(taskOp, false);
            if (_taskQueueTypeMap.find(taskQueueType) != _taskQueueTypeMap.end()) {
                _taskQueueTypeMap[taskQueueType].set(taskInd);
            }
        }

        for (const auto& item : _taskQueueTypeMap) {
            const auto& tasksInSameFIFO = item.second;
            for (auto taskInd : tasksInSameFIFO.set_bits()) {
                for (auto nextTaskInd = tasksInSameFIFO.find_next(taskInd); nextTaskInd != -1;
                     nextTaskInd = tasksInSameFIFO.find_next(nextTaskInd)) {
                    _taskControlMap[taskInd].set(nextTaskInd);
                }
            }
        }
    }

    for (size_t taskInd = _allTaskOps.size(); taskInd-- > 0;) {
        for (auto updateBarrierInd : _taskUpdateBarriers[taskInd]) {
            for (auto consumerIdx : _barrierConsumerMap[static_cast<size_t>(updateBarrierInd)]) {
                _taskControlMap[taskInd].set(consumerIdx);
            }
        }
    }
    for (size_t taskInd = _allTaskOps.size(); taskInd-- > 0;) {
        for (auto updateBarrierInd : _taskUpdateBarriers[taskInd]) {
            for (auto childTaskInd : _barrierConsumerMap[static_cast<size_t>(updateBarrierInd)]) {
                _taskControlMap[taskInd] |= _taskControlMap[static_cast<size_t>(childTaskInd)];
            }
        }
    }
}

//
// controlPathExistsBetween
//

bool vpux::BarrierInfo::controlPathExistsBetween(size_t taskAInd, size_t taskBInd, bool biDirection) const {
    // ensure that _taskControlMap is build at given time with buildTaskControlMap()
    if (biDirection) {
        return _taskControlMap[taskAInd][taskBInd] || _taskControlMap[taskBInd][taskAInd];
    }
    return _taskControlMap[taskAInd][taskBInd];
}

//
// updateIR
//

void vpux::BarrierInfo::updateIR() {
    // update IR using stored dependencies
    for (size_t taskInd = 0; taskInd < _allTaskOps.size(); ++taskInd) {
        auto taskOp = getTaskOpAtIndex(taskInd);

        taskOp.getWaitBarriersMutable().clear();
        for (auto barrierInd : _taskWaitBarriers[taskInd]) {
            auto barrierOp = getBarrierOpAtIndex(static_cast<size_t>(barrierInd));
            taskOp.getWaitBarriersMutable().append(barrierOp.getBarrier());
        }

        taskOp.getUpdateBarriersMutable().clear();
        for (auto barrierInd : _taskUpdateBarriers[taskInd]) {
            auto barrierOp = getBarrierOpAtIndex(static_cast<size_t>(barrierInd));
            taskOp.getUpdateBarriersMutable().append(barrierOp.getBarrier());
        }
    }
}

//
// logBarrierInfo
//

void vpux::BarrierInfo::logBarrierInfo() {
    // useful for logging and debugging barrier dependencies
    _log.setName("barrier-info-log");
    _log.trace("Logging BarrierInfo");

    for (size_t taskInd = 0; taskInd < _allTaskOps.size(); ++taskInd) {
        _log.nest().trace("Task '{0}'", taskInd);
        for (const auto& barrierOp : getWaitBarriers(taskInd)) {
            _log.nest(2).trace("Task '{0}' waits for '{1}'", taskInd, barrierOp);
        }
        for (const auto& barrierOp : getUpdateBarriers(taskInd)) {
            _log.nest(2).trace("Task '{0}' updates '{1}'", taskInd, barrierOp);
        }
    }
}

//
// createLegalVariantBatches
//

SmallVector<BarrierInfo::TaskSet> vpux::BarrierInfo::createLegalVariantBatches(const TaskSet& tasks,
                                                                               size_t availableSlots) {
    // store batches of tasks
    SmallVector<TaskSet> legalBatches(1);

    // store total slot count used by batch
    size_t totalSlotCount = 0;

    const auto isLegalVariantCountWith = [&](size_t numSlotsUsedByTask) -> bool {
        return (totalSlotCount + numSlotsUsedByTask) <= availableSlots;
    };

    // create batches for new barriers
    auto orderedTasks = std::set<size_t>(tasks.begin(), tasks.end());
    for (const auto& taskInd : orderedTasks) {
        // find number of slots consumed by this task
        auto numSlotsUsedByTask = getNumOfSlotsUsed(getTaskOpAtIndex(taskInd));

        // check if new batch needs to be created
        if (!isLegalVariantCountWith(numSlotsUsedByTask)) {
            legalBatches.push_back({});
            totalSlotCount = 0;
        }

        legalBatches.rbegin()->insert(taskInd);
        totalSlotCount += numSlotsUsedByTask;
    }

    return legalBatches;
}

//
// haveSameImplicitDependencyTaskQueueType
//

std::optional<VPURT::TaskQueueType> vpux::BarrierInfo::haveSameImplicitDependencyTaskQueueType(
        const TaskSet& taskInds) {
    if (taskInds.empty()) {
        return std::nullopt;
    }
    auto firstTaskInd = *taskInds.begin();
    for (const auto& item : _taskQueueTypeMap) {
        // get the task queue type for the first task, then check all the other tasks have same task queue type
        if (!item.second.test(firstTaskInd)) {
            continue;
        }

        for (const auto& taskInd : taskInds) {
            if (!item.second.test(taskInd)) {
                return std::nullopt;
            }
        }
        return item.first;
    }
    return std::nullopt;
}

//
// canBarriersBeMerged
//

bool vpux::BarrierInfo::canBarriersBeMerged(const TaskSet& barrierProducersA, const TaskSet& barrierConsumersA,
                                            const TaskSet& barrierProducersB, const TaskSet& barrierConsumersB,
                                            ArrayRef<TaskSet> origWaitBarriersMap) {
    // two barriers A and B can be merged if
    // 1. any producer of barrier A controls any consumer of barrier B
    if (!producersControlsAllConsumers(barrierProducersA, barrierConsumersB, barrierConsumersA, origWaitBarriersMap)) {
        return false;
    }

    // 2. any producer of barrier B controls any consumer of barrier A
    if (!producersControlsAllConsumers(barrierProducersB, barrierConsumersA, barrierConsumersB, origWaitBarriersMap)) {
        return false;
    }

    return true;
}

//
// getWaitBarriersMap
//

SmallVector<BarrierInfo::TaskSet> vpux::BarrierInfo::getWaitBarriersMap() {
    return _taskWaitBarriers;
}

//
// getFinalTasks
//

SmallVector<VPURT::TaskOp> vpux::BarrierInfo::getFinalTasks() {
    SmallVector<VPURT::TaskOp> finalTasks;
    for (const auto& p : _taskControlMap | indexed) {
        if (p.value().none()) {
            finalTasks.push_back(getTaskOpAtIndex(p.index()));
        }
    }
    return finalTasks;
}
