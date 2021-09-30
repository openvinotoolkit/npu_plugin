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

#include "vpux/compiler/core/list_scheduler.hpp"

#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/range.hpp"

using namespace vpux;

//
// Constructor
//

ListScheduler::ListScheduler(mlir::Attribute& memSpace, MemLiveRangeInfo& liveRangeInfo, AsyncDepsInfo& depsInfo,
                             LinearScan<mlir::Value, LinearScanHandler>& scan)
        : _memSpace(memSpace), _liveRangeInfo(liveRangeInfo), _depsInfo(depsInfo), _scan(scan) {
    // _startTimeHeap;
    // _completionTimeHeap;
    // _activeComputeOps;
    // _readyComputeOps;
    // _readyDataOps;
    // _inDegreeTable;
    // _outDegreeTable;
    // _timeBuckets;
    // _currentTime;
}

void ListScheduler::push_to_st_heap(const heap_element_t& elem) {
    _startTimeHeap.push_back(elem);
    std::push_heap(_startTimeHeap.begin(), _startTimeHeap.end(), min_heap_ordering_t());
}

ListScheduler::heap_element_t ListScheduler::pop_from_st_heap() {
    std::pop_heap(_startTimeHeap.begin(), _startTimeHeap.end(), min_heap_ordering_t());
    heap_element_t elem = _startTimeHeap.back();
    _startTimeHeap.pop_back();
    return elem;
}

void ListScheduler::push_to_ct_heap(const heap_element_t& elem) {
    _completionTimeHeap.push_back(elem);
    std::push_heap(_completionTimeHeap.begin(), _completionTimeHeap.end(), min_heap_ordering_t());
}

ListScheduler::heap_element_t ListScheduler::pop_from_ct_heap() {
    std::pop_heap(_completionTimeHeap.begin(), _completionTimeHeap.end(), min_heap_ordering_t());
    heap_element_t elem = _completionTimeHeap.back();
    _completionTimeHeap.pop_back();
    return elem;
}

bool ListScheduler::is_data_op(size_t opIdx) {
    StringRef dataType("DMA_NN");

    mlir::async::ExecuteOp op = _depsInfo.getExecuteOpAtIndex(opIdx);
    if (op->getAttr("IERT.executor").cast<mlir::StringAttr>().getValue().compare(dataType) != 0) {
        return false;
    }

    if (op.getOperands().size() == 0) {
        return true;
    }

    auto inOp = op.body().front().back().getOperand(0);
    bool CMX = true;
    if (auto memref = inOp.getType().dyn_cast<mlir::MemRefType>()) {
        if (memref.getMemorySpace()) {
            // Logger::global().error("mem space: {0}", memref.getMemorySpace());
            CMX = CMX && (_memSpace == memref.getMemorySpace());
        }
    }

    return CMX;
}

ListScheduler::heap_element_t const* ListScheduler::top_element_gen(const std::vector<heap_element_t>& heap) const {
    return heap.empty() ? NULL : &(heap.front());
}

std::vector<ListScheduler::heap_element_t> ListScheduler::pop_all_elements_at_this_time(size_t time_step) {
    std::vector<heap_element_t> poppedOps;
    heap_element_t const* top_ptr = NULL;
    while ((top_ptr = top_element_gen(_completionTimeHeap)) && (top_ptr->time_ == time_step)) {
        std::cout << "inside top is " << top_ptr->op_ << std::endl;
        poppedOps.push_back(pop_from_ct_heap());
        std::push_heap(poppedOps.begin(), poppedOps.end(), min_heap_ordering_t());
    }
    return poppedOps;
}

void ListScheduler::unschedule_op(const heap_element_t& helement) {
    std::cout << "unschedule " << helement.op_ << std::endl;

    auto op = _depsInfo.getExecuteOpAtIndex(helement.op_);
    const auto usedBufs = _liveRangeInfo.getUsedBuffers(op);
    for (auto val : usedBufs) {
        if (_liveRangeInfo.eraseUser(val, op) == 0) {
            std::cout << "mark as dead bufffer of size: " << _scan.handler().getSize(val) << std::endl;
            // val.dump();
            _scan.handler().markAsDead(val);
        }
    }
    _scan.freeNonAlive();

    // TODO: verify below
    for (auto dep : _depsInfo.getOpDeps(helement.op_)) {
        _opOutputTable.find(dep)->second.decrement_consumers();
    }

    auto _opOutput = _opOutputTable.find(helement.op_);
    if (_opOutput->second.consumed()) {
        _opOutput->second.change_state_to_consumed();
    }

    // TODO
}

bool ListScheduler::is_compute_op_with_some_active_inputs(size_t opIdx) {
    if (is_data_op(opIdx)) {
        return false;
    }
    // TODO: might need to port: active_resource_table

    auto opAllDeps = _depsInfo.getOpDeps(opIdx);
    auto opDeps = get_sorted_buffers(opIdx);

    // number of buffers needing allocation smaller than number of buffers used by the op
    return opAllDeps.size() > opDeps.size();
}

void ListScheduler::distribute_ready_ops(std::list<size_t> readyOps) {
    for (auto readyOp : readyOps) {
        if (is_data_op(readyOp)) {
            assert(_readyDataOps.find(op) == _readyDataOps.end());
            std::cout << "new ready data op " << readyOp << std::endl;
            _readyDataOps.insert(readyOp);
            std::list<size_t> newReadyOps = reduce_in_degree_of_adjacent_operations(readyOp);
            distribute_ready_ops(newReadyOps);
        } else if (is_compute_op_with_some_active_inputs(readyOp)) {
            assert(_activeComputeOps.find(readyOp) == _activeComputeOps.end());
            std::cout << "new ready active op " << readyOp << std::endl;
            _activeComputeOps.insert(readyOp);
        } else {
            std::cout << "new ready compute op " << readyOp << std::endl;
            _readyComputeOps.insert(readyOp);
        }
    }
}

void ListScheduler::unschedule_all_completing_ops_at_next_earliest_time() {
    const heap_element_t* completion_top_ptr = top_element_gen(_completionTimeHeap);
    _currentTime = completion_top_ptr->time_;

    std::cout << "got top from ct heap with time " << _currentTime << std::endl;

    std::vector<heap_element_t> unsched_ops = pop_all_elements_at_this_time(_currentTime);
    std::list<size_t> ready_ops;

    for (auto& op : unsched_ops) {
        auto opIdx = op.op_;
        unschedule_op(op);

        if (!is_data_op(opIdx) && op.is_original_op()) {
            auto new_ready_ops = reduce_in_degree_of_adjacent_operations(opIdx);
            ready_ops.insert(ready_ops.end(), new_ready_ops.begin(), new_ready_ops.end());
        }
    }

    distribute_ready_ops(ready_ops);
}

std::list<size_t> ListScheduler::reduce_in_degree_of_adjacent_operations(size_t opIdx) {
    std::list<size_t> zeroIndegreeOps;
    // reduce indegree for consumers of ready data ops
    for (auto consumer : _depsInfo.getConsumerOps(opIdx)) {
        assert(_inDegreeTable[consumer] > 0);
        if (_inDegreeTable[consumer] == 1) {
            zeroIndegreeOps.push_back(consumer);
            _inDegreeTable.erase(consumer);
        } else {
            _inDegreeTable[consumer]--;
        }
    }
    return zeroIndegreeOps;
}

void ListScheduler::compute_ready_data_list() {
    // populate ready data ops
    for (auto entry : _inDegreeTable) {
        if (_inDegreeTable[entry.second] == 0 && is_data_op(entry.first)) {
            _readyDataOps.insert(entry.first);
            // reduce indegree of op consumers
            reduce_in_degree_of_adjacent_operations(entry.first);
        }
    }
}

void ListScheduler::compute_ready_compute_list() {
    // populate ready compute ops
    for (auto entry : _inDegreeTable) {
        if (_inDegreeTable[entry.second] == 0 && !is_data_op(entry.first)) {
            _readyComputeOps.insert(entry.first);
        }
    }
}

SmallVector<mlir::Value> ListScheduler::get_sorted_buffers(size_t opIdx) {
    auto op = _depsInfo.getExecuteOpAtIndex(opIdx);
    auto usedBuffs = _liveRangeInfo.getUsedBuffers(op);
    SmallVector<std::pair<vpux::AddressType, mlir::Value>> newBufs;
    for (auto val : usedBuffs) {
        const auto type = val.getType().cast<mlir::MemRefType>();
        if (type.getMemorySpace() != _memSpace || _scan.handler().isAlive(val)) {
            continue;
        }
        auto size = _scan.handler().getSize(val);
        std::cout << "need to allocate buffer of size " << size << std::endl;
        newBufs.push_back(std::make_pair(size, val));
    }
    llvm::sort(newBufs.begin(), newBufs.end(),
               [](const std::pair<vpux::AddressType, mlir::Value>& val1,
                  const std::pair<vpux::AddressType, mlir::Value>& val2) {
                   return val1.first > val2.first;
               });
    SmallVector<mlir::Value> orderedBufs;
    for (auto pair : newBufs) {
        orderedBufs.push_back(pair.second);
    }
    return orderedBufs;
}

SmallVector<size_t> ListScheduler::get_non_empty_op_demand_list(size_t opIdx) {
    SmallVector<size_t> demandList;
    for (auto dep : _depsInfo.getOpDeps(opIdx)) {
        if (_opOutputTable.find(dep) == _opOutputTable.end()) {
            demandList.push_back(dep);
        }
    }
    return demandList;
}

bool ListScheduler::is_ready_compute_operation_schedulable(size_t opIdx) {
    // are resources available and can be allocated
    auto sortedBuffers = get_sorted_buffers(opIdx);
    return _scan.canAlloc(sortedBuffers);
}

void ListScheduler::schedule_input_op_for_compute_op(size_t inputIdx) {
    std::cout << "schedule input idx: " << inputIdx << std::endl;
    auto _opOutput = _opOutputTable.find(inputIdx);
    op_type_e op_type;
    if (_opOutput != _opOutputTable.end()) {
        (_opOutput->second).change_state_to_active();
        op_type = op_type_e::IMPLICIT_OP_READ;
        Logger::global().error("ERROR SPILL OCCURED");
    } else {
        op_type = op_type_e::ORIGINAL_OP;
        _opOutputTable.insert(
                std::make_pair(inputIdx, op_output_info_t(operation_output_e::ACTIVE, _outDegreeTable[inputIdx])));
    }
    push_to_st_heap(heap_element_t(inputIdx, _currentTime, op_type));
}

void ListScheduler::allocate_sorted_buffers(SmallVector<mlir::Value> sortedBuffers) {
    SmallVector<mlir::Value> buffersNeedingAllocation;
    for (auto val : sortedBuffers) {
        // val.dump();
        const auto type = val.getType().cast<mlir::MemRefType>();
        if (type.getMemorySpace() != _memSpace || _scan.handler().isAlive(val)) {
            continue;
        }
        std::cout << "marked as alive" << std::endl;
        _scan.handler().markAsAlive(val);
        buffersNeedingAllocation.push_back(val);
    }

    VPUX_THROW_UNLESS(_scan.alloc(buffersNeedingAllocation, /*allowSpills*/ false), "Failed LP allocation");
}

void ListScheduler::schedule_compute_op(size_t opIdx) {
    std::cout << "trying to schedule: " << opIdx << std::endl;

    // Step 1: add to output result table
    _opOutputTable.insert(std::make_pair(opIdx, op_output_info_t(operation_output_e::ACTIVE, _outDegreeTable[opIdx])));

    // Step 2: assign resources simultaneously
    auto demandList = get_non_empty_op_demand_list(opIdx);
    size_t maxInputDelay = 0;
    for (auto demand : demandList) {
        schedule_input_op_for_compute_op(demand);
        maxInputDelay = 1;
    }
    auto sortedBuffers = get_sorted_buffers(opIdx);
    std::cout << "buffers: " << std::endl;
    allocate_sorted_buffers(sortedBuffers);

    // TODO: case for inplce ops

    // Step 3: schedule the compute op
    size_t opStartTime = _currentTime + maxInputDelay;
    push_to_st_heap(heap_element_t(opIdx, opStartTime, op_type_e::ORIGINAL_OP));
    std::cout << "scheduled: " << opIdx << std::endl;
}

void ListScheduler::schedule_all_possible_ready_ops_and_update(std::unordered_set<size_t>& readyList) {
    std::cout << "redy ops: ";
    SmallVector<size_t> scheduledOps;
    for (auto opIdx : readyList) {
        std::cout << opIdx << ", ";
        if (is_ready_compute_operation_schedulable(opIdx)) {
            schedule_compute_op(opIdx);
            scheduledOps.push_back(opIdx);
        }
    }
    std::cout << std::endl;
    for (auto scheduledOp : scheduledOps) {
        readyList.erase(scheduledOp);
    }
}

size_t ListScheduler::choose_active_operation_for_eviction() {
    // TODO: choose smallest alive buffer

    return 0;
}

void ListScheduler::evict_active_op(size_t opIdx) {
    auto opOutput = _opOutputTable.find(opIdx);
    assert(opOutput != _opOutputTable.end() && (opOutput->second).active());
    (opOutput->second).change_state_to_spilled();

    auto op = _depsInfo.getExecuteOpAtIndex(opIdx);
    auto usedBuffs = _liveRangeInfo.getUsedBuffers(op);
    for (auto usedBuff : usedBuffs) {
        _scan.handler().markAsDead(usedBuff);
    }
    _scan.freeNonAlive();
}

void ListScheduler::force_schedule_active_op_eviction() {
    return;
    size_t candidateIdx = choose_active_operation_for_eviction();
    evict_active_op(candidateIdx);

    // TODO: update _activeComputeOps some may no longer be active

    push_to_st_heap(heap_element_t(candidateIdx, _currentTime, op_type_e::IMPLICIT_OP_WRITE));
}

void ListScheduler::clear_lists() {
    _readyComputeOps.clear();   // ready operations with no active input
    _readyDataOps.clear();      // ready data inputs (->CMX)
    _activeComputeOps.clear();  // compute operations with at least one active input
}

bool ListScheduler::init() {
    _currentTime = 1;

    std::cout << "\n #### OPERATION DEPENDENCIES ####\n" << std::endl;
    _depsInfo.printTokenDependencies();

    // compute op in/out degree
    _inDegreeTable = _depsInfo.calculateOpInDegreeTable();
    _outDegreeTable = _depsInfo.calculateOpOutDegreeTable();

    std::cout << _inDegreeTable.size() << std::endl;
    std::cout << _outDegreeTable.size() << std::endl;

    clear_lists();
    // check_if_input_is_dag();

    compute_ready_data_list();
    compute_ready_compute_list();
    schedule_all_possible_ready_ops_and_update(_readyComputeOps);
    next_schedulable_op();

    return true;
}

void ListScheduler::next_schedulable_op() {
    size_t outputOp = _depsInfo.getOutputOp();
    // outputOp = 23;  // debug

    // scheduling loop
    for (;;) {
        std::cout << "\n#### IN NEXT SCHEDULABE OP ####\n" << std::endl;
        std::cout << "Picking the min among the start time and completion time heaps" << std::endl;

        // choose the minimum time
        const heap_element_t* start_top_ptr = top_element_gen(_startTimeHeap);
        const heap_element_t* completion_top_ptr = top_element_gen(_completionTimeHeap);

        bool pop_from_start_heap =
                start_top_ptr && (!completion_top_ptr || (start_top_ptr->time_ < completion_top_ptr->time_));

        // bool pop_from_start_heap = !_startTimeHeap.empty();
        if (pop_from_start_heap) {
            std::cout << "popping from start time heap" << std::endl;
            // schedule first op in heap
            heap_element_t firstOp = pop_from_st_heap();
            _currentTime = firstOp.time_;
            _timeBuckets[firstOp.time_].push_back(firstOp.op_);
            // TODO: get resource info
            _scheduledOps.push_back(scheduled_op_info_t(firstOp.op_, firstOp.op_type_, firstOp.time_));
            std::cout << "adding to completion time heap: " << firstOp.op_ << ", with time " << (firstOp.time_ + 1)
                      << std::endl;
            // move to completion time heap
            push_to_ct_heap(heap_element_t(firstOp.op_, firstOp.time_ + 1, firstOp.op_type_));
            // TODO: end condition
            if (firstOp.op_ == outputOp) {
                break;
            }
        } else {
            do {
                // unschedule ops completing at the next time - completion time heap
                std::cout << "generating new ready lists and updating" << std::endl;
                unschedule_all_completing_ops_at_next_earliest_time();

                std::cout << "schedule and update active ops" << std::endl;

                schedule_all_possible_ready_ops_and_update(_activeComputeOps);
                std::cout << "schedule and update ready ops" << std::endl;
                schedule_all_possible_ready_ops_and_update(_readyComputeOps);
            } while (!_completionTimeHeap.empty() && _startTimeHeap.empty());

            if (_startTimeHeap.empty()) {
                // unable to schedule an operation, perform spill
                std::cout << "unable to schedule an operation, perform spill" << std::endl;
                // std::cout << "total CMX free size: " << _scan.totalFreeSize() << std::endl;
                // std::cout << "max CMX free size: " << _scan.maxFreeSize() << std::endl;
                // std::cout << "number of live ranges: " << _scan.liveRanges().size() << std::endl;
                // std::cout << "number of gaps: " << _scan.gaps().size() << std::endl;
                force_schedule_active_op_eviction();
                return;
            }
        }
    }

    std::cout << "\n #### RESULTED TIME SCHEDULE ####\n" << std::endl;
    std::cout << "\tTime\tOperation" << std::endl;

    // for (auto op : _scheduledOps) {
    //     std::cout << "\t" << op.time_ << ":\t" << op.op_ << "\t" << op.op_type_name() << std::endl;
    // }

    for (auto entry : _timeBuckets) {
        std::cout << "\t" << entry.first << ":\t";
        for (auto ops : _timeBuckets[entry.first]) {
            std::cout << ops << ", ";
        }
        std::cout << std::endl;
    }
}

void ListScheduler::generateSchedule() {
    std::cout << "\n #### GENERATING SCHEDULE ####\n" << std::endl;

    init();

    std::cout << "\n #### END OF SCHEDULING ####\n" << std::endl;

    // return _scheduledOps;
}