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

#include "vpux/compiler/core/prefetch_edge_generator.hpp"

#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/range.hpp"

using namespace vpux;

//
// Constructor
//

vpux::PrefetchEdgeGenerator::PrefetchEdgeGenerator(scheduledOps& initialSchedule, AsyncDepsInfo& depsInfo)
        : _log(Logger::global().nest("chain-pipelining", 0)), _scheduledOps(initialSchedule), _depsInfo(depsInfo) {
    // TODO: consider storing ops in a struct with
    // opIdx, time, out-degree, size, hasActiveResource, isData
    // sort by time -> out-degree -> size -> opIdx
}

bool vpux::PrefetchEdgeGenerator::prefetchConstraintsSatisifed(ScheduledOpInfo* dataOp, ScheduledOpInfo* computeOp) {
    // constraints for prefetching limiting the prefetch so that operations are not prefetched
    // too early, this includes levels of compute operations where level 0 is the previous compute
    // operation scheduled, level 1 is the compute before level 0, etc. as well as arbitrarty time
    // constraint where operations can only be prefetched a certain time before

    // if a compute op, increase levels
    if (!dataOp->isDataOp()) {
        ++CURRENT_COMPUTE_OP_LEVEL;
    }

    // level difference constraint
    if (CURRENT_COMPUTE_OP_LEVEL > PREFETCH_LEVEL_LIMIT) {
        return false;
    }

    // time difference constraint
    if (dataOp->time_ - computeOp->time_ > PREFETCH_TIME_LIMIT) {
        return false;
    }

    // NOTE: in future constraints taking into account number of cycles

    return true;
}

bool vpux::PrefetchEdgeGenerator::prefetchOpsHaveTheSameConsumer(operationIdxType computeIdx,
                                                                 operationIdxType dataIdx) {
    // prefetch only data operations corresponding to one compute operation

    // no prefetch consumers yet
    if (_prefetchEdges[computeIdx].empty()) {
        return true;
    }

    // look for a shared consumer
    auto currentDataOpConsumers = _depsInfo.getConsumerOps(dataIdx);
    for (auto prefetchEdge : _prefetchEdges[computeIdx]) {
        auto prefetchDataOpConsumers = _depsInfo.getConsumerOps(prefetchEdge);
        for (auto consumer : prefetchDataOpConsumers) {
            for (auto newConsumers : currentDataOpConsumers) {
                if (consumer == newConsumers) {
                    return true;
                }
            }
        }
    }

    // no shared consumer found
    return false;
}

bool vpux::PrefetchEdgeGenerator::allDataOpDependenciesExecuted(operationIdxType dataIdx) {
    // condition will be true in cases of constants such as weight, weight table
    // however for tiled activations the DMAs will have a dependency, prevent
    // the activation from being prefetched and reducing the availible free NNCMX
    // size as it will not be scheduled at that time but some other data op might

    // check if all dependencies of the operations were executed
    for (auto opDepIdx : _depsInfo.getOpDeps(dataIdx)) {
        // if a dependency was not executed this op can not be prefetched at this time
        if (_executedOps.find(opDepIdx) == _executedOps.end()) {
            return false;
        }
    }

    return true;
}

bool vpux::PrefetchEdgeGenerator::canDataOpBePrefetched(ScheduledOpInfo* dataOp) {
    // check if the data op can be prefetched - satisfies all the below conditions

    // if not data op
    if (!dataOp->isDataOp()) {
        return false;
    }

    // if op has no active resources
    if (!dataOp->hasActiveResource()) {
        return false;
    }

    // if op already prefetched
    if (_prefetchedDataOps.find(dataOp->op_) != _prefetchedDataOps.end()) {
        return false;
    }

    // if operation has some unscheduled dependencies
    if (!allDataOpDependenciesExecuted(dataOp->op_)) {
        return false;
    }

    // if all conditions satisfied, the op can be prefetched
    return true;
}

vpux::PrefetchEdgeGenerator::prefetchMap vpux::PrefetchEdgeGenerator::generatePrefetchEdges() {
    _log.trace("Creating pipelining chains");

    auto computeOp = _scheduledOps.begin();
    // skip input op, mark input as executed
    _executedOps.insert(computeOp->op_);
    ++computeOp;
    auto dataOp = computeOp;

    while (computeOp != _scheduledOps.end()) {
        // find compute op
        if (!computeOp->isDataOp() && computeOp->hasActiveResource()) {
            // find first possible data op to overlap with the compute
            CURRENT_COMPUTE_OP_LEVEL = 1;
            // NOTE: data op must be after compute
            dataOp = computeOp;
            // advance to next op
            ++dataOp;
            // store max free size
            vpux::AddressType maxFreeSize = computeOp->freeCmx_;

            // iterate over the following ops
            while (dataOp != _scheduledOps.end()) {
                // 1. verify prefetching constraints met, if not move to next compute
                if (!prefetchConstraintsSatisifed(dataOp, computeOp)) {
                    break;
                }

                // 2. all constraints met, try to find a prefetch-able data op
                if (dataOp->isOriginalOp() && computeOp->isOriginalOp() && canDataOpBePrefetched(dataOp) &&
                    prefetchOpsHaveTheSameConsumer(computeOp->op_, dataOp->op_)) {
                    auto dataOpSize = dataOp->resourceSize();
                    if (dataOpSize < maxFreeSize && computeOp->time_ < dataOp->time_) {
                        // ensure the data operation will fit through all ops scheduled intermediatly
                        _log.trace("data op = '{0}' will fit during compute = '{1}' with time dif = '{2}' and level "
                                   "dif '{3}'",
                                   dataOp->op_, computeOp->op_, (dataOp->time_ - computeOp->time_),
                                   CURRENT_COMPUTE_OP_LEVEL);
                        // store the prefetch edge
                        _prefetchEdges[computeOp->op_].insert(dataOp->op_);
                        _prefetchedDataOps.insert(dataOp->op_);
                        // reduce max free size with this data op size
                        maxFreeSize = maxFreeSize - dataOpSize;
                        dataOp->freeCmx_ = maxFreeSize;

                        // update free size for all ops to the prefetch op
                        auto temp = computeOp;
                        while (temp != dataOp && temp->time_ < dataOp->time_) {
                            VPUX_THROW_UNLESS(
                                    temp->freeCmx_ >= dataOpSize,
                                    "Prefetched operation ('{0}', size - '{1}') size does not fit at operation '{2}' "
                                    "where free CMX - '{3}', "
                                    "prefetched op size - '{1}'",
                                    dataOp->op_, dataOpSize, temp->op_, temp->freeCmx_);
                            temp->freeCmx_ -= dataOpSize;
                            ++temp;
                        }
                    }
                }

                // 3. update variables - choose the min from new free sizes
                maxFreeSize = std::min(maxFreeSize, dataOp->freeCmx_);
                // advance to data next op
                ++dataOp;
            }
        }
        // mark the operation as executed to store dependencies
        _executedOps.insert(computeOp->op_);
        // advance to next compute op
        ++computeOp;
    }

    _log.trace("prefetch edge count: '{0}'", _prefetchEdges.size());

    return _prefetchEdges;
}
