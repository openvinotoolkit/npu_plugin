//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/async_deps_info.hpp"

#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/range.hpp"

using namespace vpux;

//
// Constructor
//

vpux::AsyncDepsInfo::AsyncDepsInfo(mlir::func::FuncOp func)
        : _log(Logger::global().nest("async-deps-info", 0)),
          _indexAttrName(mlir::StringAttr::get(func->getContext(), "async-deps-index")) {
    buildDepsMap(func);
}

//
// setIndex/getIndex
//

void vpux::AsyncDepsInfo::setIndex(mlir::async::ExecuteOp execOp, uint64_t index) {
    execOp->setAttr(_indexAttrName, getIntAttr(execOp.getContext(), index));
}

uint32_t vpux::AsyncDepsInfo::getIndex(mlir::async::ExecuteOp execOp) const {
    const auto attr = execOp->getAttrOfType<mlir::IntegerAttr>(_indexAttrName);
    VPUX_THROW_UNLESS(attr != nullptr, "Attribute '{0}' was not set for '{1}' operation at '{2}'", _indexAttrName,
                      execOp->getName(), execOp->getLoc());

    return checked_cast<uint32_t>(attr.getValue().getZExtValue());
}

mlir::async::ExecuteOp vpux::AsyncDepsInfo::getExecuteOpAtIndex(size_t opIdx) const {
    VPUX_THROW_WHEN(opIdx >= _execOpCount, "Invalid index '{0}' for _allExecOps", opIdx);
    return _allExecOps[opIdx];
}

//
// buildDepsMap
//

void vpux::AsyncDepsInfo::buildDepsMap(mlir::func::FuncOp func) {
    _log.trace("Collect initial dependencies maps");
    _log = _log.nest();

    _allExecOps = to_small_vector(func.getOps<mlir::async::ExecuteOp>());
    for (const auto& p : _allExecOps | indexed) {
        setIndex(p.value(), p.index());
    }

    _execOpCount = _allExecOps.size();
    _depsMap.resize(_execOpCount);
    for (auto& deps : _depsMap) {
        deps.resize(checked_cast<uint32_t>(_allExecOps.size()));
    }

    for (auto& op : func.getOps()) {
        if (auto execOp = mlir::dyn_cast<mlir::async::ExecuteOp>(op)) {
            addExecOp(execOp);
        } else if (auto waitOp = mlir::dyn_cast<mlir::async::AwaitOp>(op)) {
            _log.trace("Found 'async.await' Operation at '{0}'", op.getLoc());

            if (waitOp.getResult() != nullptr) {
                for (auto* user : waitOp.getResult().getUsers()) {
                    VPUX_THROW_WHEN(
                            user->getParentOfType<mlir::async::ExecuteOp>() != nullptr,
                            "Got 'async.await' Operation at '{0}', which has users inside 'async.execute' region",
                            op.getLoc());
                }
            }
        }
    }

    _log = _log.unnest();
}

void vpux::AsyncDepsInfo::addExecOp(mlir::async::ExecuteOp execOp) {
    _log.trace("Found 'async.execute' Operation at '{0}'", execOp->getLoc());
    _log = _log.nest();

    const auto execInd = getIndex(execOp);

    for (auto arg : execOp->getOperands()) {
        auto argExecOp = mlir::dyn_cast<mlir::async::ExecuteOp>(arg.getDefiningOp());
        VPUX_THROW_UNLESS(argExecOp != nullptr,
                          "'async.execute' at '{0}' has operand '{1}' produced by unsupported Operation",
                          execOp->getLoc(), arg);

        _log.trace("It has a dependency from other 'async.execute' Operation at '{0}'", argExecOp->getLoc());

        const auto argExecInd = getIndex(argExecOp);
        _depsMap[execInd].set(argExecInd);
    }

    _log = _log.unnest();
}

//
// addDependency
//

void vpux::AsyncDepsInfo::addDependency(mlir::async::ExecuteOp from, mlir::async::ExecuteOp to) {
    const auto fromInd = getIndex(from);
    const auto toInd = getIndex(to);
    _depsMap[toInd].set(fromInd);
    if (!_consumerMap.empty()) {
        // also update consumer map if build
        _consumerMap[fromInd].set(toInd);
    }
}

//
// optimizeDepsMap
//

void vpux::AsyncDepsInfo::optimizeDepsMap() {
    //
    // A -> B -> C
    //
    // If B depends on A and C depends on [A, B] ==> we can remove A from C deps list,
    // since it will be implicit dependency taken from B.
    //
    // Algorithm is divided into two steps:
    //  step 1 - transitive closure
    //  step 2 - transitive reduction
    // Worst case complexity is O(N^3) but expected time will be proportional to ~N*E^2
    // so in case of sparse graphs which is a usual case for NN models expected time shouldn't
    // be as bad as N^3

    // Step 1: Transitive closure
    // Update all dependencies in a new depsMapClosure to represent transitive closure
    // of initial dependencies graph. For each node starting from beginning it will go
    // though its dependencies, take their dependensiec and update its own based on that
    // In this step even if initial graph was sparse, after it, it will be dense
    // In depsStartAndEnd for each node (represented by async deps index) store information
    // about first and last dependency index. This is used to reduce computation time of step 2
    // in case original graph had nodes which in most cases depended only on fairly close neighbours
    auto depsMapClosure = _depsMap;
    SmallVector<std::pair<int, int>> depsStartAndEnd;
    for (auto& curDeps : depsMapClosure) {
        depsStartAndEnd.push_back(std::make_pair(curDeps.find_first(), curDeps.find_last()));
        for (auto curDepInd : curDeps.set_bits()) {
            const auto& depOfDeps = depsMapClosure[curDepInd];
            curDeps |= depOfDeps;
        }
    }

    // Step 2: Transitive reduction
    // Remove all unnecessary edges.
    // Go through each node starting from the end and remove depndencies if those dependencies
    // are already represented in its dependand nodes
    // To leverage the fact that in most cases each node will only have limited range of dependencies
    // this step is optimized to only scan range from original deps (depsStartAndEnd) and not all
    // from transitive closure step. This optimization reduces expected computation time, although worst case
    // complexity can be the same as without it.
    for (int depInd = (static_cast<int>(_depsMap.size()) - 1); depInd >= 0; depInd--) {
        auto& curDeps = _depsMap[depInd];

        auto startIdx = depsStartAndEnd[depInd].first;
        auto endIdx = depsStartAndEnd[depInd].second;

        // If node does not have any dependency (negative idx) or it has
        // only one dependency then skip
        if (startIdx < 0 || endIdx < 0 || startIdx == endIdx) {
            continue;
        }

        for (int curDepInd = startIdx; curDepInd <= endIdx; ++curDepInd) {
            if (!curDeps[curDepInd]) {
                continue;
            }
            const auto& depOfDeps = depsMapClosure[curDepInd];
            curDeps.reset(depOfDeps);
        }
    }

    if (!_consumerMap.empty()) {
        // re-build consumer map using new deps map if build
        _consumerMap.clear();
        buildConsMap();
    }
}

//
// buildConsMap
//

void vpux::AsyncDepsInfo::buildConsMap() {
    _consumerMap.resize(_depsMap.size());

    for (auto& cons : _consumerMap) {
        cons.resize(checked_cast<uint32_t>(_depsMap.size()));
    }

    for (size_t idx = 0; idx < _depsMap.size(); idx++) {
        for (auto bit : _depsMap[idx].set_bits()) {
            _consumerMap[checked_cast<size_t>(bit)].set(checked_cast<uint32_t>(idx));
        }
    }
}

//
// updateTokenDependencies
//

void vpux::AsyncDepsInfo::updateTokenDependencies() {
    _log.trace("Add explicit '!async.token' based dependencies between 'async.execute' operations");
    _log = _log.nest();

    for (auto* execOpIt = _allExecOps.begin(); execOpIt != _allExecOps.begin() + _execOpCount; ++execOpIt) {
        _log.trace("Process 'async.execute' Operation at '{0}'", execOpIt->getLoc());

        const auto execInd = getIndex(*execOpIt);
        const auto& execDeps = _depsMap[execInd];

        SmallVector<mlir::Value> depsVec;
        for (auto depInd : execDeps.set_bits()) {
            depsVec.push_back(_allExecOps[depInd].getToken());
        }

        _log.nest().trace("Use the following explicit dependencies : {0}", depsVec);
        execOpIt->getDependenciesMutable().assign(ArrayRef(depsVec));
    }

    _log = _log.unnest();
}

size_t vpux::AsyncDepsInfo::insertNewExecOpToDepsMap(mlir::async::ExecuteOp execOp) {
    auto dataStructSize = _allExecOps.size();
    VPUX_THROW_WHEN(_execOpCount > dataStructSize, "Invalid execOp count '{0}'", _execOpCount);

    if (_execOpCount == dataStructSize) {
        preAllocateForNewOps(1);
    }

    _allExecOps[_execOpCount] = execOp;
    setIndex(execOp, _execOpCount);
    addExecOp(execOp);

    return _execOpCount++;
}

/* Adds more space to internal structures, it only resizes all internal structures in advance
   to avoid loss on single operation insertion.
   Use only if you know in advance how many insetions are nesessary. */
void vpux::AsyncDepsInfo::preAllocateForNewOps(size_t numOfNewOps) {
    auto newSize = _allExecOps.size() + numOfNewOps;
    _allExecOps.resize(newSize);
    _depsMap.resize(newSize);
    _consumerMap.resize(newSize);

    for (auto& deps : _depsMap) {
        deps.resize(checked_cast<uint32_t>(newSize));
    }
    for (auto& cons : _consumerMap) {
        cons.resize(checked_cast<uint32_t>(newSize));
    }
}

const llvm::BitVector& vpux::AsyncDepsInfo::getOpDeps(size_t opIdx) const {
    VPUX_THROW_WHEN(opIdx >= _execOpCount, "Invalid index '{0}' for _depsMap", opIdx);
    return _depsMap[opIdx];
}

const llvm::BitVector& vpux::AsyncDepsInfo::getConsumerOps(size_t opIdx) const {
    VPUX_THROW_WHEN(_consumerMap.empty(), "Consumer map was not build");
    VPUX_THROW_WHEN(opIdx >= _execOpCount, "Invalid index '{0}' for _consumerMap", opIdx);
    return _consumerMap[opIdx];
}

std::unordered_map<size_t, size_t> vpux::AsyncDepsInfo::calculateOpInDegreeTable() const {
    std::unordered_map<size_t, size_t> opInDegree;
    for (size_t i = 0; i < _execOpCount; ++i) {
        opInDegree[i] = static_cast<size_t>(_depsMap[i].count());
    }
    return opInDegree;
}

std::unordered_map<size_t, size_t> vpux::AsyncDepsInfo::calculateOpOutDegreeTable() const {
    VPUX_THROW_WHEN(_consumerMap.empty(), "Consumer map was not build");
    std::unordered_map<size_t, size_t> opOutDegree;
    for (size_t i = 0; i < _execOpCount; ++i) {
        opOutDegree[i] = static_cast<size_t>(_consumerMap[i].count());
    }
    return opOutDegree;
}
