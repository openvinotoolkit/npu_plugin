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

#include "vpux/compiler/core/async_deps_info.hpp"

#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/range.hpp"

using namespace vpux;

//
// Constructor
//

vpux::AsyncDepsInfo::AsyncDepsInfo(mlir::FuncOp func)
        : _log(Logger::global().nest("async-deps-info", 0)),
          _indexAttrName(mlir::Identifier::get("async-deps-index", func->getContext())) {
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

mlir::async::ExecuteOp vpux::AsyncDepsInfo::getExecuteOpAtIndex(uint32_t index) const {
    for (auto execOp : _allExecOps) {
        uint32_t currIndex = getIndex(execOp);
        if (index == currIndex) {
            return execOp;
        }
    }
    return nullptr;
}

SmallVector<size_t> vpux::AsyncDepsInfo::getOpDeps(size_t index) const {
    SmallVector<size_t> opDeps = {};
    for (auto dep : _depsMap[index].set_bits()) {
        opDeps.push_back(dep);
    }
    return opDeps;
}

//
// buildDepsMap
//

void vpux::AsyncDepsInfo::buildDepsMap(mlir::FuncOp func) {
    _log.trace("Collect initial dependencies maps");
    _log = _log.nest();

    _allExecOps = to_small_vector(func.getOps<mlir::async::ExecuteOp>());
    for (const auto& p : _allExecOps | indexed) {
        setIndex(p.value(), p.index());
    }

    _depsMap.resize(_allExecOps.size());
    for (auto& deps : _depsMap) {
        deps.resize(checked_cast<uint32_t>(_allExecOps.size()));
    }

    for (auto& op : func.getOps()) {
        if (auto execOp = mlir::dyn_cast<mlir::async::ExecuteOp>(op)) {
            addExecOp(execOp);
        } else if (auto waitOp = mlir::dyn_cast<mlir::async::AwaitOp>(op)) {
            _log.trace("Found 'async.await' Operation at '{0}'", op.getLoc());

            if (waitOp.result() != nullptr) {
                for (auto* user : waitOp.result().getUsers()) {
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

    for (auto& curDeps : _depsMap) {
        for (auto curDepInd : curDeps.set_bits()) {
            const auto& depOfDeps = _depsMap[curDepInd];
            curDeps |= depOfDeps;
        }
    }

    for (auto& curDeps : _depsMap | reversed) {
        for (auto curDepInd : curDeps.set_bits()) {
            const auto& depOfDeps = _depsMap[curDepInd];
            curDeps.reset(depOfDeps);
        }
    }
}

//
// updateTokenDependencies
//

void vpux::AsyncDepsInfo::updateTokenDependencies() {
    _log.trace("Add explicit '!async.token' based dependencies between 'async.execute' operations");
    _log = _log.nest();

    for (auto execOp : _allExecOps) {
        _log.trace("Process 'async.execute' Operation at '{0}'", execOp->getLoc());

        const auto execInd = getIndex(execOp);
        const auto& execDeps = _depsMap[execInd];

        SmallVector<mlir::Value> depsVec;
        for (auto depInd : execDeps.set_bits()) {
            depsVec.push_back(_allExecOps[depInd].token());
        }

        _log.nest().trace("Use the following explicit dependencies : {0}", depsVec);
        execOp.dependenciesMutable().assign(makeArrayRef(depsVec));
    }

    _log = _log.unnest();
}

void vpux::AsyncDepsInfo::printTokenDependencies() const {
    for (size_t i = 0; i < _depsMap.size(); ++i) {
        auto execOp = getExecuteOpAtIndex(i);
        Logger::global().error("op: id:{0} {1} {2}", i, execOp->getLoc(), execOp->getName());
        for (auto dep : _depsMap[i].set_bits()) {
            auto execDepOp = getExecuteOpAtIndex(dep);
            // execDepOp->dump();
            Logger::global().error("\tdep: id:{0} {1} {2}", dep, execDepOp->getLoc(), execDepOp->getName());
        }
    }
}

SmallVector<size_t> vpux::AsyncDepsInfo::getConsumerOps(size_t readyOp) const {
    SmallVector<size_t> consumerOps;
    for (size_t i = 0; i < _depsMap.size(); ++i) {
        for (auto consumer : _depsMap[i].set_bits()) {
            size_t castConsumer = checked_cast<size_t>(consumer);
            if (castConsumer == readyOp) {
                consumerOps.push_back(i);
            }
        }
    }
    return consumerOps;
}

std::unordered_map<size_t, size_t> vpux::AsyncDepsInfo::calculateOpInDegreeTable() const {
    std::unordered_map<size_t, size_t> opInDegree;
    for (size_t i = 0; i < _depsMap.size(); ++i) {
        opInDegree[i] = _depsMap[i].count();
    }
    return opInDegree;
}

std::unordered_map<size_t, size_t> vpux::AsyncDepsInfo::calculateOpOutDegreeTable() const {
    std::unordered_map<size_t, size_t> opOutDegree;
    for (size_t i = 0; i < _depsMap.size(); ++i) {
        for (auto consumer : _depsMap[i].set_bits()) {
            size_t castConsumer = checked_cast<size_t>(consumer);
            if (opOutDegree.find(castConsumer) != opOutDegree.end()) {
                opOutDegree[castConsumer]++;
            } else {
                opOutDegree[castConsumer] = 1;
            }
        }
    }
    return opOutDegree;
}

size_t vpux::AsyncDepsInfo::getOutputOp() const {
    return _depsMap.size() - 2;
}
