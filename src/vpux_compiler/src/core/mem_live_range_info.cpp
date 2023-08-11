//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/mem_live_range_info.hpp"

#include "vpux/compiler/utils/analysis.hpp"

#include "vpux/utils/core/error.hpp"

#include <algorithm>

using namespace vpux;

vpux::MemLiveRangeInfo::MemLiveRangeInfo(mlir::func::FuncOp funcOp, mlir::AnalysisManager& am)
        : _log(Logger::global().nest("mem-live-range-info", 0)),
          _aliasInfo(am.getAnalysis<AliasesInfo, mlir::func::FuncOp>()) {
    _log.trace("Collect all buffer allocations");
    _log = _log.nest();

    funcOp->walk([&](mlir::Operation* op) {
        if (!isBufAllocOp(op)) {
            return;
        }

        _log.trace("Got buffer value allocated by '{0}' at '{1}'", op->getName(), op->getLoc());
        _log = _log.nest();

        for (auto res : op->getResults()) {
            addNewBuffer(res);
        }

        _log = _log.unnest();
    });

    _log = _log.unnest();
}

void vpux::MemLiveRangeInfo::addNewBuffer(mlir::Value val) {
    _log.trace("Collect all direct and indirect users");
    _log = _log.nest();

    auto* valRegion = val.getParentRegion();
    const auto& aliases = _aliasInfo.getAllAliases(val);
    auto& allUsers = _allUsersInBlock[val];

    for (auto alias : aliases) {
        _log.trace("Process alias '{0}'", alias);
        _log = _log.nest();

        if (alias.getParentBlock() == val.getParentBlock()) {
            _log.trace("The alias belongs to the same block, traverse its users");
            _log = _log.nest();

            for (auto* user : alias.getUsers()) {
                _log.trace("Got alias user '{0}' at '{1}'", user->getName(), user->getLoc());

                auto* userAncestor = valRegion->findAncestorOpInRegion(*user);
                VPUX_THROW_UNLESS(userAncestor != nullptr,
                                  "Alias user '{0}' doesn't belong to the same block or its sub-region as '{1}'",
                                  user->getLoc(), val);

                _log.trace("It has an ancestor '{0}' at '{1}' in root value parent region", userAncestor->getName(),
                           userAncestor->getLoc());

                allUsers.insert(userAncestor);
                _reverseUsers[userAncestor].insert(val);
            }

            _log = _log.unnest();
        } else {
            _log.trace("The alias belongs to the sub-region of root block");

            auto* aliasParentOp = alias.getParentRegion()->getParentOp();
            VPUX_THROW_UNLESS(aliasParentOp != nullptr, "Alias '{0}' has no parent Operation", alias);

            _log.trace("It belongs to the operation '{0}' at '{1}'", aliasParentOp->getName(), aliasParentOp->getLoc());

            auto* parentAncestor = val.getParentRegion()->findAncestorOpInRegion(*aliasParentOp);
            VPUX_THROW_UNLESS(parentAncestor != nullptr,
                              "Alias '{0}' doesn't belong to the same block or its sub-region as '{1}'", alias, val);

            _log.trace("It has an ancestor '{0}' at '{1}' in root value parent region", parentAncestor->getName(),
                       parentAncestor->getLoc());

            allUsers.insert(parentAncestor);
            _reverseUsers[parentAncestor].insert(val);
        }

        _log = _log.unnest();
    }

    _log = _log.unnest();
}

ValueOrderedSet vpux::MemLiveRangeInfo::getUsedBuffers(mlir::Operation* op) const {
    const auto it = _reverseUsers.find(op);
    if (it != _reverseUsers.end()) {
        return it->second;
    }

    return {};
}

bool vpux::MemLiveRangeInfo::isBufferUsedByOp(mlir::Value val, mlir::Operation* op) const {
    const auto valIt = _allUsersInBlock.find(val);
    VPUX_THROW_UNLESS(valIt != _allUsersInBlock.end(), "Value '{0}' is not a buffer", val);
    auto& allUsers = valIt->second;

    return allUsers.find(op) != allUsers.end();
}

size_t vpux::MemLiveRangeInfo::eraseUser(mlir::Value val, mlir::Operation* op) {
    const auto valIt = _allUsersInBlock.find(val);
    VPUX_THROW_UNLESS(valIt != _allUsersInBlock.end(), "Value '{0}' is not a buffer", val);
    auto& allUsers = valIt->second;

    const auto opIt = _reverseUsers.find(op);
    VPUX_THROW_UNLESS(opIt != _reverseUsers.end(), "Operation '{0}' at '{1}' is not a buffer user", op->getName(),
                      op->getLoc());
    auto& allBufs = opIt->second;

    VPUX_THROW_UNLESS(allUsers.erase(op), "Operation '{0}' at '{1}' is not a buffer '{2}' user", op->getName(),
                      op->getLoc(), val);
    allBufs.erase(val);

    return allUsers.size();
}
