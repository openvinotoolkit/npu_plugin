//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/utils/stl_extras.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/AnalysisManager.h>

namespace vpux {

class MemLiveRangeInfo final {
    using UsersMap = ValueOrderedMap<OpOrderedSet>;
    using ReverseUsersMap = OpOrderedMap<ValueOrderedSet>;

public:
    MemLiveRangeInfo(mlir::FuncOp funcOp, mlir::AnalysisManager& am);

public:
    ValueOrderedSet getUsedBuffers(mlir::Operation* op) const;
    size_t eraseUser(mlir::Value val, mlir::Operation* op);
    bool isBufferUsedByOp(mlir::Value val, mlir::Operation* op) const;

private:
    void addNewBuffer(mlir::Value val);

private:
    Logger _log;
    const AliasesInfo& _aliasInfo;
    UsersMap _allUsersInBlock;
    ReverseUsersMap _reverseUsers;
};

}  // namespace vpux
