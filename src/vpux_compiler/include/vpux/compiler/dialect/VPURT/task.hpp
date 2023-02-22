//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/dialect/VPURT/ops.hpp"

#include <mlir/IR/Builders.h>

namespace vpux {
namespace VPURT {

template <typename OpTy, typename... Args>
OpTy wrapIntoTaskOp(mlir::OpBuilder& builder, mlir::ValueRange waitBarriers, mlir::ValueRange updateBarriers,
                    mlir::Location loc, Args&&... args) {
    auto taskOp = builder.create<vpux::VPURT::TaskOp>(loc, waitBarriers, updateBarriers);
    auto& block = taskOp.body().emplaceBlock();

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&block);

    return builder.create<OpTy>(loc, std::forward<Args>(args)...);
}

mlir::LogicalResult verifyTaskOp(TaskOp task);

template <typename OpTy, typename... Args>
OpTy createOp(mlir::PatternRewriter& rewriter, mlir::Operation* insertionPoint, Args&&... args) {
    VPUX_THROW_WHEN(insertionPoint == nullptr, "Insertion point is empty");
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(insertionPoint);
    return rewriter.create<OpTy>(std::forward<Args>(args)...);
}

struct TaskQueueType {
    VPU::ExecutorKind type;
    int64_t index;
    bool operator<(const TaskQueueType& other) const {
        if (type == other.type) {
            return index < other.index;
        }
        return type < other.type;
    }
    bool operator==(const TaskQueueType& other) const {
        return type == other.type && index == other.index;
    }
    bool operator!=(const TaskQueueType& other) const {
        return !(*this == other);
    }
};

SmallVector<int64_t> getDMATaskPorts(TaskOp task);

Optional<SmallVector<TaskQueueType>> getDMATaskQueueType(TaskOp task);

TaskQueueType getTaskQueueType(TaskOp task, bool ignoreIndexForNce = true);
}  // namespace VPURT
}  // namespace vpux
