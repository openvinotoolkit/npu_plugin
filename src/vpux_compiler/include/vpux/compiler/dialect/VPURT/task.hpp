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

#pragma once

#include <mlir/IR/Builders.h>
#include "vpux/compiler/dialect/VPURT/ops.hpp"

namespace vpux {
namespace VPURT {

template <typename OpTy, typename... Args>
OpTy WrapIntoTaskOp(mlir::OpBuilder builder, mlir::ValueRange waitBarriers, mlir::ValueRange updateBarriers,
                    mlir::Location location, Args&&... args) {
    auto taskOp = builder.create<vpux::VPURT::TaskOp>(location, waitBarriers, updateBarriers);
    auto& block = taskOp.op().emplaceBlock();
    mlir::OpBuilder::InsertPoint lastInsertionPoint = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(&block);
    auto ret = builder.create<OpTy>(location, std::forward<Args>(args)...);
    builder.restoreInsertionPoint(lastInsertionPoint);
    return ret;
}

mlir::LogicalResult verifyTaskOp(TaskOp task);

}  // namespace VPURT
}  // namespace vpux
