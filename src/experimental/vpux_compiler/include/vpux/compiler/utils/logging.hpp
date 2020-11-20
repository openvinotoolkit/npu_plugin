//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

#include "vpux/utils/core/logger.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>

namespace vpux {

void addLogging(mlir::MLIRContext& ctx, LogLevel level);
void addLogging(mlir::PassManager& pm, LogLevel level);

class OpBuilderLogger final : public mlir::OpBuilder::Listener {
public:
    explicit OpBuilderLogger(LogLevel level): _log(level) {
    }

public:
    void notifyOperationInserted(mlir::Operation* op) final;
    void notifyBlockCreated(mlir::Block* block) final;

private:
    Logger _log;
};

}  // namespace vpux
