//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/utils/core/logger.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>

namespace vpux {

void addLogging(mlir::MLIRContext& ctx, Logger log);
void addLogging(mlir::PassManager& pm, Logger log);

class OpBuilderLogger final : public mlir::OpBuilder::Listener {
public:
    explicit OpBuilderLogger(Logger log): _log(log) {
    }

public:
    void notifyOperationInserted(mlir::Operation* op) final;
    void notifyBlockCreated(mlir::Block* block) final;

private:
    Logger _log;
};

}  // namespace vpux
