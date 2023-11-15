//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes/insert_identity_pool_before_op.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/BlockAndValueMapping.h>

using namespace vpux;

bool vpux::IE::isEligiblePostOp(mlir::Operation* op, Logger log) {
    const auto postOpInterface = op->getOperand(0).getDefiningOp<IE::LayerWithPostOpInterface>();
    if (postOpInterface != nullptr && postOpInterface->getResult(0).hasOneUse()) {
        log.trace("A PostOp at {0} has already got a suitable producer", op->getLoc());
        return false;
    }
    return true;
}

mlir::LogicalResult vpux::IE::genericIdInserter(mlir::Operation* concreteOp, const InsertIdFunctor& insertId,
                                                mlir::PatternRewriter& rewriter, Logger log) {
    mlir::Operation* identityOp = insertId(concreteOp, rewriter, log);
    if (identityOp == nullptr) {
        return mlir::failure();
    }

    mlir::BlockAndValueMapping mapper;
    const SmallVector<mlir::Value> inputsToMap = {identityOp->getResult(0)};
    mapper.map(concreteOp->getOperands(), makeArrayRef(inputsToMap));
    auto* newLayerOp = rewriter.clone(*concreteOp, mapper);
    rewriter.replaceOp(concreteOp, newLayerOp->getResult(0));

    return mlir::success();
}
