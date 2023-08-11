//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/passes.hpp"

#include "vpux/compiler/core/ops_interfaces.hpp"

#include "vpux/utils/core/range.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>

using namespace vpux;

namespace {

//
// MoveDeclarationsToTopPass
//

class MoveDeclarationsToTopPass final : public MoveDeclarationsToTopBase<MoveDeclarationsToTopPass> {
public:
    explicit MoveDeclarationsToTopPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void MoveDeclarationsToTopPass::safeRunOnFunc() {
    auto& block = getOperation().getBody().front();

    SmallVector<mlir::Operation*> allDeclOps;
    for (auto& op : block) {
        if (op.hasTrait<DeclarationOp>() || mlir::isa<mlir::memref::AllocOp>(&op)) {
            allDeclOps.push_back(&op);
        }
    }

    if (allDeclOps.empty()) {
        return;
    }

    auto* firstDeclOp = allDeclOps.front();
    firstDeclOp->moveBefore(&block, block.begin());

    for (auto i : irange(allDeclOps.size() - 1)) {
        allDeclOps[i + 1]->moveAfter(allDeclOps[i]);
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::createMoveDeclarationsToTopPass(Logger log) {
    return std::make_unique<MoveDeclarationsToTopPass>(log);
}
