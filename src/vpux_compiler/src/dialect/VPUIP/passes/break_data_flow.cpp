//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/passes.hpp"

using namespace vpux;

namespace {

//
// BreakDataFlowPass
//

class BreakDataFlowPass final : public VPUIP::BreakDataFlowBase<BreakDataFlowPass> {
public:
    explicit BreakDataFlowPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void BreakDataFlowPass::safeRunOnFunc() {
    auto funcOp = getOperation();

    funcOp.walk([](VPUIP::LayerOpInterface op) {
        for (const auto res : op->getOpResults()) {
            const auto ind = res.getResultNumber();
            const auto resBuf = op.getOutputs()[ind];
            res.replaceAllUsesWith(resBuf);
        }
    });
}

}  // namespace

//
// createBreakDataFlowPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createBreakDataFlowPass(Logger log) {
    return std::make_unique<BreakDataFlowPass>(log);
}
