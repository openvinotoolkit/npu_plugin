//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include <mlir/Pass/PassManager.h>

using namespace vpux;

namespace {

//
// AdjustInputDataForExplicitSETablePass
//

class AdjustInputDataForExplicitSETablePass final :
        public VPUIP::AdjustInputDataForExplicitSETableBase<AdjustInputDataForExplicitSETablePass> {
public:
    explicit AdjustInputDataForExplicitSETablePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AdjustInputDataForExplicitSETablePass::safeRunOnFunc() {
    auto func = getOperation();

    func.walk([&](VPUIP::NCEClusterTaskOp nceOp) {
        if (nceOp.input_storage_element_table() == nullptr) {
            return;
        }

        _log.trace("Got '{0}' at '{1}'", nceOp->getName(), nceOp->getLoc());
        VPUX_THROW_UNLESS(nceOp.input_se_size().has_value(), "Missing input storage element size");

        auto getNewType = [&](VPURT::DeclareBufferOp declareOp, mlir::Value seOperand) {
            auto type = seOperand.getType().cast<vpux::NDTypeInterface>();
            auto newShape = Shape(type.getShape().raw());
            newShape[Dims4D::Act::C] *= nceOp.input_se_size().value();
            auto newType = declareOp.getType().cast<vpux::NDTypeInterface>().changeShape(newShape);
            return newType;
        };

        auto adaptTypeFor = [&](mlir::Value operand, mlir::Value seOperand) {
            auto declareOp = operand.getDefiningOp<VPURT::DeclareBufferOp>();
            VPUX_THROW_UNLESS(declareOp != nullptr, "Expected buffer declaration, got {0}", operand.getDefiningOp());
            mlir::OpBuilder builder(declareOp);

            auto newDeclareOp = builder.clone(*declareOp.getOperation());
            auto newType = getNewType(declareOp, seOperand);
            newDeclareOp->getResult(0).setType(newType);

            declareOp.getBuffer().replaceUsesWithIf(newDeclareOp->getResult(0), [&](mlir::OpOperand& operand) -> bool {
                return operand.getOwner() == nceOp.getOperation();
            });
            if (declareOp.getBuffer().use_empty()) {
                declareOp->erase();
            }
        };

        adaptTypeFor(nceOp.input(), nceOp.input_storage_element_table());
        adaptTypeFor(nceOp.parent_input(), nceOp.parent_input_storage_element_table());
    });
}

}  // namespace

//
// createAdjustInputDataForExplicitSETablePass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createAdjustInputDataForExplicitSETablePass(Logger log) {
    return std::make_unique<AdjustInputDataForExplicitSETablePass>(log);
}
