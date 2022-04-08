//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// WrapOpsInSparsifyDesparsifyPairsPass
//

class WrapOpsInSparsifyDesparsifyPairsPass final :
        public VPU::WrapOpsInSparsifyDesparsifyPairsBase<WrapOpsInSparsifyDesparsifyPairsPass> {
public:
    explicit WrapOpsInSparsifyDesparsifyPairsPass(VPU::SparsityProfileCreateFunc sparsityProfileCreateCb, Logger log)
            : _sparsityProfileCreateCb(std::move(sparsityProfileCreateCb)) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

private:
    VPU::ActivationSparsityProfile _sparsityProfile{VPU::ActivationSparsityProfile::S0};
    VPU::SparsityProfileCreateFunc _sparsityProfileCreateCb;
};

mlir::LogicalResult WrapOpsInSparsifyDesparsifyPairsPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    const auto parsedSparsityProfile = _sparsityProfileCreateCb(sparsityProfile.getValue());
    if (!parsedSparsityProfile.hasValue()) {
        return mlir::failure();
    }
    _sparsityProfile = parsedSparsityProfile.getValue();
    return mlir::success();
}

//
// safeRunOnFunc
//

void WrapOpsInSparsifyDesparsifyPairsPass::safeRunOnFunc() {
    using namespace VPU;
    using namespace VPU::NCESparsity;

    auto func = getFunction();

    const auto outputWrapper = [&](mlir::Operation* producerOp, mlir::Location loc) {
        auto result = producerOp->getResult(0);
        VPUX_THROW_WHEN(result.getType().isa<VPU::SparseTensorType>(),
                        "Operation at '{0}' already have sparse output, which is not expected by "
                        "WrapOpsInSparsifyDesparsifyPairs pass",
                        result.getLoc());

        mlir::OpBuilder builder(producerOp);
        builder.setInsertionPointAfter(producerOp);

        const auto sparsifyOp = builder.create<VPU::SparsifyOp>(loc, result);
        const auto desparsifyOp = builder.create<VPU::DesparsifyOp>(loc, sparsifyOp->getResult(0));
        result.replaceAllUsesExcept(desparsifyOp->getResult(0), sparsifyOp);
        _log.trace("Added Desparsify->Sparsify chain for '{0}' output", producerOp);
    };
    const auto inputWrapper = [&](mlir::Operation* consumerOp, unsigned int operandId, mlir::Location loc) {
        mlir::OpBuilder builder(consumerOp);

        const auto producer = consumerOp->getOperand(operandId);
        VPUX_THROW_WHEN(producer.getType().isa<VPU::SparseTensorType>(),
                        "Operation at '{0}' already have sparse input, which is not expected by "
                        "WrapOpsInSparsifyDesparsifyPairs pass",
                        producer.getLoc());

        const auto sparsifyOp = builder.create<VPU::SparsifyOp>(loc, producer);
        auto newResult = sparsifyOp->getResult(0);
        if (_sparsityProfile == ActivationSparsityProfile::S1) {
            const auto desparsifyOp = builder.create<VPU::DesparsifyOp>(loc, newResult);
            newResult = desparsifyOp->getResult(0);
            _log.trace("Added Desparsify->Sparsify chain for input #{0} of '{1}'", operandId, consumerOp);
        } else {
            _log.trace("Added Sparsify for input #{0} of '{1}'", operandId, consumerOp);
        }
        consumerOp->setOperand(operandId, newResult);
    };

    func->walk([&](VPU::SparseOpInterface sparsifiableOp) {
        const auto loc = sparsifiableOp->getLoc();
        if (supportsSparseOutputs(sparsifiableOp)) {
            outputWrapper(sparsifiableOp, loc);
        }
        if (supportsSparseInputs(sparsifiableOp)) {
            inputWrapper(sparsifiableOp, DEFAULT_SPARSIFIABLE_INPUT_OPERAND_ID, loc);
            if (mlir::isa<VPU::NCEEltwiseOp>(sparsifiableOp)) {
                // TODO: handle eltwise weight sparsity
                inputWrapper(sparsifiableOp, ELTWISE_SPARSIFIABLE_SECOND_INPUT_OPERAND_ID, loc);
            }
        }
    });
}

}  // namespace

//
// createWrapOpsInSparsifyDesparsifyPairsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createWrapOpsInSparsifyDesparsifyPairsPass(
        VPU::SparsityProfileCreateFunc sparsityProfileCreateCb, Logger log) {
    return std::make_unique<WrapOpsInSparsifyDesparsifyPairsPass>(std::move(sparsityProfileCreateCb), log);
}
