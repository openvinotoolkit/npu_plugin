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

#include <vpux/compiler/dialect/VPUIP/nce_invariant.hpp>
#include "vpux/compiler/dialect/IE/passes.hpp"

using namespace vpux;

namespace {

//
// RemoveQuantDequantSeqPass
//

class RemoveQuantDequantSeqPass final : public IE::RemoveQuantDequantSeqBase<RemoveQuantDequantSeqPass> {
public:
    explicit RemoveQuantDequantSeqPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }
    bool isDPU(mlir::Operation* op) {
        auto convOp = mlir::dyn_cast<IE::ConvolutionOp>(op);
        if (convOp != nullptr) {
            if (!VPUIP::NCEInvariant::verifyKernel(convOp, _log).failed()) {
                return true;
            }
        }
        auto grConvOp = mlir::dyn_cast<IE::GroupConvolutionOp>(op);
        if (grConvOp != nullptr) {
            if (!VPUIP::NCEInvariant::verifyKernel(grConvOp, _log).failed()) {
                return true;
            }
        }
        auto maxPoolOp = mlir::dyn_cast<IE::MaxPoolOp>(op);
        if (maxPoolOp != nullptr) {
            if (!VPUIP::NCEInvariant::verifyKernel(maxPoolOp, _log).failed()) {
                return true;
            }
        }
        auto andOp = mlir::dyn_cast<IE::AndOp>(op);
        if (andOp != nullptr) {
            if (!VPUIP::NCEInvariant::verifyKernel(andOp, _log).failed()) {
                return true;
            }
        }
        auto subtractOp = mlir::dyn_cast<IE::SubtractOp>(op);
        if (subtractOp != nullptr) {
            if (!VPUIP::NCEInvariant::verifyKernel(subtractOp, _log).failed()) {
                return true;
            }
        }
        auto multiplyOp = mlir::dyn_cast<IE::MultiplyOp>(op);
        if (multiplyOp != nullptr) {
            if (!VPUIP::NCEInvariant::verifyKernel(multiplyOp, _log).failed()) {
                return true;
            }
        }
        auto addOp = mlir::dyn_cast<IE::AddOp>(op);
        if (addOp != nullptr) {
            if (!VPUIP::NCEInvariant::verifyKernel(addOp, _log).failed()) {
                return true;
            }
        }

        return false;
    }

private:
    void safeRunOnFunc() final;
};

void RemoveQuantDequantSeqPass::safeRunOnFunc() {
    auto func = getFunction();
    // Remove remaining Quantize->Dequantize sequence to not perform explicit FakeQuantize.
    // This might have slight impact on accuracy but gives visible performance improvement
    // TODO: Evaluate possibility of replacing such sequence with ClampOp fused with DPU task
    func.walk([this](vpux::IE::QuantizeOp quantizeOp) {
        if (!quantizeOp->hasOneUse()) {
            return;
        }
        auto dequantizeOp = mlir::dyn_cast<vpux::IE::DequantizeOp>(*quantizeOp->getUsers().begin());
        if (dequantizeOp == nullptr) {
            return;
        }
        if (!dequantizeOp.output().hasOneUse()) {
            return;
        }
        if (isDPU(quantizeOp.input().getDefiningOp())) {
            return;
        }
        if (isDPU(*dequantizeOp.output().getUsers().begin())) {
            return;
        }
        // quantizeOp.input().getDefiningOp()->dump();
        // dequantizeOp.output().getUsers().begin()->dump();
        dequantizeOp.replaceAllUsesWith(quantizeOp.input());
    });
}  // namespace

}  // namespace

//
// createRemoveQuantDequantSeqPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createRemoveQuantDequantSeqPass(Logger log) {
    return std::make_unique<RemoveQuantDequantSeqPass>(log);
}
