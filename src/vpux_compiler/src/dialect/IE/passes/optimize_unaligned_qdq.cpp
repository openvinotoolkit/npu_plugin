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

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
using namespace vpux;

namespace {

class OptimizeUnalignedQDQSeqPass final : public IE::OptimizeUnalignedQDQSeqBase<OptimizeUnalignedQDQSeqPass> {
public:
    explicit OptimizeUnalignedQDQSeqPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

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
};

void OptimizeUnalignedQDQSeqPass::safeRunOnFunc() {
    auto func = getFunction();

    func.walk([this](vpux::IE::AffineReshapeOp affineReshape) {
        if (!affineReshape->hasOneUse()) {
            return;
        }
        auto fakeQuantize = mlir::dyn_cast<vpux::IE::FakeQuantizeOp>(*affineReshape->getUsers().begin());
        if (fakeQuantize == nullptr) {
            return;
        }
        if (!fakeQuantize->hasOneUse()) {
            return;
        }
        const auto outType = affineReshape.getType().dyn_cast<vpux::NDTypeInterface>();
        if (outType.getRank() != 4) {
            return;
        }
        if ((outType.getShape()[Dims4D::Act::C] % 16) == 0) {
            return;
        }
        if (!isDPU(affineReshape.input().getDefiningOp())) {
            return;
        }

        mlir::OpBuilder fakeOpBuilder(affineReshape);
        auto newFakeQuantize = fakeOpBuilder.create<IE::FakeQuantizeOp>(
                fakeQuantize->getLoc(), affineReshape.input(), fakeQuantize.input_low(), fakeQuantize.input_high(),
                fakeQuantize.output_low(), fakeQuantize.output_high(), fakeQuantize.levelsAttr(),
                fakeQuantize.auto_broadcastAttr());
        affineReshape.setOperand(newFakeQuantize);
        fakeQuantize.replaceAllUsesWith(affineReshape.output());
    });
}

}  // namespace

//
// createOptimizeUnalignedQDQSeq
//

std::unique_ptr<mlir::Pass> vpux::IE::createOptimizeUnalignedQDQSeqPass(Logger log) {
    return std::make_unique<OptimizeUnalignedQDQSeqPass>(log);
}
