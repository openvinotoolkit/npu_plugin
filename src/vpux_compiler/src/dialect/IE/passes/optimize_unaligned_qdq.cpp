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

#include "vpux/compiler/core/type_interfaces.hpp"
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
};

void OptimizeUnalignedQDQSeqPass::safeRunOnFunc() {
    auto func = getFunction();

    func.walk([this](vpux::IE::AffineReshapeOp affineReshape) {
        if (!affineReshape->hasOneUse()) {
            return;
        }
        const auto outType = affineReshape.getType().dyn_cast<vpux::NDTypeInterface>();
        if ((outType.getShape()[Dims4D::Act::C] % 16) == 0) {
            return;
        }
        const auto inType = affineReshape.input().getType().dyn_cast<vpux::NDTypeInterface>();
        if ((inType.getShape()[Dims4D::Act::C] % 16) != 0) {
            return;
        }
        auto fakeQuantize = mlir::dyn_cast<vpux::IE::FakeQuantizeOp>(*affineReshape->getUsers().begin());
        if (fakeQuantize == nullptr) {
            return;
        }
        if (!fakeQuantize->hasOneUse()) {
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
