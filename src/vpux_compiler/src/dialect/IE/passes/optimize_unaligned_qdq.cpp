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

    func.walk([this](vpux::IE::AffineReshapeOp firstAffineReshape) {
        if (!firstAffineReshape->hasOneUse()) {
            return;
        }
        const auto outType = firstAffineReshape.getType().dyn_cast<vpux::NDTypeInterface>();
        if ((outType.getShape()[Dims4D::Act::C] % 16) == 0) {
            return;
        }
        const auto inType = firstAffineReshape.input().getType().dyn_cast<vpux::NDTypeInterface>();
        if ((inType.getShape()[Dims4D::Act::C] % 16) != 0) {
            return;
        }
        auto fakeQuantize = mlir::dyn_cast<vpux::IE::FakeQuantizeOp>(*firstAffineReshape->getUsers().begin());
        if (fakeQuantize == nullptr) {
            return;
        }
        if (!fakeQuantize->hasOneUse()) {
            return;
        }
        mlir::OpBuilder fakeOpBuilder(firstAffineReshape);
        auto newFakeQuantize = fakeOpBuilder.create<IE::FakeQuantizeOp>(
                fakeQuantize->getLoc(), firstAffineReshape.input(), fakeQuantize.input_low(), fakeQuantize.input_high(),
                fakeQuantize.output_low(), fakeQuantize.output_high(), fakeQuantize.levelsAttr(),
                fakeQuantize.auto_broadcastAttr());
        firstAffineReshape.setOperand(newFakeQuantize);
        fakeQuantize.replaceAllUsesWith(firstAffineReshape.output());
#if 0
        llvm::errs() << "------------------------------------------\n";
        firstAffineReshape.getOperand().dump();
        firstAffineReshape->dump();
        fakeQuantize.dump();
        newFakeQuantize->dump();
        newFakeQuantize.getOperand(0).dump();
        firstAffineReshape->getUsers().begin()->dump();
        llvm::errs() << "------------------------------------------\n";
        auto lastAffineReshape = lastPermCast.getOperand().getDefiningOp<IE::AffineReshapeOp>();
        if (lastAffineReshape == nullptr) {
            return;
        }

        auto lastMemPerm = lastAffineReshape.getOperand().getDefiningOp<IE::MemPermuteOp>();
        if (lastMemPerm == nullptr) {
            return;
        }

        auto sliceOp = lastMemPerm.getOperand().getDefiningOp<IE::SliceOp>();
        if (sliceOp == nullptr) {
            return;
        }

        auto secondAndOp = sliceOp.getOperand().getDefiningOp<IE::AndOp>();
        if (secondAndOp == nullptr) {
            return;
        }

        auto firstAndOp = secondAndOp.getOperand(0).getDefiningOp<IE::AndOp>();
        if (firstAndOp == nullptr) {
            return;
        }
        for (const auto& user : firstAndOp->getUsers()) {
            if (mlir::dyn_cast<vpux::IE::AndOp>(user) != secondAndOp) {
                return;
            }
        }

        auto memPerm = firstAndOp.getOperand(0).getDefiningOp<IE::MemPermuteOp>();
        if (memPerm == nullptr) {
            return;
        }
        auto expand = memPerm->getOperand(0).getDefiningOp<IE::ExpandOp>();
        if (expand == nullptr) {
            return;
        }
        auto affineReshape = expand->getOperand(0).getDefiningOp<IE::AffineReshapeOp>();
        if (affineReshape == nullptr) {
            return;
        }
        auto firstMemPerm = affineReshape->getOperand(0).getDefiningOp<IE::MemPermuteOp>();
        if (firstMemPerm == nullptr) {
            return;
        }
        auto convOp = firstMemPerm->getOperand(0).getDefiningOp<IE::ConvolutionOp>();
        if (convOp == nullptr) {
            return;
        }

        auto lastConvOp = mlir::dyn_cast<vpux::IE::ConvolutionOp>(*lastPermCast->getUsers().begin());
        if (lastConvOp == nullptr) {
            return;
        }

        mlir::OpBuilder andOpBuilder(firstMemPerm);

        const auto quantType = firstAndOp.getType().dyn_cast<vpux::NDTypeInterface>();
        const auto tensorType = firstMemPerm.getType().cast<mlir::RankedTensorType>();
        const auto tensorNdType = firstMemPerm.getType().cast<vpux::NDTypeInterface>();
        const auto newType =
                vpux::getTensorType(tensorNdType.getShape(), quantType.getElementType(), quantType.getDimsOrder(),
                                    quantType.getMemSpace(), IE::isSparse(tensorType));

        auto newQuantOp =
                andOpBuilder.create<IE::AndOp>(firstAndOp->getLoc(), newType, convOp.output(), convOp.output(),
                                               firstAndOp.auto_broadcastAttr(), firstAndOp.post_opAttr());
        auto newDequantOp = andOpBuilder.create<IE::AndOp>(secondAndOp->getLoc(), convOp.getType(), newQuantOp.output(),
                                                           newQuantOp.output(), firstAndOp.auto_broadcastAttr(),
                                                           firstAndOp.post_opAttr());
        firstMemPerm.setOperand(newDequantOp);
        lastAffineReshape.setOperand(affineReshape.output());
#endif
    });
}

}  // namespace

//
// createOptimizeUnalignedQDQSeq
//

std::unique_ptr<mlir::Pass> vpux::IE::createOptimizeUnalignedQDQSeqPass(Logger log) {
    return std::make_unique<OptimizeUnalignedQDQSeqPass>(log);
}
