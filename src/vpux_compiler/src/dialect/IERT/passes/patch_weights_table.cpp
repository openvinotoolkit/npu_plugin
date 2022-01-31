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

#include "vpux/compiler/dialect/IERT/passes.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/compiler/utils/logging.hpp"
#include "vpux/utils/core/range.hpp"

using namespace vpux;

namespace {

//
// PatchWeightsTablePass
//

class PatchWeightsTablePass final : public IERT::PatchWeightsTableBase<PatchWeightsTablePass> {
public:
    explicit PatchWeightsTablePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    uint64_t getPointer(mlir::Value value, uint64_t defaultValue);
    void relocateWeightsTable(Const::DeclareOp cst, VPUIP::NNDMAOp dmaOp, VPUIP::NCEClusterTaskOp nceOp);
};

//
// safeRunOnFunc
//

void PatchWeightsTablePass::safeRunOnFunc() {
    auto funcOp = getFunction();
    // For each nceOp.weight_table find related DeclareBufferOp. Next find dmaOp which
    // fills the buffer. DmaOp's input is expected to be Const::DeclareOp which
    // should be modified by adding relocateWeightTable transformation.
    funcOp.walk([this](vpux::VPUIP::NCEClusterTaskOp nceOp) {
        auto wTable = nceOp.weight_table();
        if (wTable == nullptr) {
            return;
        }
        auto wtDecBuf = wTable.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(wtDecBuf != nullptr, "DeclareBufferOp expected as a weight_table parent");
        VPUIP::NNDMAOp dmaOp;
        auto wtBuffResult = wtDecBuf.getResult();
        for (const auto& user : wtBuffResult.getUsers()) {
            dmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(user);
            if ((dmaOp != nullptr) && (dmaOp.output_buff() == wtBuffResult)) {
                break;
            }
        }
        VPUX_THROW_UNLESS(dmaOp != nullptr, "DmaOp expected, but not found for the weight table");
        const auto dmaInput = dmaOp.input();
        auto cst = dmaInput.getDefiningOp<Const::DeclareOp>();
        VPUX_THROW_UNLESS(cst != nullptr, "Constant expected as DMA input for weights table.");

        // On top of existing transformation a new transformation is added to the content attribute
        // of weight table const. The new transformation will patch offsets in this constant
        // with sparsity and weights pointers. The pointers are passed as  parameters of the
        // new transformation.
        relocateWeightsTable(cst, dmaOp, nceOp);
    });
}

void PatchWeightsTablePass::relocateWeightsTable(Const::DeclareOp cst, VPUIP::NNDMAOp dmaOp,
                                                 VPUIP::NCEClusterTaskOp nceOp) {
    // Retrieve sparsity and weight pointers which have correct values as they are already allocated
    // by the memory scheduler
    auto activationWindow = nceOp.activation_window();
    uint64_t sparsityBasePtr = getPointer(activationWindow, VPU::NCESparsity::SPARSITY_PTR_WHEN_NO_SPARISTY);
    auto weights = nceOp.weights();
    uint64_t weightBasePointer = getPointer(weights, 0);

    // Extract content attrib with existing transformations
    auto origConstAttr = cst.contentAttr();
    // Create new attribute based on existing one by adding new relocateWeightTable
    // transformation
    auto newConstAttr = origConstAttr.relocateWeightsTablePointers(weightBasePointer, sparsityBasePtr);
    mlir::OpBuilder builder(cst);

    // Create new DeclareOp with the new content attribute and replace the old DeclareOp
    // with it
    auto newConstOp = builder.create<Const::DeclareOp>(cst.getLoc(), cst.output().getType(), newConstAttr);
    dmaOp.setOperand(0, newConstOp.output());
    if (cst->getUses().empty()) {
        cst.erase();
    }
}

uint64_t PatchWeightsTablePass::getPointer(mlir::Value value, uint64_t defaultValue) {
    if (value == nullptr) {
        return defaultValue;
    }
    auto valueDeclareBuffer = mlir::dyn_cast<VPURT::DeclareBufferOp>(value.getDefiningOp());
    if (valueDeclareBuffer == nullptr) {
        return defaultValue;
    }
    return valueDeclareBuffer.byteOffset();
}

}  // namespace

//
// createPatchWeightsTablePass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createPatchWeightsTablePass(Logger log) {
    return std::make_unique<PatchWeightsTablePass>(log);
}
