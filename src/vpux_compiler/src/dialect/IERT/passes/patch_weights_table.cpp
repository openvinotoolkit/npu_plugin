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
    void patchWeightsTable(Const::DeclareOp cstOp, mlir::Operation* cstLoadOp, VPUIP::NCEClusterTaskOp nceOp);
    mlir::Value getTopBufferOfNCEClusterTiling(mlir::Operation* innerOp, mlir::Value buffer);
    mlir::Operation* getLoadOpForDstBuffer(mlir::Value dstBuffer);
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
        auto wTable = getTopBufferOfNCEClusterTiling(nceOp, nceOp.weight_table());
        if (wTable == nullptr) {
            return;
        }
        _log.trace("WeightTable identified for operation '{0}'", nceOp->getLoc());
        auto wtDecBuf = wTable.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(wtDecBuf != nullptr, "DeclareBufferOp expected as a weight_table parent");

        mlir::Value inputBuffer;

        // Get operation that loads weights table to CMX for NCE Task
        auto* cstLoadOp = getLoadOpForDstBuffer(wtDecBuf.getResult());
        VPUX_THROW_UNLESS(cstLoadOp != nullptr, "Operation loading weight table expected, but not located");

        // Get the constant definition op whose content will be patched
        inputBuffer = cstLoadOp->getOperand(0);
        auto cstOp = inputBuffer.getDefiningOp<Const::DeclareOp>();

        // In case weights table was spilled there can be a sequence of DMAs
        // Need to resolve it and update this DMA to have const as input directly
        while (cstOp == nullptr) {
            cstLoadOp = getLoadOpForDstBuffer(inputBuffer);
            VPUX_THROW_UNLESS(cstLoadOp != nullptr, "Next DMA op as source operation expected");

            inputBuffer = cstLoadOp->getOperand(0);
            cstOp = inputBuffer.getDefiningOp<Const::DeclareOp>();
        }

        _log.nest().trace("Operation loading weight table '{0}' '{1}'", cstLoadOp->getName(), cstLoadOp->getLoc());

        VPUX_THROW_UNLESS(cstOp != nullptr, "Constant expected as DMA input for weights table - {0}", *cstLoadOp);

        // On top of existing transformation a new transformation is added to the content attribute
        // of weight table const. The new transformation will patch offsets in this constant
        // with sparsity and weights pointers. The pointers are passed as  parameters of the
        // new transformation.
        _log.nest().trace("Constant for patching '{0}'", cstOp->getLoc());
        patchWeightsTable(cstOp, cstLoadOp, nceOp);
    });
}

// In case operation is wrapped in NCEClusterTiling this method will return mlir::Value at parent level
// corresponding to mlir::Value used by wrapped operation
// In case operation is not wrapped in NCEClusterTiling then just return same mlir::Value
mlir::Value PatchWeightsTablePass::getTopBufferOfNCEClusterTiling(mlir::Operation* innerOp, mlir::Value buffer) {
    if (buffer == nullptr) {
        return buffer;
    }

    if (auto nceClustOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(innerOp->getParentOp())) {
        auto* bodyBlock = &nceClustOp.body().front();
        const auto blockArg = buffer.dyn_cast<mlir::BlockArgument>();
        VPUX_THROW_WHEN(blockArg == nullptr || blockArg.getOwner() != bodyBlock,
                        "Matching argument was not identified");

        return nceClustOp->getOperand(blockArg.getArgNumber());
    }
    return buffer;
}

// Find a DMA operation that loads data into a given buffer
mlir::Operation* PatchWeightsTablePass::getLoadOpForDstBuffer(mlir::Value dstBuffer) {
    for (const auto& user : dstBuffer.getUsers()) {
        auto dmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(user);
        if ((dmaOp != nullptr) && (dmaOp.output_buff() == dstBuffer)) {
            return dmaOp.getOperation();
        }

        auto nceClustOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(user);
        if (nceClustOp != nullptr && nceClustOp.getOutputs()[0] == dstBuffer &&
            mlir::isa<VPUIP::NNDMAOp>(nceClustOp.getInnerTaskOp())) {
            return nceClustOp.getOperation();
        }
    }
    return nullptr;
}

void PatchWeightsTablePass::patchWeightsTable(Const::DeclareOp cstOp, mlir::Operation* cstLoadOp,
                                              VPUIP::NCEClusterTaskOp nceOp) {
    // Retrieve sparsity and weight pointers which have correct values as they are already allocated
    // by the memory scheduler
    auto activationWindow = nceOp.activation_window();
    uint64_t sparsityBasePtr = getPointer(activationWindow, VPU::NCESparsity::SPARSITY_PTR_WHEN_NO_SPARSITY);
    auto weights = getTopBufferOfNCEClusterTiling(nceOp, nceOp.weights());
    uint64_t weightBasePointer = getPointer(weights, 0);

    // Extract content attrib with existing transformations
    auto origConstAttr = cstOp.contentAttr();
    // Create new attribute based on existing one by adding new relocateWeightTable
    // transformation
    auto newConstAttr = origConstAttr.relocateWeightsTablePointers(weightBasePointer, sparsityBasePtr);
    mlir::OpBuilder builder(cstOp);

    // Create new DeclareOp with the new content attribute and replace the old DeclareOp
    // with it
    auto newConstOp = builder.create<Const::DeclareOp>(cstOp.getLoc(), cstOp.output().getType(), newConstAttr);
    cstLoadOp->setOperand(0, newConstOp.output());
    if (cstOp->getUses().empty()) {
        cstOp.erase();
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
