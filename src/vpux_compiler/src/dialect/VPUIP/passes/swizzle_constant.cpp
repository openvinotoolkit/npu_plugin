//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/swizzle_transform.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/range.hpp"

using namespace vpux;

namespace {

//
// SwizzleConstantPass
//

class SwizzleConstantPass final : public VPUIP::SwizzleConstantBase<SwizzleConstantPass> {
public:
    explicit SwizzleConstantPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    void attachSwizzleTransformation(int64_t swizzleKey, Const::DeclareOp cstOp, mlir::Operation* cstLoadOp,
                                     uint64_t arch);
    mlir::Operation* getLoadOpForDstBuffer(mlir::Value dstBuffer);
    void swizzleConstant(VPUIP::NCEClusterTaskOp nceOp, VPU::ArchKind archKind, mlir::Value constant);
};

//
// safeRunOnFunc
//

void SwizzleConstantPass::safeRunOnFunc() {
    auto funcOp = getFunction();
    auto module = funcOp->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    if (arch != VPU::ArchKind::VPUX37XX && arch != VPU::ArchKind::VPUX40XX) {
        _log.trace("Swizzling is supported starting from VPUX37XX");
        return;
    }

    funcOp.walk([&](VPUIP::NCEClusterTaskOp nceOp) {
        if (nceOp.task_type() == VPUIP::NCETaskType::ELTWISE) {
            return;
        }
        swizzleConstant(nceOp, arch, nceOp.weights());
        swizzleConstant(nceOp, arch, nceOp.weight_table());
    });
}

void SwizzleConstantPass::swizzleConstant(VPUIP::NCEClusterTaskOp nceOp, VPU::ArchKind archKind, mlir::Value constant) {
    if (constant == nullptr) {
        return;
    }

    if (mlir::isa_and_nonnull<VPUIP::NCEClusterTilingOp>(nceOp->getParentOp())) {
        constant = VPUIP::getTopBufferOfNCEClusterTiling(nceOp, constant);
    }

    auto constantDecBuf = constant.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(constantDecBuf != nullptr, "DeclareBufferOp expected as a constant's parent");
    auto swizzleKey = constantDecBuf.swizzlingKey().getValueOr(0);

    if (!swizzleKey) {
        _log.nest().trace("Swizzle Key is 0, buffer not set to swizzle  '{0}'", nceOp->getLoc());
        return;
    }

    // Get operation that loads constant to CMX for NCE Task
    auto* cstLoadOp = getLoadOpForDstBuffer(constantDecBuf.getResult());
    VPUX_THROW_UNLESS(cstLoadOp != nullptr, "Operation loading constant expected, but not located");

    // Get the constant definition op whose content will be swizzled
    auto inputBuffer = cstLoadOp->getOperand(0);
    auto cstOp = inputBuffer.getDefiningOp<Const::DeclareOp>();

    // In case constant was spilled there can be a sequence of DMAs
    // Need to resolve it and update this DMA to have const as input directly
    while (cstOp == nullptr) {
        cstLoadOp = getLoadOpForDstBuffer(inputBuffer);
        VPUX_THROW_UNLESS(cstLoadOp != nullptr, "Next DMA op as source operation expected for weights");

        inputBuffer = cstLoadOp->getOperand(0);
        cstOp = inputBuffer.getDefiningOp<Const::DeclareOp>();
    }

    _log.nest().trace("Operation loading weight table '{0}' '{1}'", cstLoadOp->getName(), cstLoadOp->getLoc());

    VPUX_THROW_UNLESS(cstOp != nullptr, "Constant expected as DMA input for constant - {0}", *cstLoadOp);

    // On top of existing transformation a new transformation is added to the content attribute
    // of weight table const. The new transformation will swizzle the constant with swizzle key parameter
    _log.nest().trace("Constant for swizzling '{0}'", cstOp->getLoc());
    attachSwizzleTransformation(swizzleKey, cstOp, cstLoadOp, static_cast<uint64_t>(archKind));
}

// Find a DMA operation that loads data into a given buffer
mlir::Operation* SwizzleConstantPass::getLoadOpForDstBuffer(mlir::Value dstBuffer) {
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

void SwizzleConstantPass::attachSwizzleTransformation(int64_t swizzleKey, Const::DeclareOp cstOp,
                                                      mlir::Operation* cstLoadOp, uint64_t arch) {
    // Extract content attrib with existing transformations
    auto constAttr = cstOp.contentAttr();

    // Create new attribute based on existing one by adding new swizzleConstant transformation
    auto newConstAttr = constAttr.swizzleConstant(swizzleKey, arch);
    mlir::OpBuilder builder(cstOp);

    auto newConstOp = builder.create<Const::DeclareOp>(cstOp.getLoc(), cstOp.output().getType(), newConstAttr);
    cstLoadOp->setOperand(0, newConstOp.output());
    if (cstOp->getUses().empty()) {
        cstOp.erase();
    }
}

}  // namespace

//
// createSwizzleConstantPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createSwizzleConstantPass(Logger log) {
    return std::make_unique<SwizzleConstantPass>(log);
}
