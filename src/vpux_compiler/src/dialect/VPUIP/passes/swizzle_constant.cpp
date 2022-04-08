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
};

//
// safeRunOnFunc
//

void SwizzleConstantPass::safeRunOnFunc() {
    auto funcOp = getFunction();
    auto module = funcOp->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    funcOp.walk([&](VPUIP::NNDMAOp nnDMA) {
        auto inputBufferOp = nnDMA.input().getDefiningOp<Const::DeclareOp>();
        auto outputBufferOp = nnDMA.output_buff().getDefiningOp<VPURT::DeclareBufferOp>();

        if (inputBufferOp == nullptr || outputBufferOp == nullptr) {
            return;
        }

        for (auto transAttr : inputBufferOp.contentAttr().getTransformations()) {
            // Check if swizzling transformation is already attached
            auto swizzleConstAttr = transAttr.dyn_cast_or_null<vpux::Const::SwizzleConstantAttr>();
            if (swizzleConstAttr != nullptr) {
                return;
            }
        }

        auto swizzleKey = outputBufferOp.swizzlingKey().getValueOr(0);
        if (!swizzleKey) {
            return;
        }

        _log.nest().trace("Found operation that copy constant to swizzled buffer '{0}'", nnDMA);
        _log.nest().trace("Swizzle key: '{0}'", swizzleKey);

        attachSwizzleTransformation(swizzleKey, inputBufferOp, nnDMA, static_cast<uint64_t>(arch));
    });
}

void SwizzleConstantPass::attachSwizzleTransformation(int64_t swizzleKey, Const::DeclareOp cstOp,
                                                      mlir::Operation* cstLoadOp, uint64_t arch) {
    // Extract content attrib with existing transformations
    auto constAttr = cstOp.contentAttr();

    // Create new attribute based on existing one by adding new swizzleConstant transformation
    auto newConstAttr = constAttr.swizzleConstant(swizzleKey, arch);
    mlir::OpBuilder builder(cstOp);

    auto newConstOp = builder.create<Const::DeclareOp>(cstOp.getLoc(), cstOp.output().getType(), newConstAttr);
    _log.nest().trace("Create new content with swizzle transformation", cstOp);

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
