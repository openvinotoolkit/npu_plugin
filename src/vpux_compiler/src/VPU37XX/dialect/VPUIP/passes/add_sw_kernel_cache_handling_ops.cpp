//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU37XX/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/sw_utils.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

constexpr StringLiteral vpuTaskTypeAttrName{"VPU.task_type"};
constexpr StringLiteral cacheFuncName{"cache_flush_invalidate"};

namespace {

mlir::SmallVector<mlir::Value> getDDRBuffers(mlir::ValueRange buffers) {
    mlir::SmallVector<mlir::Value> ddrBuffers;
    for (auto&& buffer : buffers) {
        auto bufferType = buffer.getType().cast<vpux::NDTypeInterface>();
        if (bufferType.getMemoryKind() == VPU::MemoryKind::DDR) {
            ddrBuffers.push_back(buffer);
        }
    }

    return ddrBuffers;
}

//
// AddSwKernelCacheHandlingOpsPass
//

class AddSwKernelCacheHandlingOpsPass final :
        public VPUIP::arch37xx::AddSwKernelCacheHandlingOpsBase<AddSwKernelCacheHandlingOpsPass> {
public:
    explicit AddSwKernelCacheHandlingOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AddSwKernelCacheHandlingOpsPass::safeRunOnFunc() {
    auto func = getOperation();

    func.walk([&](VPUIP::SwKernelOp origOp) {
        mlir::OpBuilder builder(origOp);
        OpBuilderLogger builderLog(_log);

        auto loc = origOp.getLoc();
        auto ctx = builder.getContext();

        auto inputBuffs = origOp.getInputs();
        auto outputBuffs = origOp.getOutputBuffs();

        // at least one input/output buffer must be in DDR
        auto ddrInputBuffs = getDDRBuffers(inputBuffs);
        auto ddrOutputBuffs = getDDRBuffers(outputBuffs);
        if (ddrInputBuffs.empty() && ddrOutputBuffs.empty()) {
            return;
        }

        if (isCacheHandlingOp(origOp)) {
            return;
        }

        auto origTaskOp = origOp->getParentOfType<VPURT::TaskOp>();
        auto origTaskOpUpdateBarriers = origTaskOp.getUpdateBarriers();
        mlir::SmallVector<mlir::Value> origTaskOpUpdateBarriersVector;
        for (const auto& barrier : origTaskOpUpdateBarriers) {
            origTaskOpUpdateBarriersVector.push_back(barrier);
        }

        builder.setInsertionPoint(origTaskOp);

        auto newUpdateBarrier = builder.create<VPURT::DeclareVirtualBarrierOp>(loc).getBarrier();
        origTaskOp.getUpdateBarriersMutable().clear();
        origTaskOp.getUpdateBarriersMutable().append(newUpdateBarrier);

        builder.setInsertionPointAfter(origTaskOp);

        // create builtin function containing CACHE_FLUSH_INVALIDATE
        auto origOpModule = origOp->getParentOfType<mlir::ModuleOp>();
        auto vpuswModule = vpux::VPUIP::getVPUSWModule(origOpModule, _log);
        auto functionName = mlir::StringRef(cacheFuncName);
        auto functionNameSymbol = mlir::SymbolRefAttr::get(ctx, functionName);
        auto functionSymbol = mlir::SymbolRefAttr::get(ctx, vpuswModule.getName().value(), {functionNameSymbol});

        // check if this function was already created
        auto innerModuleBuilder = mlir::OpBuilder::atBlockBegin(vpuswModule.getBody(), &builderLog);
        auto prebuiltFunction = vpuswModule.lookupSymbol<mlir::func::FuncOp>(functionName);
        if (prebuiltFunction == nullptr) {
            const auto funcType = mlir::FunctionType::get(ctx, {}, {});
            auto newFuncOp =
                    innerModuleBuilder.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(ctx), functionName, funcType);

            // modify attributes
            newFuncOp.setSymVisibilityAttr(mlir::StringAttr::get(ctx, "private"));
            VPU::ActShaveTaskType cacheOpType = VPU::ActShaveTaskType::CACHE_FLUSH_INVALIDATE;
            if (!ddrInputBuffs.empty() && ddrOutputBuffs.empty()) {
                // only need to invalidate cache partition when input is on DDR but output is on CMX
                // CACHE_FLUSH_INVALIDATE would cause DEVICE_LOST, see E#100623
                cacheOpType = VPU::ActShaveTaskType::CACHE_INVALIDATE;
            }
            newFuncOp->setAttr(vpuTaskTypeAttrName,
                               mlir::SymbolRefAttr::get(ctx, VPU::stringifyActShaveTaskType(cacheOpType)));
        }

        // create cache-handling op
        const int64_t tileIndex = 0;
        mlir::SmallVector<mlir::Value> buffers = {};
        const auto buffersRange = mlir::ValueRange(buffers);

        const auto newLoc = appendLoc(loc, "_cache_handling_op");
        auto cacheHandlingSwKernel = vpux::VPURT::wrapIntoTaskOp<VPUIP::SwKernelOp>(
                builder, mlir::ValueRange(newUpdateBarrier), mlir::ValueRange(origTaskOpUpdateBarriersVector), newLoc,
                buffersRange, buffersRange, nullptr, functionSymbol, getIntAttr(builder, tileIndex));

        const mlir::SmallVector<mlir::Attribute> args = {};
        vpux::VPUIP::initSwKernel(cacheHandlingSwKernel, buffersRange, buffersRange, args, _log.nest());
    });
}

}  // namespace

//
// createAddSwKernelCacheHandlingOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::arch37xx::createAddSwKernelCacheHandlingOpsPass(Logger log) {
    return std::make_unique<AddSwKernelCacheHandlingOpsPass>(log);
}
