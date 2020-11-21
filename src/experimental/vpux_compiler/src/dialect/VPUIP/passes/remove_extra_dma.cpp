//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "remove_extra_dma"

using namespace vpux;

namespace {

class RemoveExtraDMAPass final
        : public VPUIP::RemoveExtraDMABase<RemoveExtraDMAPass> {
public:
    RemoveExtraDMAPass();

public:
    void runOnFunction() final;

private:
    void passBody();

private:
    mlir::OpPassManager _cleanUpIR;
};

RemoveExtraDMAPass::RemoveExtraDMAPass()
        : _cleanUpIR(mlir::FuncOp::getOperationName(),
                     mlir::OpPassManager::Nesting::Implicit) {
    _cleanUpIR.addPass(mlir::createCanonicalizerPass());
}

void RemoveExtraDMAPass::runOnFunction() {
    try {
        passBody();
    } catch (const std::exception& e) {
        printTo(getOperation().emitError(),
                "RemoveExtraDMAPass failed : {0}",
                e.what());
        signalPassFailure();
    }
}

void RemoveExtraDMAPass::passBody() {
    auto func = getFunction();

    const auto callback = [](VPUIP::TaskOpInterface dmaTask) {
        using MemEffect =
                mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>;

        auto* dmaTaskOp = dmaTask.getOperation();

        if (dmaTask.getTaskType() != VPUIP::TaskType::UPADMA &&
            dmaTask.getTaskType() != VPUIP::TaskType::NNDMA) {
            return;
        }

        LLVM_DEBUG(printTo(llvm::dbgs(), "Got DMA task : {0} \n", *dmaTaskOp));

        auto dmaSrc = dmaTask.inputTensors().front();
        auto dmaDst = dmaTask.outputTensors().front();

        if (dmaSrc.getType() != dmaDst.getType()) {
            LLVM_DEBUG(
                    printTo(llvm::dbgs(),
                            "Source and destination types are different \n"));

            return;
        }

        mlir::OpOperand* srcProducer = nullptr;
        SmallVector<mlir::OpOperand*, 1> dstConsumers;

        for (auto& srcUse : dmaSrc.getUses()) {
            auto* srcUserOp = srcUse.getOwner();

            if (srcUserOp == dmaTaskOp) {
                continue;
            }

            if (!srcUserOp->isBeforeInBlock(dmaTaskOp)) {
                LLVM_DEBUG(printTo(
                        llvm::dbgs(),
                        "There are source users after DMA task : {0} \n",
                        *srcUserOp));

                return;
            }

            auto opEffects =
                    mlir::dyn_cast<mlir::MemoryEffectOpInterface>(srcUserOp);

            if (opEffects == nullptr) {
                LLVM_DEBUG(printTo(llvm::dbgs(),
                                   "Unknown source user : {0} \n",
                                   *srcUserOp));

                return;
            }

            SmallVector<MemEffect, 1> valEffects;
            opEffects.getEffectsOnValue(dmaSrc, valEffects);

            if (valEffects.size() != 1) {
                LLVM_DEBUG(printTo(llvm::dbgs(),
                                   "Source user {0} has unsupported memory "
                                   "effects count \n",
                                   *srcUserOp));

                return;
            }

            const auto& effect = valEffects.front();

            if (effect.getEffect() != mlir::MemoryEffects::Write::get()) {
                LLVM_DEBUG(printTo(
                        llvm::dbgs(),
                        "Source user {0} has unsupported memory effect \n",
                        *srcUserOp));

                return;
            }

            srcProducer = &srcUse;
        }

        for (auto& dstUse : dmaDst.getUses()) {
            auto* dstUserOp = dstUse.getOwner();

            if (dstUserOp == dmaTaskOp) {
                continue;
            }

            if (dstUserOp->isBeforeInBlock(dmaTaskOp)) {
                LLVM_DEBUG(printTo(
                        llvm::dbgs(),
                        "There are destination users before DMA task : {0} \n",
                        *dstUserOp));

                return;
            }

            auto opEffects =
                    mlir::dyn_cast<mlir::MemoryEffectOpInterface>(dstUserOp);

            if (opEffects == nullptr) {
                LLVM_DEBUG(printTo(llvm::dbgs(),
                                   "Unknown destination user : {0} \n",
                                   *dstUserOp));

                return;
            }

            SmallVector<MemEffect, 1> valEffects;
            opEffects.getEffectsOnValue(dmaDst, valEffects);

            if (valEffects.size() != 1) {
                LLVM_DEBUG(
                        printTo(llvm::dbgs(),
                                "Destination user {0} has unsupported memory "
                                "effects count \n",
                                *dstUserOp));

                return;
            }

            const auto& effect = valEffects.front();

            if (effect.getEffect() != mlir::MemoryEffects::Read::get()) {
                LLVM_DEBUG(printTo(
                        llvm::dbgs(),
                        "Destination user {0} has unsupported memory effect \n",
                        *dstUserOp));

                return;
            }

            dstConsumers.push_back(&dstUse);
        }

        if (srcProducer == nullptr && dstConsumers.empty()) {
            LLVM_DEBUG(printTo(llvm::dbgs(),
                               "Both source and destination values are block "
                               "arguments, can't remove DMA task {0} \n",
                               *dmaTaskOp));

            return;
        } else if (srcProducer != nullptr && dstConsumers.empty()) {
            LLVM_DEBUG(printTo(llvm::dbgs(),
                               "Destination value is block argument, redirect "
                               "source producer and remove DMA task {0} \n",
                               *dmaTaskOp));

            srcProducer->set(dmaDst);

            dmaTaskOp->erase();
        } else {
            LLVM_DEBUG(printTo(
                    llvm::dbgs(),
                    "Redirect destination users and remove DMA task {0} \n",
                    *dmaTaskOp));

            for (auto* dstUse : dstConsumers) {
                dstUse->set(dmaSrc);
            }

            dmaTaskOp->erase();
        }
    };

    func.walk(callback);

    if (mlir::failed(runPipeline(_cleanUpIR, func))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPUIP::createRemoveExtraDMAPass() {
    return std::make_unique<RemoveExtraDMAPass>();
}
