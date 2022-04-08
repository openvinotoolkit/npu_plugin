//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

using namespace vpux;

namespace {

//
// ConvertEltwiseToInPlace
//

class ConvertEltwiseToInPlace final : public mlir::OpRewritePattern<VPUIP::NCEClusterTaskOp> {
public:
    ConvertEltwiseToInPlace(mlir::MLIRContext* ctx, AliasesInfo& aliasesInfo, Logger log)
            : mlir::OpRewritePattern<VPUIP::NCEClusterTaskOp>(ctx), _aliasesInfo(aliasesInfo), _log(log) {
    }

private:
    AliasesInfo& _aliasesInfo;
    Logger _log;
};

void makeInPlaceEltwise(VPUIP::NCEClusterTaskOp clusterTaskOp, const AliasesInfo& aliasesInfo, Logger log) {
    if (clusterTaskOp.task_type() != VPUIP::NCETaskType::ELTWISE) {
        return;
    }

    if (!clusterTaskOp.is_inplace().getValueOr(false)) {
        return;
    }

    log.trace("Found inplace eltwise at {0}", clusterTaskOp.getLoc());

    auto outputBuff = VPUIP::getTopBufferOfNCEClusterTiling(clusterTaskOp, clusterTaskOp.output_buff());

    auto outputRootBuff = *aliasesInfo.getRoots(outputBuff).begin();
    auto eltwiseAllInputs = clusterTaskOp.getInputs();

    for (auto input : eltwiseAllInputs) {
        log.nest().trace("Checking input {0}", input.getType());

        auto inputBuff = VPUIP::getTopBufferOfNCEClusterTiling(clusterTaskOp, input);
        auto inputRootBuff = *aliasesInfo.getRoots(inputBuff).begin();

        auto nestLog = log.nest(2);
        if (inputRootBuff == outputRootBuff) {
            nestLog.trace("Output already replaced with input");
            return;
        }

        auto isNotSingleUsage = llvm::any_of(inputBuff.getUsers(), [&](mlir::Operation* userOp) {
            auto nceClustOp = clusterTaskOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
            return (userOp != clusterTaskOp.getOperation() && userOp != nceClustOp.getOperation());
        });

        if (isNotSingleUsage) {
            // This input is used by another operation, try next input
            continue;
        }

        auto inInterface = inputBuff.getType().dyn_cast<NDTypeInterface>();
        auto outInterface = outputRootBuff.getType().dyn_cast<NDTypeInterface>();
        if (!isCompatibleForInplaceOp(inInterface, outInterface, nestLog)) {
            continue;
        }

        if (inInterface.getElementType() != outInterface.getElementType()) {
            mlir::OpBuilder builder(clusterTaskOp);
            builder.setInsertionPointAfterValue(inputRootBuff);
            auto supportView =
                    builder.create<VPUIP::ViewOp>(inputBuff.getLoc(), outputRootBuff.getType(), inputRootBuff);
            outputRootBuff.replaceAllUsesWith(supportView.result());
            log.trace("Replacing output {0} with view result {1}", outputRootBuff.getLoc(), supportView.getLoc());
            return;
        }

        log.trace("Eltwise input Replaced with output {0}", inputRootBuff.getLoc());
        outputRootBuff.replaceAllUsesWith(inputRootBuff);
        return;
    }

    // If pass was unable to convert eltwise then compilation must fail since
    // operation was not tiled to fit into CMX.
    VPUX_THROW("Failed to convert Eltwise to in-place Eltwise {0}", clusterTaskOp.getLoc());
}

//
// ConvertEltwiseToInPlacePass
//

class ConvertEltwiseToInPlacePass final : public VPUIP::ConvertEltwiseToInPlaceBase<ConvertEltwiseToInPlacePass> {
public:
    explicit ConvertEltwiseToInPlacePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertEltwiseToInPlacePass::safeRunOnFunc() {
    auto& aliasesInfo = getAnalysis<AliasesInfo>();

    auto func = getFunction();
    func->walk([&](VPUIP::NCEClusterTaskOp op) {
        makeInPlaceEltwise(op, aliasesInfo, _log);
    });
}

}  // namespace

//
// createConvertEltwiseToInPlacePass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertEltwiseToInPlacePass(Logger log) {
    return std::make_unique<ConvertEltwiseToInPlacePass>(log);
}
