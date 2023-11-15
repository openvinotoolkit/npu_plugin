//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

using namespace vpux;

namespace {

// Who can be the NCEEltwiseOp input producer:
// 1. Input/Constant
// 2. Generic AllocOp
// 3. Generic TaskOp
// 4. Chain of pure ViewLike ops followed by a TaskOp/AllocOp/Input/Constant
// In all cases check that the result of actual TaskOp/AllocOp/Input/Constant is used only by inplace
// NCEEltwiseOp
bool isEltwiseTheOnlyConsumer(VPUIP::NCEClusterTaskOp clusterTaskOp, mlir::Value inputBuff, Logger log) {
    // Utility function for checking if an operation is Copy or pure ViewLike op
    const auto isNoDataEffectOp = [&](mlir::Operation* op) {
        auto clustOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(op);
        if (clustOp == nullptr) {
            return VPUIP::isPureViewOp(op) || mlir::isa<VPUIP::CopyOp>(op);
        }
        auto innerOp = clustOp.getInnerTaskOp();
        return mlir::isa<VPUIP::CopyOp>(innerOp);
    };

    // Utility function for checking that two different SubViews have the same function
    const auto areSameSubView = [](VPUIP::SubViewOp srcSubView, VPUIP::SubViewOp siblingSubView) {
        return (srcSubView.static_offsets() == siblingSubView.static_offsets()) &&
               (srcSubView.static_sizes() == siblingSubView.static_sizes()) &&
               (srcSubView.static_strides() == siblingSubView.static_strides());
    };

    // Utility function for checking if an operation placed between in place NCEEltwise and the root input
    // producer is consumed only by the in place NCEEltwise
    //  Root Input producer
    //   |            |
    // CopyOp()      CopyOp()
    //   \            /
    //    NCEEltwise()
    const auto isThisUserOfOp = [&](mlir::Operation* userToCompare, mlir::Operation* upperOp) {
        auto userOp = upperOp;
        while (userOp != nullptr && isNoDataEffectOp(userOp)) {
            if (!userOp->getResult(0).hasOneUse()) {
                return false;
            }
            userOp = *userOp->getResult(0).getUsers().begin();
        }
        return userOp == userToCompare;
    };

    // Utility function that checks if input of noDataEffectOp is used by only one Task Op
    const auto isSupportedMultiUserScenario = [&](mlir::Operation* noDataEffectOp) {
        // If the input of noDataEffectOp has more users then it can be one of the following scenarios
        // 1. The users are all SubViewOps in which case it is needed to check if there are different SubView
        // ops which do exactly the same thing and if yes then it means that the potentialViewLikeInputOp has
        // different users
        // 2. There are users which are not SubView ops, in this case it is needed to check if all these users
        // goes as input to the same NCEEltwise, if not it means that potentialViewLikeInputOp more then one
        // user
        auto noDataEffectOpInput = noDataEffectOp->getOperand(0);
        if (mlir::isa<VPUIP::SubViewOp>(noDataEffectOp)) {
            auto subViewOp = mlir::dyn_cast<VPUIP::SubViewOp>(noDataEffectOp);
            for (auto userOp : llvm::make_early_inc_range(noDataEffectOpInput.getUsers())) {
                auto siblingSubViewOp = mlir::dyn_cast<VPUIP::SubViewOp>(userOp);
                if (siblingSubViewOp == nullptr) {
                    return false;
                }
                if (siblingSubViewOp != subViewOp && areSameSubView(subViewOp, siblingSubViewOp)) {
                    log.nest().trace("The NCEEltiwse input has sibling SubView ops with the same function.");
                    return false;
                }
            }
        } else {
            auto nceClustOp = clusterTaskOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
            auto originalTaskOp = nceClustOp != nullptr ? nceClustOp.getOperation() : clusterTaskOp.getOperation();
            auto firstUserOp = *noDataEffectOpInput.getUsers().begin();
            for (auto userOp : llvm::make_early_inc_range(noDataEffectOpInput.getUsers())) {
                if (firstUserOp != userOp && !isThisUserOfOp(originalTaskOp, userOp)) {
                    log.nest().trace("The NCEEltiwse root input is used by other TaskOp");
                    return false;
                }
            }
        }
        return true;
    };

    // Move up over all pure ViewLikeOps and CopyOps to get the actual producer of the NCEEltwise's input
    auto potentialInputProducerValue = inputBuff;
    auto nceClustOp = clusterTaskOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
    auto lastVisitedOp = nceClustOp != nullptr ? nceClustOp.getOperation() : clusterTaskOp.getOperation();
    do {
        if (!potentialInputProducerValue.hasOneUse() && !isSupportedMultiUserScenario(lastVisitedOp)) {
            return false;
        }
        auto potentialInputProducerOp = potentialInputProducerValue.getDefiningOp();
        if (potentialInputProducerOp == nullptr || potentialInputProducerOp->getOperands().empty()) {
            log.nest().trace("Found potentialInputProducerOp that has no operands.");
            return true;
        }
        lastVisitedOp = potentialInputProducerOp;
        potentialInputProducerValue = potentialInputProducerOp->getOperand(0);
    } while (lastVisitedOp != nullptr && isNoDataEffectOp(lastVisitedOp));
    return true;
}

void makeInPlaceEltwise(VPUIP::NCEClusterTaskOp clusterTaskOp, AliasesInfo& aliasesInfo, Logger log) {
    auto eltwiseAllInputs = clusterTaskOp.getInputs();
    // Get the root output buffer of the clusterTaskOp
    auto getOutputRootBuffOfNCEClusterTiling = [](mlir::Operation* innerOp) {
        if (auto nceClustOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(innerOp->getParentOp())) {
            return VPUIP::getLayerOutputs(nceClustOp)[0];
        }
        return VPUIP::getLayerOutputs(innerOp)[0];
    };

    auto outputRootBuff = getOutputRootBuffOfNCEClusterTiling(clusterTaskOp);
    for (auto input : eltwiseAllInputs) {
        log.trace("Checking input at `{0}`", input.getLoc());

        auto inputBuff = VPUIP::getTopBufferOfNCEClusterTiling(clusterTaskOp, input);
        auto inputRootBuff = *aliasesInfo.getRoots(inputBuff).begin();

        auto nestLog = log.nest(2);
        if (inputRootBuff == outputRootBuff) {
            nestLog.trace("Output already replaced with input");
            return;
        }

        if (!isEltwiseTheOnlyConsumer(clusterTaskOp, inputBuff, log)) {
            nestLog.trace("This input is used by another operation, try next input.");
            continue;
        }

        const auto inInterface = inputBuff.getType().dyn_cast<NDTypeInterface>();
        const auto outInterface = outputRootBuff.getType().dyn_cast<NDTypeInterface>();
        if (!isCompatibleForInplaceOp(inInterface, outInterface, nestLog)) {
            continue;
        }

        // Ensure element type compatibility
        if (inInterface.getElementType() != outInterface.getElementType()) {
            mlir::OpBuilder builder(clusterTaskOp);
            builder.setInsertionPointAfterValue(inputRootBuff);
            auto supportView =
                    builder.create<VPUIP::ViewOp>(inputBuff.getLoc(), outputRootBuff.getType(), inputRootBuff);
            aliasesInfo.addAlias(inputRootBuff, supportView.result());
            inputRootBuff = supportView.result();
        }

        // Ensure distribution compatibility
        const auto inDistributedType = VPUIP::extractDataType(inputRootBuff).dyn_cast<VPUIP::DistributedBufferType>();
        const auto outDistributedType = VPUIP::extractDataType(outputRootBuff).dyn_cast<VPUIP::DistributedBufferType>();
        if (inDistributedType != nullptr && outDistributedType != nullptr &&
            inDistributedType.getDistribution() != outDistributedType.getDistribution()) {
            if (VPU::areDistributionAttrsCompatible(inDistributedType, outDistributedType, true).succeeded()) {
                mlir::OpBuilder builder(clusterTaskOp);
                builder.setInsertionPointAfterValue(inputRootBuff);
                auto distributedCastOp = builder.create<VPUIP::DistributedCastOp>(
                        clusterTaskOp.getLoc(), outputRootBuff.getType(), inputRootBuff);
                aliasesInfo.addAlias(inputRootBuff, distributedCastOp.output());
                inputRootBuff = distributedCastOp.output();
            } else {
                nestLog.trace("Incompatible input/output dist modes {0} {1}", inDistributedType, outDistributedType);
                continue;
            }
        }

        outputRootBuff.replaceAllUsesWith(inputRootBuff);

        const auto getEltwiseResult = [&]() {
            if (auto nceClustOp = mlir::dyn_cast_or_null<VPUIP::NCEClusterTilingOp>(clusterTaskOp->getParentOp())) {
                return nceClustOp->getResult(0);
            } else {
                return clusterTaskOp->getResult(0);
            }
        };

        const auto eltwiseResult = getEltwiseResult();
        aliasesInfo.removeAlias(eltwiseResult);
        aliasesInfo.addAlias(inputRootBuff, eltwiseResult);

        log.trace("Eltwise input Replaced with output {0}", inputRootBuff.getLoc());
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
    auto func = getOperation();

    const auto isEltwiseInplaceCandidate = [](VPUIP::NCEClusterTaskOp op) {
        if (op.task_type() != VPUIP::NCETaskType::ELTWISE) {
            return false;
        }
        return op.is_inplace().value_or(false);
    };

    func->walk([&](VPUIP::NCEClusterTaskOp op) {
        if (isEltwiseInplaceCandidate(op)) {
            _log.trace("Found inplace eltwise at {0}", op.getLoc());
            makeInPlaceEltwise(op, aliasesInfo, _log);
        }
    });
}

}  // namespace

//
// createConvertEltwiseToInPlacePass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertEltwiseToInPlacePass(Logger log) {
    return std::make_unique<ConvertEltwiseToInPlacePass>(log);
}
