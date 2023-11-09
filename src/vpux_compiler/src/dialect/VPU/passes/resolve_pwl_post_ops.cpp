//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/pwl_utils.hpp"

#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/numeric.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// FusableOpRewriter
//

class FusableOpRewriter final : public mlir::OpInterfaceRewritePattern<VPU::NCEOpInterface> {
public:
    FusableOpRewriter(mlir::MLIRContext* ctx, VPU::ArchKind arch, Logger log)
            : mlir::OpInterfaceRewritePattern<VPU::NCEOpInterface>(ctx), _arch(arch), _log(log) {
        this->setDebugName("FusableOpRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(VPU::NCEOpInterface origOp, mlir::PatternRewriter& rewriter) const final;

    mlir::LogicalResult ensureRequantizationRange(VPU::NCEOpInterface origOp, mlir::PatternRewriter& rewriter,
                                                  const VPU::PwlQuantReqs& quantReqs) const;

private:
    VPU::ArchKind _arch;
    Logger _log;
};

mlir::LogicalResult FusableOpRewriter::ensureRequantizationRange(VPU::NCEOpInterface origOp,
                                                                 mlir::PatternRewriter& rewriter,
                                                                 const VPU::PwlQuantReqs& quantReqs) const {
    _log.nest().trace("Ensure requantization range for {0}", origOp->getName());

    const auto origType = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto qType = origType.getElementType().dyn_cast<mlir::quant::UniformQuantizedType>();
    VPUX_THROW_UNLESS(qType != nullptr, "Output tensor expected to be quantized per-tensor. Got {0}", origType);

    const auto pwlInElemType = mlir::quant::UniformQuantizedType::getChecked(
            origOp.getLoc(), qType.getFlags(), qType.getStorageType(), qType.getExpressedType(), quantReqs.input.scale,
            quantReqs.input.zeroPoint, qType.getStorageTypeMin(), qType.getStorageTypeMax());
    const auto pwlOutElemType = mlir::quant::UniformQuantizedType::getChecked(
            origOp.getLoc(), qType.getFlags(), qType.getStorageType(), qType.getExpressedType(), quantReqs.output.scale,
            quantReqs.output.zeroPoint, qType.getStorageTypeMin(), qType.getStorageTypeMax());

    const auto pwlInType = origType.changeElemType(pwlInElemType);
    const auto pwlOutType = origType.changeElemType(pwlOutElemType);

    const auto alreadyRequantized = [&](mlir::Operation* user) {
        if (auto quantizeCastOp = mlir::dyn_cast<VPU::QuantizeCastOp>(user)) {
            return quantizeCastOp.input().getType() == pwlInType && quantizeCastOp.output().getType() == pwlOutType;
        }
        // Here we check if the next operation is a DequantizeOp. DequantizeOp is converted to NCEEltwiseOp.
        if (auto eltwiseOp = mlir::dyn_cast<VPU::NCEEltwiseOp>(user)) {
            return eltwiseOp.input1().getType() == pwlInType && eltwiseOp.input2().getType() == pwlInType &&
                   eltwiseOp.output().getType().cast<vpux::NDTypeInterface>().getElementType().isF16();
        }
        return false;
    };
    if (llvm::all_of(origOp->getResult(0).getUsers(), alreadyRequantized)) {
        return matchFailed(_log.nest(), rewriter, origOp, "Operation is already requantized");
    }

    auto clone = rewriter.clone(*origOp.getOperation());
    clone->getResult(0).setType(pwlInType);

    auto isRequantizeNeeded = llvm::any_of(origOp->getResult(0).getUsers(), [](mlir::Operation* userOp) {
        return !mlir::isa<VPU::NCEOpInterface>(userOp);
    });

    if (!isRequantizeNeeded) {
        _log.nest().trace("Adding QuantizeCast output operation");
        rewriter.replaceOpWithNewOp<VPU::QuantizeCastOp>(origOp, pwlOutType, clone->getResult(0), pwlOutElemType);

        return mlir::success();
    }
    _log.nest().trace("Adding Requantization Eltwise operations");

    auto realElemType = mlir::FloatType::getF16(getContext());

    auto opType = VPU::EltwiseType::AND;

    auto floatOutType = pwlInType.changeElemType(realElemType);

    auto ppeTaskAttr = VPU::getNCEEltwisePPETaskAttr(pwlInType, pwlInType, floatOutType, nullptr, origOp.getLoc(),
                                                     opType, origOp.getContext(), _arch);
    auto nceOpDequant =
            rewriter.create<VPU::NCEEltwiseOp>(origOp.getLoc(), floatOutType, clone->getResult(0), clone->getResult(0),
                                               VPU::EltwiseTypeAttr::get(this->getContext(), opType), ppeTaskAttr,
                                               /*multi_cluster_strategyAttr=*/nullptr,
                                               /*is_inplace=*/nullptr);

    ppeTaskAttr = VPU::getNCEEltwisePPETaskAttr(floatOutType, floatOutType, origType, nullptr, origOp.getLoc(), opType,
                                                origOp.getContext(), _arch);

    rewriter.replaceOpWithNewOp<VPU::NCEEltwiseOp>(origOp, origType, nceOpDequant.getResult(), nceOpDequant.getResult(),
                                                   VPU::EltwiseTypeAttr::get(this->getContext(), opType), ppeTaskAttr,
                                                   /*multi_cluster_strategyAttr=*/nullptr,
                                                   /*is_inplace=*/nullptr);

    return mlir::success();
}

/*
 * The PWL table requires fixed input and output quantization ranges. This pass inserts QuantizeCast/Requantization
 * after operations with fused PWL post-ops to ensure correct ranges after the PPE requantization step and to ensure
 * the consumer ops use the correct range that is generated by the PWL table.
 *
 * This pass inserts QuantizeCast if the next operation is a hardware operation, otherwise we insert a dequantize
 * quantize to match the expected input of the next software operation
 */
mlir::LogicalResult FusableOpRewriter::matchAndRewrite(VPU::NCEOpInterface origOp,
                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got operation '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto ppeTask = origOp.getPPE();
    if (!ppeTask.has_value()) {
        return matchFailed(_log.nest(), rewriter, origOp, "Operation has no PPE task");
    }

    const auto PPEMode = ppeTask.value().getMode().getValue();

    if (PPEMode == VPU::PPEMode::SIGMOID) {
        return ensureRequantizationRange(origOp, rewriter, VPU::getPwlQuantReqs(VPU::PPEMode::SIGMOID));
    } else if (PPEMode == VPU::PPEMode::TANH) {
        return ensureRequantizationRange(origOp, rewriter, VPU::getPwlQuantReqs(VPU::PPEMode::TANH));
    } else if (PPEMode == VPU::PPEMode::FLEXARB) {
        return matchFailed(_log.nest(), rewriter, origOp, "FLEXARB PWL operations does not need requantization");
    }

    return matchFailed(_log.nest(), rewriter, origOp, "Non-PWL post-op");
}

//
// ResolvePWLPostOpsPass
//

class ResolvePWLPostOpsPass final : public VPU::ResolvePWLPostOpsBase<ResolvePWLPostOpsPass> {
public:
    explicit ResolvePWLPostOpsPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void ResolvePWLPostOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FusableOpRewriter>(&ctx, arch, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createResolvePWLPostOpsPass(Logger log) {
    return std::make_unique<ResolvePWLPostOpsPass>(log);
}
