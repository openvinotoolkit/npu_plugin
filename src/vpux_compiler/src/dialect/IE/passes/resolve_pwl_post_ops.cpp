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

#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/VPU/pwl_utils.hpp"

#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// FusableOpRewriter
//

class FusableOpRewriter final : public mlir::OpInterfaceRewritePattern<IE::LayerWithPostOpInterface> {
public:
    FusableOpRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<IE::LayerWithPostOpInterface>(ctx), _log(log) {
        this->setDebugName("FusableOpRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::LayerWithPostOpInterface postOp,
                                        mlir::PatternRewriter& rewriter) const final;

    mlir::LogicalResult ensureRequantizationRange(IE::LayerWithPostOpInterface origOp, mlir::PatternRewriter& rewriter,
                                                  const VPU::PwlQuantReqs& quantReqs, const bool quantType) const;

    template <class PostOpType>
    mlir::LogicalResult unfusePostOp(IE::LayerWithPostOpInterface origOp, llvm::StringRef postOpName,
                                     mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
};

mlir::LogicalResult FusableOpRewriter::ensureRequantizationRange(IE::LayerWithPostOpInterface origOp,
                                                                 mlir::PatternRewriter& rewriter,
                                                                 const VPU::PwlQuantReqs& quantReqs,
                                                                 const bool quantType) const {
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
        if (auto quantizeCastOp = mlir::dyn_cast<IE::QuantizeCastOp>(user)) {
            return quantizeCastOp.input().getType() == pwlInType && quantizeCastOp.output().getType() == pwlOutType;
        }
        return false;
    };
    if (llvm::all_of(origOp->getResult(0).getUsers(), alreadyRequantized)) {
        return matchFailed(_log.nest(), rewriter, origOp, "Operation is already requantized");
    }

    _log.nest().trace("Adding QuantizeCast output operation");

    auto clone = rewriter.clone(*origOp.getOperation());
    clone->getResult(0).setType(pwlInType);
    if (quantType) {
        rewriter.replaceOpWithNewOp<IE::QuantizeCastOp>(origOp, pwlOutType, clone->getResult(0), pwlOutElemType);
    } else {
        auto realElemType = mlir::FloatType::getF16(getContext());

        _log.nest().trace("Insert Dequantize op '{0}' -> '{1}'", pwlInElemType, realElemType);
        auto dequantizeOp = rewriter.create<IE::DequantizeOp>(origOp.getLoc(), clone->getResult(0), realElemType);

        auto result = dequantizeOp.getResult();

        _log.nest().trace("Insert Quantize op '{0}' -> '{1}'", realElemType, pwlOutElemType);
        rewriter.replaceOpWithNewOp<IE::QuantizeOp>(origOp, result, pwlOutElemType);
    }
    return mlir::success();
}

template <class PostOpType>
mlir::LogicalResult FusableOpRewriter::unfusePostOp(IE::LayerWithPostOpInterface origOp, llvm::StringRef postOpName,
                                                    mlir::PatternRewriter& rewriter) const {
    _log.nest().trace("Unfusing post-op {0} from {1}", postOpName, origOp->getName());
    rewriter.setInsertionPointAfter(origOp);
    auto postOp =
            rewriter.create<PostOpType>(origOp->getLoc(), origOp->getResult(0), origOp.getPostOpAttrs().getValue());
    origOp->getResult(0).replaceAllUsesExcept(postOp->getResult(0), llvm::SmallPtrSet<mlir::Operation*, 1>{postOp});
    origOp.clearPostOp();

    return mlir::success();
}

/*
 * The PWL table requires fixed input and output quantization ranges. This pass inserts QuantizeCast after operations
 * with fused PWL post-ops to ensure correct ranges after the PPE requantization step and to ensure the consumer ops
 * use the correct range that is generated by the PWL table.
 *
 * Additionally, PWL activation functions can provide low accuracy results for float computation, so they get unfused
 * for SHAVE execution. The cases with per-axis quantization are also unfused until proper support is added
 * (EISW-25777).
 */
mlir::LogicalResult FusableOpRewriter::matchAndRewrite(IE::LayerWithPostOpInterface origOp,
                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got operation '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto isQuantizedPerTensor = [](mlir::Operation* op) -> bool {
        auto inputType = op->getOperand(0).getType().cast<vpux::NDTypeInterface>().getElementType();
        auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>().getElementType();
        return inputType.isa<mlir::quant::UniformQuantizedType>() &&
               outputType.isa<mlir::quant::UniformQuantizedType>();
    };

    const auto postOp = origOp.getPostOp();
    if (!postOp.hasValue()) {
        return matchFailed(_log.nest(), rewriter, origOp, "Operation has no fused post-op");
    }

    const auto postOpName = postOp.getValue().getStringRef();

    if (postOpName == IE::SigmoidOp::getOperationName()) {
        if (isQuantizedPerTensor(origOp)) {
            return ensureRequantizationRange(origOp, rewriter, VPU::getPwlQuantReqs(VPU::PPEMode::SIGMOID), true);
        } else {
            return unfusePostOp<IE::SigmoidOp>(origOp, postOpName, rewriter);
        }
    } else if (postOpName == IE::TanhOp::getOperationName()) {
        if (isQuantizedPerTensor(origOp)) {
            return ensureRequantizationRange(origOp, rewriter, VPU::getPwlQuantReqs(VPU::PPEMode::TANH), true);
        } else {
            return unfusePostOp<IE::TanhOp>(origOp, postOpName, rewriter);
        }
    } else if (postOpName == IE::LeakyReluOp::getOperationName()) {
        if (isQuantizedPerTensor(origOp)) {
            const auto requantCustomPWL = VPU::getCustomPwlQuantReqs(origOp);
            if (requantCustomPWL == nullptr) {
                return matchFailed(_log.nest(), rewriter, origOp, "QuantizeCast or Requantize is not needed");
            }

            return ensureRequantizationRange(origOp, rewriter, requantCustomPWL[0], false);
        } else {
            return unfusePostOp<IE::LeakyReluOp>(origOp, postOpName, rewriter);
        }
    }

    return matchFailed(_log.nest(), rewriter, origOp, "Non-PWL post-op");
}

//
// ResolvePWLPostOpsPass
//

class ResolvePWLPostOpsPass final : public IE::ResolvePWLPostOpsBase<ResolvePWLPostOpsPass> {
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

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FusableOpRewriter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createResolvePWLPostOpsPass(Logger log) {
    return std::make_unique<ResolvePWLPostOpsPass>(log);
}
