//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/Value.h>

using namespace vpux;

namespace {
bool hasLeakyReLUPostOp(mlir::Operation* op) {
    auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(op);
    if (layerWithPostOp == nullptr) {
        return false;
    }

    const auto postOpName = layerWithPostOp.getPostOp();
    return postOpName.has_value() && postOpName.value().getStringRef() == IE::LeakyReluOp::getOperationName();
}

bool isMixPrecisionSupported(mlir::Operation* origOp, const VPU::ArchKind& arch, const bool isPReLUSupported,
                             Logger log) {
    if (!mlir::isa<IE::ConvolutionOp, IE::GroupConvolutionOp, IE::MaxPoolOp, IE::AddOp, IE::AvgPoolOp>(origOp)) {
        return false;
    }

    // The old arch VPUX30XX is not supported.
    const std::set<VPU::ArchKind> incompatibleTargets = {
            VPU::ArchKind::VPUX30XX,
    };

    // Check that the kernel size are not exceding the NCE HW limits
    if (VPUIP::NCEInvariant::verifyKernel(origOp, log).failed()) {
        return false;
    }

    // If the Add operands have different shapes the operation will be mapped on SHAVE, which does not support mixed
    // precision operations
    if (mlir::isa<IE::AddOp>(origOp)) {
        auto addOp = mlir::dyn_cast<IE::AddOp>(origOp);
        const auto shape1 = getShape(addOp.input1());
        const auto shape2 = getShape(addOp.input2());
        if (shape1 != shape2)
            return false;
    }

    // Mixed precision for average pooling is not supported for VPUX30XX target
    if (mlir::isa<IE::AvgPoolOp>(origOp) && incompatibleTargets.count(arch) > 0) {
        return false;
    }

    if (incompatibleTargets.count(arch) == 0) {
        // Float input with quantized output supports leaky ReLU when quantize out is per-tensor.
        // Further checks are not necessary, bail out.
        if (isPReLUSupported) {
            return true;
        }

        // HW limitations below do not apply to VPUX37XX
        // However, leaky ReLU does not work accurately in quant in / float out mode.
        // In quant in / float out flow, PReLU alpha coefficient can only be represented as prelu_mult.
        // prelu_shift is not available in such configuration.
        // Therefore, it becomes problematic to express rational negative slopes.
        // See E#58368 for details.
        const auto hasLeakyReLUConsumer = llvm::any_of(origOp->getUsers(), [](mlir::Operation* op) {
            return mlir::isa<IE::LeakyReluOp>(op);
        });

        // Thus, mixed precision is supported only when consumers and post-ops are not leaky ReLU
        return !hasLeakyReLUConsumer && !hasLeakyReLUPostOp(origOp);
    }

    // NOTE: HW limitation, in mixed mode the grids of the MPEs are conflicting between
    // each other, which leads to 1x1 workloads.
    auto outputShape = getShape(origOp->getResult(0));
    return outputShape[Dims4D::Act::H] == 1 && outputShape[Dims4D::Act::W] == 1;
}

bool areAnyUserQuantizeOps(mlir::Operation* op) {
    return llvm::any_of(op->getUsers(), [](mlir::Operation* op) {
        return mlir::isa<IE::QuantizeOp>(op);
    });
}

/*
 *  Bias will be rescaled for mixed precision and written in weight table later, so need to check whether the
 *  rescaled bias range exceeds or not
 */
template <class ConcreteOp>
mlir::LogicalResult checkRescaledBiasRange(ConcreteOp op) {
    auto inputDequantizeOp = op.input().template getDefiningOp<IE::DequantizeOp>();
    auto filterDequantizeOp = op.filter().template getDefiningOp<IE::DequantizeOp>();
    if (!inputDequantizeOp || !filterDequantizeOp) {
        return mlir::failure();
    }

    if (auto biasAttr = op.bias()) {
        const auto inElemType =
                inputDequantizeOp.input().getType().template cast<vpux::NDTypeInterface>().getElementType();
        const auto filterElemType =
                filterDequantizeOp.input().getType().template cast<vpux::NDTypeInterface>().getElementType();
        auto biasConstOp = biasAttr.template getDefiningOp<Const::DeclareOp>();
        Const::ContentAttr bias;
        if (biasConstOp) {
            bias = biasConstOp.getContentAttr();
        } else {
            auto biasDequantOp = biasAttr.template getDefiningOp<IE::DequantizeOp>();
            if (!biasDequantOp) {
                return mlir::failure();
            }
            auto inputConst = biasDequantOp.input().template getDefiningOp<Const::DeclareOp>();
            const auto newConstAttr = inputConst.getContentAttr().dequantize();
            bias = newConstAttr;
        }
        const auto OC = getShape(op.filter())[Dims4D::Filter::OC];
        if (mlir::failed(VPU::NCESparsity::getRescaledBias(bias, inElemType, filterElemType, OC))) {
            return mlir::failure();
        }
    }
    return mlir::success();
}

bool checkQuantApproximation(mlir::Operation* op) {
    SmallVector<double> scales;
    const auto outElemType = op->getResult(0).getType().cast<vpux::NDTypeInterface>().getElementType();
    if (outElemType.isa<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto perAxis = outElemType.cast<mlir::quant::UniformQuantizedPerAxisType>();
        std::copy(perAxis.getScales().begin(), perAxis.getScales().end(), std::back_inserter(scales));
    } else if (outElemType.isa<mlir::quant::UniformQuantizedType>()) {
        const auto perTensor = outElemType.cast<mlir::quant::UniformQuantizedType>();
        scales = {perTensor.getScale()};
    } else {
        return false;
    }

    // Check that all scales can be approximated without post-shift (i.e. exponent must fit 15 bits).
    // Negative power is used here because rescaling is computed as scale_in * scale_w / scale_out
    // In case of float input and float weights, scale_in = 1, scale_w = 1, thus we get 1 / scale_out.
    const double scaleLimit = std::pow(2, -15);
    for (const auto& scale : scales) {
        if (std::fabs(scale) < scaleLimit) {
            return false;
        }
    }

    return true;
}

//
// ConvertToMixedPrecisionPass
//

class ConvertToMixedPrecisionPass final : public IE::ConvertToMixedPrecisionBase<ConvertToMixedPrecisionPass> {
public:
    explicit ConvertToMixedPrecisionPass(const bool allowFP16ToU8, Logger log)
            : _convertFloatInQuantOut(allowFP16ToU8), _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    const bool _convertFloatInQuantOut;
    Logger _log;
};

class FloatOutConvRewriter final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    FloatOutConvRewriter(mlir::MLIRContext* ctx, const VPU::ArchKind& arch, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _arch(arch), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp convolutionOp, mlir::PatternRewriter& rewriter) const final;

private:
    const VPU::ArchKind _arch;
    Logger _log;
};

mlir::LogicalResult FloatOutConvRewriter::matchAndRewrite(IE::ConvolutionOp convolutionOp,
                                                          mlir::PatternRewriter& rewriter) const {
    if (areAnyUserQuantizeOps(convolutionOp) || !isMixPrecisionSupported(convolutionOp, _arch, false, _log)) {
        return mlir::failure();
    }
    if (mlir::failed(checkRescaledBiasRange(convolutionOp))) {
        return mlir::failure();
    }
    auto dequantizeOp = convolutionOp.input().getDefiningOp<IE::DequantizeOp>();
    auto filterDequantizeOp = convolutionOp.filter().getDefiningOp<IE::DequantizeOp>();

    rewriter.replaceOpWithNewOp<IE::ConvolutionOp>(
            convolutionOp, convolutionOp.getType(), dequantizeOp.input(), filterDequantizeOp.input(),
            convolutionOp.bias(), convolutionOp.strides(), convolutionOp.pads_begin(), convolutionOp.pads_end(),
            convolutionOp.dilations(), convolutionOp.post_opAttr());

    return mlir::success();
}

class FloatOutGroupConvRewriter final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    FloatOutGroupConvRewriter(mlir::MLIRContext* ctx, const VPU::ArchKind& arch, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx), _arch(arch), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp groupConvolutionOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    const VPU::ArchKind _arch;
    Logger _log;
};

mlir::LogicalResult FloatOutGroupConvRewriter::matchAndRewrite(IE::GroupConvolutionOp groupConvolutionOp,
                                                               mlir::PatternRewriter& rewriter) const {
    if (areAnyUserQuantizeOps(groupConvolutionOp) || !isMixPrecisionSupported(groupConvolutionOp, _arch, false, _log)) {
        return mlir::failure();
    }
    if (mlir::failed(checkRescaledBiasRange(groupConvolutionOp))) {
        return mlir::failure();
    }

    auto dequantizeOp = groupConvolutionOp.input().getDefiningOp<IE::DequantizeOp>();
    auto filterDequantizeOp = groupConvolutionOp.filter().getDefiningOp<IE::DequantizeOp>();

    rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(
            groupConvolutionOp, groupConvolutionOp.getType(), dequantizeOp.input(), filterDequantizeOp.input(),
            groupConvolutionOp.bias(), groupConvolutionOp.strides(), groupConvolutionOp.pads_begin(),
            groupConvolutionOp.pads_end(), groupConvolutionOp.dilations(), groupConvolutionOp.groupsAttr(),
            groupConvolutionOp.post_opAttr());

    return mlir::success();
}

class FloatOutMaxPoolRewriter final : public mlir::OpRewritePattern<IE::MaxPoolOp> {
public:
    FloatOutMaxPoolRewriter(mlir::MLIRContext* ctx, const VPU::ArchKind& arch, Logger log)
            : mlir::OpRewritePattern<IE::MaxPoolOp>(ctx), _arch(arch), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MaxPoolOp maxPoolOp, mlir::PatternRewriter& rewriter) const final;

private:
    const VPU::ArchKind _arch;
    Logger _log;
};

mlir::LogicalResult FloatOutMaxPoolRewriter::matchAndRewrite(IE::MaxPoolOp maxPoolOp,
                                                             mlir::PatternRewriter& rewriter) const {
    if (areAnyUserQuantizeOps(maxPoolOp) || !isMixPrecisionSupported(maxPoolOp, _arch, false, _log)) {
        return mlir::failure();
    }
    auto dequantizeOp = maxPoolOp.input().getDefiningOp<IE::DequantizeOp>();
    if (dequantizeOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::MaxPoolOp>(
            maxPoolOp, maxPoolOp.getType(), dequantizeOp.input(), maxPoolOp.kernel_size(), maxPoolOp.strides(),
            maxPoolOp.pads_begin(), maxPoolOp.pads_end(), maxPoolOp.rounding_type(), maxPoolOp.post_opAttr());

    return mlir::success();
}

class FloatOutAvgPoolRewriter final : public mlir::OpRewritePattern<IE::AvgPoolOp> {
public:
    FloatOutAvgPoolRewriter(mlir::MLIRContext* ctx, const VPU::ArchKind& arch, Logger log)
            : mlir::OpRewritePattern<IE::AvgPoolOp>(ctx), _arch(arch), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AvgPoolOp avgPoolOp, mlir::PatternRewriter& rewriter) const final;

private:
    const VPU::ArchKind _arch;
    Logger _log;
};

mlir::LogicalResult FloatOutAvgPoolRewriter::matchAndRewrite(IE::AvgPoolOp avgPoolOp,
                                                             mlir::PatternRewriter& rewriter) const {
    if (areAnyUserQuantizeOps(avgPoolOp) || !isMixPrecisionSupported(avgPoolOp, _arch, false, _log)) {
        return mlir::failure();
    }
    auto dequantizeOp = avgPoolOp.input().getDefiningOp<IE::DequantizeOp>();
    if (dequantizeOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::AvgPoolOp>(avgPoolOp, avgPoolOp.getType(), dequantizeOp.input(),
                                               avgPoolOp.kernel_size(), avgPoolOp.strides(), avgPoolOp.pads_begin(),
                                               avgPoolOp.pads_end(), avgPoolOp.rounding_typeAttr(),
                                               avgPoolOp.exclude_padsAttr(), avgPoolOp.post_opAttr());

    return mlir::success();
}

class FloatOutAddRewriter final : public mlir::OpRewritePattern<IE::AddOp> {
public:
    FloatOutAddRewriter(mlir::MLIRContext* ctx, const VPU::ArchKind& arch, Logger log)
            : mlir::OpRewritePattern<IE::AddOp>(ctx), _arch(arch), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const VPU::ArchKind _arch;
    Logger _log;
};

mlir::Value findQuantizedInput(mlir::Value addOpInput) {
    // When the input is not a DequantizeOp, the pass is not applicable
    auto maybeDequant = addOpInput.getDefiningOp<IE::DequantizeOp>();
    if (maybeDequant == nullptr) {
        return nullptr;
    }

    // Only per-tensor quantization is supported for AddOp
    const auto dequantType = maybeDequant.input().getType().cast<vpux::NDTypeInterface>();
    if (!dequantType.getElementType().isa<mlir::quant::UniformQuantizedType>()) {
        return nullptr;
    }

    return maybeDequant.input();
}

mlir::LogicalResult FloatOutAddRewriter::matchAndRewrite(IE::AddOp addOp, mlir::PatternRewriter& rewriter) const {
    if (areAnyUserQuantizeOps(addOp) || !isMixPrecisionSupported(addOp, _arch, false, _log)) {
        return mlir::failure();
    }
    // This transformation assumes that each input has IE::DequantizeOp producer
    auto lhsDequant = findQuantizedInput(addOp.input1());
    if (lhsDequant == nullptr) {
        return mlir::failure();
    }
    auto rhsDequant = findQuantizedInput(addOp.input2());
    if (rhsDequant == nullptr) {
        return mlir::failure();
    }

    // If target architecture does not support different scales, check that they are the same
    const bool allowDifferentScales = supportsPerInputEltwiseScale(_arch);
    if (!allowDifferentScales) {
        auto lhsType = lhsDequant.getType().cast<vpux::NDTypeInterface>();
        auto lhsQuantType = lhsType.getElementType().cast<mlir::quant::UniformQuantizedType>();

        auto rhsType = rhsDequant.getType().cast<vpux::NDTypeInterface>();
        auto rhsQuantType = rhsType.getElementType().cast<mlir::quant::UniformQuantizedType>();
        if (!isDoubleEqual(lhsQuantType.getScale(), rhsQuantType.getScale())) {
            return mlir::failure();
        }
    }

    rewriter.replaceOpWithNewOp<IE::AddOp>(addOp, addOp.getType(), lhsDequant, rhsDequant, addOp.auto_broadcast(),
                                           addOp.post_opAttr());

    return mlir::success();
}

// Search for FloatInput in the following patterns
// BlockArg -> IE.Quantize -> NCE
mlir::Value findFloatInput(mlir::Value nceOpInput) {
    auto maybeQuantize = nceOpInput.getDefiningOp<IE::QuantizeOp>();
    if (maybeQuantize == nullptr) {
        return nullptr;
    }

    // So far, only the first NCE task should be executed in fp16/u8 mode.
    // The main problem with this mode is that FakeQuantize is removed from the input completely.
    // Without FakeQuantize the information about data clamping is lost.
    // It makes sense to omit clamping only when the input data fits the range required for a given NCE task.
    // For some models performance gain is worth the risk of losing the clamping information.
    if (!maybeQuantize.input().isa<mlir::BlockArgument>()) {
        return nullptr;
    }

    return maybeQuantize.input();
}

bool isFloatInputSupported(mlir::Operation* origOp, const VPU::ArchKind& arch, const bool onlyPerTensorQuant,
                           Logger log) {
    if (!isMixPrecisionSupported(origOp, arch, false, log)) {
        return false;
    }

    const auto outElemType = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getElementType();
    if (!outElemType.isa<mlir::quant::QuantizedType>()) {
        return false;
    }

    if (onlyPerTensorQuant && outElemType.isa<mlir::quant::UniformQuantizedPerAxisType>()) {
        return false;
    }

    if (!checkQuantApproximation(origOp)) {
        return false;
    }

    // Before the conversion, the input of the operation is quantized.
    const auto inElemType = origOp->getOperand(0).getType().cast<vpux::NDTypeInterface>().getElementType();
    return inElemType.isa<mlir::quant::QuantizedType>();
}

class FloatInConvRewriter final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    FloatInConvRewriter(mlir::MLIRContext* ctx, const VPU::ArchKind& arch, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _arch(arch), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const VPU::ArchKind _arch;
    Logger _log;
};

mlir::LogicalResult FloatInConvRewriter::matchAndRewrite(IE::ConvolutionOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    if (!isFloatInputSupported(origOp, _arch, false, _log)) {
        return mlir::failure();
    }

    auto maybeFloatInput = findFloatInput(origOp->getOperand(0));
    if (maybeFloatInput == nullptr) {
        return mlir::failure();
    }

    const auto dstElemType = maybeFloatInput.getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto dequantFilter = rewriter.createOrFold<IE::DequantizeOp>(origOp->getLoc(), origOp.filter(), dstElemType);
    VPUX_THROW_UNLESS(dequantFilter != nullptr, "Failed to de-quantize given filter");
    rewriter.replaceOpWithNewOp<IE::ConvolutionOp>(origOp, origOp.getType(), maybeFloatInput, dequantFilter,
                                                   origOp.bias(), origOp.strides(), origOp.pads_begin(),
                                                   origOp.pads_end(), origOp.dilations(), origOp.post_opAttr());

    return mlir::success();
}

class FloatInGroupConvRewriter final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    FloatInGroupConvRewriter(mlir::MLIRContext* ctx, const VPU::ArchKind& arch, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx), _arch(arch), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const VPU::ArchKind _arch;
    Logger _log;
};

mlir::LogicalResult FloatInGroupConvRewriter::matchAndRewrite(IE::GroupConvolutionOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    if (!isFloatInputSupported(origOp, _arch, false, _log)) {
        return mlir::failure();
    }

    auto maybeFloatInput = findFloatInput(origOp->getOperand(0));
    if (maybeFloatInput == nullptr) {
        return mlir::failure();
    }

    const auto dstElemType = maybeFloatInput.getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto dequantFilter = rewriter.createOrFold<IE::DequantizeOp>(origOp->getLoc(), origOp.filter(), dstElemType);
    VPUX_THROW_UNLESS(dequantFilter != nullptr, "Failed to de-quantize given filter");
    rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(
            origOp, origOp.getType(), maybeFloatInput, dequantFilter, origOp.bias(), origOp.strides(),
            origOp.pads_begin(), origOp.pads_end(), origOp.dilations(), origOp.groupsAttr(), origOp.post_opAttr());

    return mlir::success();
}

class FloatInAvgPoolRewriter final : public mlir::OpRewritePattern<IE::AvgPoolOp> {
public:
    FloatInAvgPoolRewriter(mlir::MLIRContext* ctx, const VPU::ArchKind& arch, Logger log)
            : mlir::OpRewritePattern<IE::AvgPoolOp>(ctx), _arch(arch), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const VPU::ArchKind _arch;
    Logger _log;
};

mlir::LogicalResult FloatInAvgPoolRewriter::matchAndRewrite(IE::AvgPoolOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    if (!isFloatInputSupported(origOp, _arch, true, _log)) {
        return mlir::failure();
    }

    auto maybeFloatInput = findFloatInput(origOp->getOperand(0));
    if (maybeFloatInput == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::AvgPoolOp>(
            origOp, origOp.getType(), maybeFloatInput, origOp.kernel_size(), origOp.strides(), origOp.pads_begin(),
            origOp.pads_end(), origOp.rounding_typeAttr(), origOp.exclude_padsAttr(), origOp.post_opAttr());

    return mlir::success();
}

class FloatInAddRewriter final : public mlir::OpRewritePattern<IE::AddOp> {
public:
    FloatInAddRewriter(mlir::MLIRContext* ctx, const VPU::ArchKind& arch, Logger log)
            : mlir::OpRewritePattern<IE::AddOp>(ctx), _arch(arch), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const VPU::ArchKind _arch;
    Logger _log;
};

mlir::LogicalResult FloatInAddRewriter::matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const {
    if (!isFloatInputSupported(origOp, _arch, true, _log)) {
        return mlir::failure();
    }

    // Check that both inputs of IE.Add have float source.
    SmallVector<mlir::Value> floatInputs;
    for (unsigned idx = 0; idx < 2; idx++) {
        floatInputs.push_back(findFloatInput(origOp->getOperand(idx)));
    }
    const auto nullptrPredicate = [](const mlir::Value operand) -> bool {
        return operand == nullptr;
    };
    if (std::any_of(floatInputs.begin(), floatInputs.end(), nullptrPredicate)) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::AddOp>(origOp, origOp.getType(), floatInputs[0], floatInputs[1],
                                           origOp.auto_broadcast(), origOp.post_opAttr());

    return mlir::success();
}

class QuantizeWithNCERewriter final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    QuantizeWithNCERewriter(mlir::MLIRContext* ctx, const VPU::ArchKind& arch, Logger log)
            : mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _arch(arch), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const VPU::ArchKind _arch;
    Logger _log;
};

mlir::LogicalResult QuantizeWithNCERewriter::matchAndRewrite(IE::QuantizeOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    const auto maybeNCETask = origOp.input().getDefiningOp();
    if (maybeNCETask == nullptr) {
        return matchFailed(rewriter, origOp, "Producer is a block argument");
    }
    if (!maybeNCETask->getResult(0).hasOneUse()) {
        return matchFailed(rewriter, origOp, "NCE task has more than one consumer");
    }
    if (mlir::isa<IE::MaxPoolOp>(maybeNCETask)) {
        return matchFailed(rewriter, origOp, "IE.MaxPool does not support fp16 input and quantized output");
    }

    const auto quantType = origOp.output().getType();
    const bool isPerChannel =
            quantType.cast<vpux::NDTypeInterface>().getElementType().isa<mlir::quant::UniformQuantizedPerAxisType>();
    if (mlir::isa<IE::AddOp, IE::AvgPoolOp>(maybeNCETask) && isPerChannel) {
        return matchFailed(rewriter, origOp, "IE.AvgPool and IE.Add do not support per-channel quantized output");
    }

    // NCE tasks with float input and quant output support LeakyReLU only per-tensor quantize output.
    // One would expect that with ops ran sequential: BIAS->SCALE->PRELU, we could easily support prelu and per axis
    // quant params. But actually in HW, depending on the sign of the FP BIAS result, you either execute SCALE or PRELU.
    // So for the negative values we'd have to combine the prelu alpha parameter and the requant scale into the per
    // tensor param for prelu scale. This explains why we can't have prelu with per axis quant in fp mode
    if (!isMixPrecisionSupported(maybeNCETask, _arch, !isPerChannel, _log)) {
        return matchFailed(rewriter, origOp, "Producer {0} is not supported", maybeNCETask->getName());
    }

    auto* newNCETask = rewriter.clone(*maybeNCETask);
    newNCETask->getResult(0).setType(quantType);
    rewriter.replaceOp(origOp, newNCETask->getResult(0));
    rewriter.eraseOp(maybeNCETask);

    return mlir::success();
}

void ConvertToMixedPrecisionPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FloatOutConvRewriter>(&ctx, arch, _log);
    patterns.add<FloatOutGroupConvRewriter>(&ctx, arch, _log);
    // TODO: #67754
    // patterns.add<FloatOutMaxPoolRewriter>(&ctx, arch, _log);
    patterns.add<FloatOutAddRewriter>(&ctx, arch, _log);

    const std::set<VPU::ArchKind> incompatibleTargets = {
            VPU::ArchKind::VPUX30XX,
    };

    if (_convertFloatInQuantOut && incompatibleTargets.count(arch) == 0) {
        // Max pooling is omitted intentionally.
        // When we do floating point maxpool the activation datatype appears into the PPE.
        // However, the PPE has only conversion functions from float32, not float16.
        patterns.add<FloatInConvRewriter>(&ctx, arch, _log);
        patterns.add<FloatInGroupConvRewriter>(&ctx, arch, _log);
        patterns.add<FloatInAvgPoolRewriter>(&ctx, arch, _log);
        patterns.add<FloatInAddRewriter>(&ctx, arch, _log);
    }

    if (incompatibleTargets.count(arch) == 0) {
        patterns.add<FloatOutAvgPoolRewriter>(&ctx, arch, _log);
        patterns.add<QuantizeWithNCERewriter>(&ctx, arch, _log);
    }

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertToMixedPrecision
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertToMixedPrecision(const bool allowFP16ToU8, Logger log) {
    return std::make_unique<ConvertToMixedPrecisionPass>(allowFP16ToU8, log);
}
