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

#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPU/pwl_utils.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/utils/core/numeric.hpp"

#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/enums.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;
using namespace VPU;

namespace {

struct QuantizationParams {
    SmallVector<int64_t> quantMult;
    SmallVector<int64_t> quantShift;
    int64_t postShift;
};

struct PostOpParams {
    VPU::PPEMode layerType;
    int64_t clampLow;
    int64_t clampHigh;
    int64_t LreluMult;
    int64_t LreluShift;
    Optional<QuantizationParams> quantParams;

    PostOpParams(VPU::PPEMode layerType, int64_t clampLow, int64_t clampHigh, int64_t LreluMult, int64_t LreluShift)
            : layerType(layerType),
              clampLow(clampLow),
              clampHigh(clampHigh),
              LreluMult(LreluMult),
              LreluShift(LreluShift) {
    }

    PostOpParams(VPU::PPEMode layerType, int64_t clampLow, int64_t clampHigh, int64_t LreluMult, int64_t LreluShift,
                 const QuantizationParams& quantParams)
            : layerType(layerType),
              clampLow(clampLow),
              clampHigh(clampHigh),
              LreluMult(LreluMult),
              LreluShift(LreluShift),
              quantParams(quantParams) {
    }
};

int64_t getPwlClamp(const mlir::Type inElemType, const mlir::Type outElemType, const VPU::PPEMode ppeType,
                    const bool getMin) {
    constexpr int64_t CLAMP_MIN = -4096;
    constexpr int64_t CLAMP_MAX = 4095;

    // Input type defines the compute type
    if (inElemType.template isa<mlir::FloatType>()) {
        return getMin ? CLAMP_MIN : CLAMP_MAX;
    }

    const auto quantReqs = VPU::getPwlQuantReqs(ppeType);
    const auto rMin = quantReqs.input.rMin;
    const auto rMax = quantReqs.input.rMax;

    if (const auto qElemType = outElemType.template dyn_cast<mlir::quant::UniformQuantizedType>()) {
        const auto scale = qElemType.getScale();
        const auto actualMin = std::max(CLAMP_MIN, static_cast<int64_t>(std::ceil(rMin / scale)));
        const auto actualMax = std::min(CLAMP_MAX, static_cast<int64_t>(std::floor(rMax / scale)));
        return getMin ? actualMin : actualMax;
    }
    VPUX_THROW("Unexpected output element type: {0}", outElemType);
}

int64_t getPwlPostShift(const VPU::PPEMode ppeType) {
    const auto quantReqs = VPU::getPwlQuantReqs(ppeType);
    return quantReqs.input.postShift;
}

PostOpParams getPwlPostOpParams(const mlir::Type inElemType, const mlir::Type outElemType, VPU::PPEMode ppeType) {
    const int64_t clampLow = getPwlClamp(inElemType, outElemType, ppeType, true);
    const int64_t clampHigh = getPwlClamp(inElemType, outElemType, ppeType, false);
    const int64_t LreluMult = 1;
    const int64_t LreluShift = 0;
    const int64_t postShift = getPwlPostShift(ppeType);

    // Dummy values for mult & shift, as the actual values will be computed in the weights table
    SmallVector<int64_t> quantMult = {1};
    SmallVector<int64_t> quantShift = {0};

    return PostOpParams{ppeType,   clampLow,   clampHigh,
                        LreluMult, LreluShift, QuantizationParams{quantMult, quantShift, postShift}};
}

std::pair<int64_t, int64_t> getClampValuesForQuantizedOps(mlir::quant::QuantizedType outElemQType,
                                                          mlir::Type outElemType) {
    const auto zps = extractScalesAndZeroPoints(outElemType).second;
    auto clampLow = outElemQType.getStorageTypeMin() - zps.front();
    auto clampHigh = outElemQType.getStorageTypeMax() - zps.front();
    return {clampLow, clampHigh};
}

mlir::Optional<PostOpParams> parsePostOp(IE::PostOp postOp, const mlir::Type inElemType, const mlir::Type outElemType,
                                         VPU::ArchKind arch, mlir::Location loc) {
    if (postOp == nullptr) {
        return mlir::None;
    }

    auto outElemQType = outElemType.dyn_cast<mlir::quant::QuantizedType>();
    int64_t clampLowQuantized = 0;
    int64_t clampHighQuantized = 0;
    if (outElemQType != nullptr) {
        clampLowQuantized = getClampValuesForQuantizedOps(outElemQType, outElemType).first;
        clampHighQuantized = getClampValuesForQuantizedOps(outElemQType, outElemType).second;
    }

    if (postOp.name().getValue() == IE::ReLUOp::getOperationName()) {
        VPUX_THROW_UNLESS(postOp.attrs().empty(), "'{0}' PostOp should not have any attributes", postOp.name());

        int64_t clampLow = 0;
        int64_t clampHigh = (outElemQType != nullptr) ? clampHighQuantized : std::numeric_limits<int32_t>::max();
        const int64_t LreluMult = 1;
        const int64_t LreluShift = 0;

        return PostOpParams{VPU::PPEMode::LRELU, clampLow, clampHigh, LreluMult, LreluShift};
    } else if (postOp.name().getValue() == IE::ClampOp::getOperationName()) {
        IE::ClampOp::Adaptor clamp(None, postOp.attrs());
        VPUX_THROW_UNLESS(clamp.verify(loc).succeeded(), "Wrong attributes '{0}' for '{1}' PostOp", postOp.attrs(),
                          postOp.name());

        int64_t clampLow =
                (outElemQType != nullptr) ? clampLowQuantized : vpux::toFixedPoint(clamp.min().getValueAsDouble());
        int64_t clampHigh =
                (outElemQType != nullptr) ? clampHighQuantized : vpux::toFixedPoint(clamp.max().getValueAsDouble());
        const int64_t LreluMult = 1;
        const int64_t LreluShift = 0;

        return PostOpParams{VPU::PPEMode::NOOP, clampLow, clampHigh, LreluMult, LreluShift};
    } else if (postOp.name().getValue() == IE::LeakyReluOp::getOperationName()) {
        IE::LeakyReluOp::Adaptor leakyRelu(None, postOp.attrs());
        VPUX_THROW_UNLESS(leakyRelu.verify(loc).succeeded(), "Wrong attributes '{0}' for '{1}' PostOp", postOp.attrs(),
                          postOp.name());

        const auto alpha = leakyRelu.negative_slope().getValueAsDouble();
        int32_t clampLow = static_cast<int32_t>(std::numeric_limits<int32_t>::min() / alpha);
        if (outElemQType != nullptr) {
            clampLow = (arch == VPU::ArchKind::MTL) ? static_cast<int32_t>(clampLowQuantized)
                                                    : static_cast<int32_t>(clampLowQuantized / alpha);
        }

        int64_t clampHigh = (outElemQType != nullptr) ? clampHighQuantized : std::numeric_limits<int32_t>::max();
        uint32_t leakyAccuracyBits = arch == VPU::ArchKind::MTL ? 31 : 7;
        uint32_t LreluMult = 1;
        uint32_t LreluShift = 0;
        if (isDoubleEqual(alpha, 0.0)) {
            LreluMult = 0;
        } else if (!isDoubleEqual(alpha, 1.0)) {
            vpux::VPU::NCESparsity::computeQuantMultShift(alpha, LreluShift, LreluMult, leakyAccuracyBits);
        }
        return PostOpParams{VPU::PPEMode::LPRELU, static_cast<int64_t>(clampLow), clampHigh,
                            static_cast<int64_t>(LreluMult), static_cast<int64_t>(LreluShift)};
    } else if (postOp.name().getValue() == IE::SigmoidOp::getOperationName()) {
        return getPwlPostOpParams(inElemType, outElemType, VPU::PPEMode::SIGMOID);
    } else if (postOp.name().getValue() == IE::TanhOp::getOperationName()) {
        return getPwlPostOpParams(inElemType, outElemType, VPU::PPEMode::TANH);
    }

    VPUX_THROW("Unsupported PostOp '{0}'", postOp.name());
}

VPU::PPETaskAttr getPPEAttr(PostOpParams postOpParams, mlir::MLIRContext* ctx) {
    if (postOpParams.quantParams.hasValue()) {
        const auto quantParams = postOpParams.quantParams.getValue();
        return getPPETaskAttr(ctx, postOpParams.layerType, postOpParams.clampLow, postOpParams.clampHigh,
                              postOpParams.LreluMult, postOpParams.LreluShift, quantParams.quantMult,
                              quantParams.quantShift, quantParams.postShift);
    } else {
        return getPPETaskAttr(ctx, postOpParams.layerType, postOpParams.clampLow, postOpParams.clampHigh,
                              postOpParams.LreluMult, postOpParams.LreluShift);
    }
}

llvm::Optional<double> calculateQuantScaleVectorForEltwise(VPU::NCEEltwiseOp origOp, VPU::ArchKind arch) {
    const auto input1 = origOp.input1();
    const auto input2 = origOp.input2();
    const auto output = origOp.output();

    const auto input1ElementType = input1.getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto input2ElementType = input2.getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto outputElementType = output.getType().cast<vpux::NDTypeInterface>().getElementType();

    // In case of fully not quantized operation return
    if (!input1ElementType.isa<mlir::quant::QuantizedType>() && !input2ElementType.isa<mlir::quant::QuantizedType>() &&
        !outputElementType.isa<mlir::quant::QuantizedType>()) {
        return ::llvm::None;
    }

    VPUX_THROW_WHEN(input1ElementType.isa<mlir::quant::UniformQuantizedPerAxisType>() ||
                            input2ElementType.isa<mlir::quant::UniformQuantizedPerAxisType>() ||
                            outputElementType.isa<mlir::quant::UniformQuantizedPerAxisType>(),
                    "Only per-tensor quantization is supported");

    double scaleInput1 = 0;
    double scaleOutput = 0;

    // floats in the compute pipeline are represented as S16.16 values
    // In order to convert from I32 to S16.16 and back, we need to multiply/divide by 1<<16
    // Depends on target hardware
    const double fp16_scale = (VPU::ArchKind::MTL == arch) ? (1.0) : (1.0 / 65536);

    if (!input1ElementType.isa<mlir::quant::QuantizedType>() && !input2ElementType.isa<mlir::quant::QuantizedType>()) {
        scaleOutput = extractScalesAndZeroPoints(outputElementType).first.front();
        scaleInput1 = fp16_scale;
    } else if (!outputElementType.isa<mlir::quant::QuantizedType>()) {
        scaleInput1 = extractScalesAndZeroPoints(input1ElementType).first.front();
        scaleOutput = fp16_scale;
    } else {
        scaleInput1 = extractScalesAndZeroPoints(input1ElementType).first.front();
        scaleOutput = extractScalesAndZeroPoints(outputElementType).first.front();
    }

    VPUX_THROW_UNLESS(scaleInput1 != 0, "Invalid input scale value '0'");
    VPUX_THROW_UNLESS(scaleOutput != 0, "Invalid output scale value '0'");

    double ppeScale = 1.0;

    if (origOp.op_type() == VPU::EltwiseType::MULTIPLY) {
        const auto scaleInput2 = extractScalesAndZeroPoints(input2ElementType).first.front();
        VPUX_THROW_UNLESS(scaleInput2 != 0, "Invalid input scale value '0'");
        ppeScale = scaleInput1 * scaleInput2 / scaleOutput;
    } else {  // Add, Subtract, And
        ppeScale = scaleInput1 / scaleOutput;
    }

    return {ppeScale};
}

//
// ConvertPostOpsToPPE
//

class ConvertPostOpsToPPEPass final : public ConvertPostOpsToPPEBase<ConvertPostOpsToPPEPass> {
public:
    explicit ConvertPostOpsToPPEPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

    template <class ConcreteOp>
    void updatePPETasks(ConcreteOp op, VPU::ArchKind arch);

    void updateNCEEltwisePPETasks(VPU::NCEEltwiseOp op, VPU::ArchKind arch);

private:
    Logger _log;
};

template <class ConcreteOp>
void ConvertPostOpsToPPEPass::updatePPETasks(ConcreteOp op, VPU::ArchKind arch) {
    if (!op.post_op().hasValue()) {
        return;
    }

    const auto inElemType = op.input().getType().template cast<vpux::NDTypeInterface>().getElementType();
    const auto outElemType = op.output().getType().template cast<vpux::NDTypeInterface>().getElementType();
    const auto postOpParams = parsePostOp(op.post_opAttr(), inElemType, outElemType, arch, op.getLoc());

    if (!postOpParams.hasValue()) {
        return;
    }

    const auto ppeTask = getPPEAttr(postOpParams.getValue(), op.getContext());
    op.ppeAttr(ppeTask);
    op.removePost_opAttr();
}

void ConvertPostOpsToPPEPass::updateNCEEltwisePPETasks(VPU::NCEEltwiseOp op, VPU::ArchKind arch) {
    int64_t clampLow = std::numeric_limits<int32_t>::min();
    int64_t clampHigh = std::numeric_limits<int32_t>::max();
    int64_t LreluMult = 1;
    int64_t LreluShift = 0;

    auto origOutType = op.output().getType().cast<vpux::NDTypeInterface>();
    auto outElemType = origOutType.getElementType();
    const auto inElemType = op.input1().getType().cast<vpux::NDTypeInterface>().getElementType();
    if (auto outElemQType = outElemType.dyn_cast<mlir::quant::QuantizedType>()) {
        const auto zps = extractScalesAndZeroPoints(outElemType).second;

        clampLow = outElemQType.getStorageTypeMin() - zps.front();
        clampHigh = outElemQType.getStorageTypeMax() - zps.front();
    }

    auto* ctx = op.getContext();

    auto postOp = op.post_op().hasValue() ? op.post_opAttr() : nullptr;
    const auto postOpParams = parsePostOp(postOp, inElemType, outElemType, arch, op.getLoc());
    if (postOpParams.hasValue()) {
        clampLow = postOpParams->clampLow;
        clampHigh = postOpParams->clampHigh;
        LreluMult = postOpParams->LreluMult;
        LreluShift = postOpParams->LreluShift;
    }

    VPU::PPEMode ppeType = VPU::getPPEMode(op.op_type());
    auto ppeAttr = getPPETaskAttr(ctx, ppeType);

    // Since Eltwise operation doesn't have weights table it requires final quantization scaling
    // to be part of output tensor description. Scale vector will be placed in PPE block and
    // later used during NCE task serialization
    auto quantScale = calculateQuantScaleVectorForEltwise(op, arch);
    if (quantScale.hasValue()) {
        const auto scale = quantScale.getValue();

        const auto mult = getQuantMultFromScale(scale);
        const auto shifts = getQuantShiftAndPostShiftFromScale(scale);

        const auto shift = shifts.first;
        const auto post_shift = shifts.second;

        ppeAttr = getPPETaskAttr(ctx, ppeType, clampLow, clampHigh, LreluMult, LreluShift, ArrayRef<int64_t>{mult},
                                 ArrayRef<int64_t>{shift}, post_shift);
    } else {
        ppeAttr = getPPETaskAttr(ctx, ppeType, clampLow, clampHigh, LreluMult, LreluShift);
    }

    op.ppeAttr(ppeAttr);
    // Can't have both 'post_op' and 'ppe' attributes at the same time
    op.removePost_opAttr();
}

void ConvertPostOpsToPPEPass::safeRunOnFunc() {
    auto func = getFunction();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    const auto callback = [&](mlir::Operation* op) {
        llvm::TypeSwitch<mlir::Operation*, void>(op)
                .Case<VPU::NCEConvolutionOp>([&](VPU::NCEConvolutionOp op) {
                    updatePPETasks<VPU::NCEConvolutionOp>(op, arch);
                })
                .Case<VPU::NCEMaxPoolOp>([&](VPU::NCEMaxPoolOp op) {
                    updatePPETasks<VPU::NCEMaxPoolOp>(op, arch);
                })
                .Case<VPU::NCEDepthConvolutionOp>([&](VPU::NCEDepthConvolutionOp op) {
                    updatePPETasks<VPU::NCEDepthConvolutionOp>(op, arch);
                })
                .Case<VPU::NCEEltwiseOp>([&](VPU::NCEEltwiseOp op) {
                    updateNCEEltwisePPETasks(op, arch);
                });
    };

    func.walk(callback);
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createConvertPostOpsToPPEPass(Logger log) {
    return std::make_unique<ConvertPostOpsToPPEPass>(log);
}
