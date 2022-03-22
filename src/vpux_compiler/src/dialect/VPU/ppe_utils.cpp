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

#include "vpux/compiler/dialect/VPU/ppe_utils.hpp"

#include <numeric>

#include "vpux/compiler/utils/custom_pwl_table.hpp"
#include "vpux/compiler/utils/quantization.hpp"

namespace vpux {
namespace VPU {

// This operation based on input and output of Eltwise op will prepare final quantization scale value

llvm::Optional<double> calculateQuantScaleVectorForEltwise(mlir::ShapedType input1ShapedType,
                                                           mlir::ShapedType input2ShapedType,
                                                           mlir::ShapedType outputShapedType, VPU::ArchKind arch,
                                                           bool isMultiplyOp) {
    const auto input1ElementType = input1ShapedType.getElementType();
    const auto input2ElementType = input2ShapedType.getElementType();
    const auto outputElementType = outputShapedType.getElementType();

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

    if (isMultiplyOp) {
        const auto scaleInput2 = extractScalesAndZeroPoints(input2ElementType).first.front();
        VPUX_THROW_UNLESS(scaleInput2 != 0, "Invalid input scale value '0'");
        ppeScale = scaleInput1 * scaleInput2 / scaleOutput;
    } else {  // Add, Subtract, And
        ppeScale = scaleInput1 / scaleOutput;
    }

    return {ppeScale};
}

llvm::Optional<double> calculateQuantScaleVectorForAvgPool(mlir::ShapedType inputShapedType,
                                                           mlir::ShapedType outputShapedType,
                                                           ArrayRef<int64_t> filter_size, VPU::ArchKind arch) {
    const auto inputElementType = inputShapedType.getElementType();
    const auto outputElementType = outputShapedType.getElementType();

    // In case of fully not quantized operation return
    if (!inputElementType.isa<mlir::quant::QuantizedType>() && !outputElementType.isa<mlir::quant::QuantizedType>()) {
        return ::llvm::None;
    }

    VPUX_THROW_WHEN(inputElementType.isa<mlir::quant::UniformQuantizedPerAxisType>() ||
                            outputElementType.isa<mlir::quant::UniformQuantizedPerAxisType>(),
                    "Only per-tensor quantization is supported");

    // floats in the compute pipeline are represented as S16.16 values
    // In order to convert from I32 to S16.16 and back, we need to multiply/divide by 1<<16
    // Depends on target hardware
    const double fp16_scale = (VPU::ArchKind::MTL == arch) ? (1.0) : (1.0 / 65536);

    auto scaleInput = fp16_scale;
    auto scaleOutput = fp16_scale;

    if (inputElementType.isa<mlir::quant::QuantizedType>())
        scaleInput = extractScalesAndZeroPoints(inputElementType).first.front();

    if (outputElementType.isa<mlir::quant::QuantizedType>())
        scaleOutput = extractScalesAndZeroPoints(outputElementType).first.front();

    VPUX_THROW_UNLESS(scaleInput != 0, "Invalid input scale value '0'");
    VPUX_THROW_UNLESS(scaleOutput != 0, "Invalid output scale value '0'");

    double ppeScale = 1.0;

    int64_t divisor = std::accumulate(filter_size.begin(), filter_size.end(), 1ll, std::multiplies<int64_t>());
    ppeScale = scaleInput / scaleOutput / static_cast<double>(divisor);

    return {ppeScale};
}

VPU::PPETaskAttr getPPEAttr(VPU::PostOpParams postOpParams, mlir::MLIRContext* ctx) {
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

VPU::PPETaskAttr getPPETaskAttrFromPostOpsParams(mlir::Value opInput, mlir::Value opOutput, vpux::IE::PostOp postOpAttr,
                                                 mlir::Location loc, mlir::MLIRContext* ctx, VPU::ArchKind arch) {
    const auto inElemType = opInput.getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto outElemType = opOutput.getType().cast<vpux::NDTypeInterface>().getElementType();

    const auto postOpParams = VPU::parsePostOp(postOpAttr, inElemType, outElemType, arch, loc);

    VPU::PPETaskAttr ppeTaskAttr;
    if (postOpParams.hasValue()) {
        ppeTaskAttr = VPU::getPPEAttr(postOpParams.getValue(), ctx);
    }

    return ppeTaskAttr;
}

VPU::PPETaskAttr getNCEEltwisePPETaskAttr(mlir::Value opInput1, mlir::Value opInput2, mlir::Value opOutput,
                                          vpux::IE::PostOp postOpAttr, mlir::Location loc, VPU::EltwiseType opType,
                                          mlir::MLIRContext* ctx, VPU::ArchKind arch) {
    int64_t clampLow = std::numeric_limits<int32_t>::min();
    int64_t clampHigh = std::numeric_limits<int32_t>::max();
    int64_t LreluMult = 1;
    int64_t LreluShift = 0;

    auto origOutType = opOutput.getType().cast<vpux::NDTypeInterface>();
    auto outElemType = origOutType.getElementType();
    const auto inElemType = opInput1.getType().cast<vpux::NDTypeInterface>().getElementType();
    if (auto outElemQType = outElemType.dyn_cast<mlir::quant::QuantizedType>()) {
        const auto zps = extractScalesAndZeroPoints(outElemType).second;

        clampLow = outElemQType.getStorageTypeMin() - zps.front();
        clampHigh = outElemQType.getStorageTypeMax() - zps.front();
    }

    const auto postOpParams = parsePostOp(postOpAttr, inElemType, outElemType, arch, loc);
    if (postOpParams.hasValue()) {
        clampLow = postOpParams->clampLow;
        clampHigh = postOpParams->clampHigh;
        LreluMult = postOpParams->LreluMult;
        LreluShift = postOpParams->LreluShift;
    }

    VPU::PPEMode ppeType = VPU::getPPEMode(opType);
    auto ppeAttr = getPPETaskAttr(ctx, ppeType);

    // Since Eltwise operation doesn't have weights table it requires final quantization scaling
    // to be part of output tensor description. Scale vector will be placed in PPE block and
    // later used during NCE task serialization
    auto quantScale = VPU::calculateQuantScaleVectorForEltwise(
            opInput1.getType().cast<mlir::ShapedType>(), opInput2.getType().cast<mlir::ShapedType>(),
            opOutput.getType().cast<mlir::ShapedType>(), arch, opType == VPU::EltwiseType::MULTIPLY);
    if (quantScale.hasValue()) {
        const auto scale = quantScale.getValue();

        const auto scaleApproximation = QuantizationApproximation(arch, scale);

        ppeAttr = getPPETaskAttr(ctx, ppeType, clampLow, clampHigh, LreluMult, LreluShift,
                                 ArrayRef<int64_t>{scaleApproximation.mult()},
                                 ArrayRef<int64_t>{scaleApproximation.shift()}, scaleApproximation.postShift());
    } else {
        ppeAttr = getPPETaskAttr(ctx, ppeType, clampLow, clampHigh, LreluMult, LreluShift);
    }

    return ppeAttr;
}

VPU::PostOpParams getPwlPostOpParams(const mlir::Type inElemType, const mlir::Type outElemType, VPU::PPEMode ppeType) {
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

PostOpParams getCustomPwlPostOpParams(IE::PostOp postOp, mlir::Type outElemType) {
    auto pwlTable = findCustomPWLTable(postOp.name().getValue(), outElemType);

    VPUX_THROW_UNLESS(pwlTable.hasValue(), "Custom PWL Table was not found for {0} {1}", postOp.name().getValue(),
                      outElemType);

    auto pwlTableRange = pwlTable.getValue().range;

    VPUX_THROW_UNLESS(!pwlTableRange.empty(), "Custom PWL Table range is empty for {0} {1}", postOp.name().getValue(),
                      outElemType);

    const int64_t clampLow = pwlTableRange[0];
    const int64_t clampHigh = pwlTableRange[pwlTableRange.size() - 1];
    const int64_t LreluMult = 1;
    const int64_t LreluShift = 0;
    const int64_t postShift = pwlTable.getValue().postShift;

    // Dummy values for mult & shift, as the actual values will be computed in the weights table
    SmallVector<int64_t> quantMult = {1};
    SmallVector<int64_t> quantShift = {0};

    return PostOpParams{VPU::PPEMode::FLEXARB,
                        clampLow,
                        clampHigh,
                        LreluMult,
                        LreluShift,
                        QuantizationParams{quantMult, quantShift, postShift}};
}

llvm::Optional<VPU::PostOpParams> parsePostOp(IE::PostOp postOp, const mlir::Type inElemType,
                                              const mlir::Type outElemType, VPU::ArchKind arch, mlir::Location loc) {
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
        // PWL case
        if (arch != VPU::ArchKind::MTL && outElemQType != nullptr) {
            return getCustomPwlPostOpParams(postOp, outElemType);
        }

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
        int64_t preluMult = 1;
        int64_t preluShift = 0;
        if (isDoubleEqual(alpha, 0.0)) {
            preluMult = 0;
        } else if (!isDoubleEqual(alpha, 1.0)) {
            const auto alphaApproximation = PReLUApproximation(arch, alpha);
            preluMult = alphaApproximation.mult();
            preluShift = alphaApproximation.shift();
        }
        return PostOpParams{VPU::PPEMode::LPRELU, static_cast<int64_t>(clampLow), clampHigh, preluMult, preluShift};
    } else if (postOp.name().getValue() == IE::SigmoidOp::getOperationName()) {
        return VPU::getPwlPostOpParams(inElemType, outElemType, VPU::PPEMode::SIGMOID);
    } else if (postOp.name().getValue() == IE::TanhOp::getOperationName()) {
        return VPU::getPwlPostOpParams(inElemType, outElemType, VPU::PPEMode::TANH);
    }

    VPUX_THROW("Unsupported PostOp '{0}'", postOp.name());
}

}  // namespace VPU
}  // namespace vpux
