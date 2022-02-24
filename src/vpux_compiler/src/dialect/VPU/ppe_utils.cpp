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

        const auto mult = getQuantMultFromScale(scale);
        const auto shifts = getQuantShiftAndPostShiftFromScale(scale);

        const auto shift = shifts.first;
        const auto post_shift = shifts.second;

        ppeAttr = getPPETaskAttr(ctx, ppeType, clampLow, clampHigh, LreluMult, LreluShift, ArrayRef<int64_t>{mult},
                                 ArrayRef<int64_t>{shift}, post_shift);
    } else {
        ppeAttr = getPPETaskAttr(ctx, ppeType, clampLow, clampHigh, LreluMult, LreluShift);
    }

    return ppeAttr;
}

}  // namespace VPU
}  // namespace vpux
