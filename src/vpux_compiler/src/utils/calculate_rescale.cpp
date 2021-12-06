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

#include "vpux/compiler/utils/calculate_rescale.hpp"

#include <numeric>

#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/utils/quantization.hpp"

namespace vpux {

// This operation based on input and output of Eltwise op will prepare final quantization scale value

llvm::Optional<double> calculateQuantScaleVectorForEltwise(IERT::LayerOpInterface layerOp, VPU::ArchKind arch) {
    if (layerOp == nullptr || layerOp.getInputs().size() < 2) {
        return ::llvm::None;
    }

    const auto input1 = layerOp.getInputs()[0];
    const auto input2 = layerOp.getInputs()[1];
    const auto output = layerOp.getOutputs()[0];

    const auto input1ElementType = input1.getType().cast<mlir::ShapedType>().getElementType();
    const auto input2ElementType = input2.getType().cast<mlir::ShapedType>().getElementType();
    const auto outputElementType = output.getType().cast<mlir::ShapedType>().getElementType();

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

    if (mlir::isa<IERT::MultiplyOp>(layerOp)) {
        const auto scaleInput2 = extractScalesAndZeroPoints(input2ElementType).first.front();
        VPUX_THROW_UNLESS(scaleInput2 != 0, "Invalid input scale value '0'");
        ppeScale = scaleInput1 * scaleInput2 / scaleOutput;
    } else {  // Add, Subtract, And
        ppeScale = scaleInput1 / scaleOutput;
    }

    return {ppeScale};
}

llvm::Optional<double> calculateQuantScaleVectorForAvgPool(IERT::LayerOpInterface layerOp,
                                                           std::vector<int64_t>& filter_size, VPU::ArchKind arch) {
    if (layerOp == nullptr || layerOp.getInputs().size() < 1) {
        return ::llvm::None;
    }

    const auto input = layerOp.getInputs()[0];
    const auto output = layerOp.getOutputs()[0];

    const auto inputElementType = input.getType().cast<mlir::ShapedType>().getElementType();
    const auto outputElementType = output.getType().cast<mlir::ShapedType>().getElementType();

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

    int64_t divisor = std::accumulate(filter_size.begin(), filter_size.end(), 1l, std::multiplies<int64_t>());
    ppeScale = scaleInput / scaleOutput / static_cast<double>(divisor);

    return {ppeScale};
}

}  // namespace vpux
