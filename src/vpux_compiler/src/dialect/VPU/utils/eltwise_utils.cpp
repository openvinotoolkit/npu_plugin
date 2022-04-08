//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/utils/eltwise_utils.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/numeric.hpp"

using namespace vpux;
using namespace VPU;

bool vpux::VPU::isNCEEltwiseSupported(VPU::ArchKind arch, mlir::ValueRange operands, mlir::Value result,
                                      bool allowDifferentScales, bool allowDifferentZp, LogCb logCb) {
    VPUX_THROW_UNLESS(operands.size() == 2, "Expected 2 operands, got '{0}'", operands.size());
    const auto input1Type = operands[0].getType().cast<vpux::NDTypeInterface>();
    const auto input2Type = operands[1].getType().cast<vpux::NDTypeInterface>();
    const auto outputType = result.getType().cast<vpux::NDTypeInterface>();

    if (input1Type.getRank() != 4 || input2Type.getRank() != 4 || outputType.getRank() != 4) {
        logCb(formatv("Only 4D tensors are supported"));
        return false;
    }

    if (input1Type.getShape() != input2Type.getShape()) {
        logCb(formatv("Broadcasting is not supported"));
        return false;
    }

    // Output type can differ from input type. In case of quantization this can be different quant scale value.
    // Input types can also differ when both of them are quantized. E.g. scale value for Eltwise Multiply
    const auto input1ElemType = input1Type.getElementType();
    const auto input2ElemType = input2Type.getElementType();

    if (!input1ElemType.isa<mlir::quant::QuantizedType>() && !input2ElemType.isa<mlir::quant::QuantizedType>()) {
        if (input1ElemType != input2ElemType) {
            return false;
        }
    } else if (input1ElemType.isa<mlir::quant::UniformQuantizedType>() &&
               input2ElemType.isa<mlir::quant::UniformQuantizedType>()) {
        auto qInput1 = input1ElemType.cast<mlir::quant::UniformQuantizedType>();
        auto qInput2 = input2ElemType.cast<mlir::quant::UniformQuantizedType>();

        if (qInput1.getExpressedType() != qInput2.getExpressedType() ||
            qInput1.getStorageType() != qInput2.getStorageType() || qInput1.isSigned() != qInput2.isSigned()) {
            logCb(formatv("Mismatch in inputs quantization parameters"));
            return false;
        }

        if (!allowDifferentZp && qInput1.getZeroPoint() != qInput2.getZeroPoint()) {
            logCb(formatv("Mismatch in inputs zero points"));
            return false;
        }

        if (!allowDifferentScales &&
            !isFloatEqual(static_cast<float>(qInput1.getScale()), static_cast<float>(qInput2.getScale()))) {
            logCb(formatv("Mismatch in inputs quantization scales"));
            return false;
        }
    } else {
        logCb(formatv("Unsupported inputs element types"));
        return false;
    }

    if (!NCEInvariant::isInputActTypeSupported(arch, input1Type, NCEEltwiseOp::getInputChannelAlignmentImpl(input1Type),
                                               false) ||
        !NCEInvariant::isInputActTypeSupported(arch, input2Type, NCEEltwiseOp::getInputChannelAlignmentImpl(input2Type),
                                               false) ||
        !NCEInvariant::isOutputActTypeSupported(outputType, NCEEltwiseOp::getOutputChannelAlignmentImpl(outputType))) {
        logCb(formatv("Misaligned tensor shape"));
        return false;
    }

    return NCEInvariant::checkLayouts(operands, result, arch, 2, logCb);
}
