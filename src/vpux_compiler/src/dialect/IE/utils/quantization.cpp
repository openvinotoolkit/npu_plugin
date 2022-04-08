//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/utils/quantization.hpp"

using namespace vpux;

llvm::Optional<int64_t> vpux::IE::getFQAxisIndex(IE::FakeQuantizeOp fq) {
    const auto extractAxis = [](mlir::Value input) -> Optional<int64_t> {
        const auto greaterThanOne = [](auto dim) {
            return dim > 1;
        };

        const auto shape = getShape(input);

        const auto axisCount = llvm::count_if(shape, greaterThanOne);
        VPUX_THROW_UNLESS(axisCount <= 1, "FakeQuantize constant input has incorrect shape");

        auto axis = llvm::find_if(shape, greaterThanOne);
        if (axis != shape.end()) {
            return std::distance(shape.begin(), axis);
        }

        return None;
    };

    const auto inputLowAxis = extractAxis(fq.input_low());
    const auto outputLowAxis = extractAxis(fq.output_low());

    if (!inputLowAxis && !outputLowAxis) {
        return {};
    }

    if (inputLowAxis && outputLowAxis) {
        VPUX_THROW_UNLESS(*inputLowAxis == *outputLowAxis, "FakeQuantize constant inputs use different axis");
    }

    return inputLowAxis ? *inputLowAxis : *outputLowAxis;
}

llvm::Optional<int64_t> vpux::IE::getQuantAxisIndex(mlir::Operation* op) {
    llvm::Optional<int64_t> axis = None;
    const auto getPerAxisQType = [](mlir::Value tensor) {
        return tensor.getType()
                .cast<vpux::NDTypeInterface>()
                .getElementType()
                .dyn_cast<mlir::quant::UniformQuantizedPerAxisType>();
    };

    if (auto fqOp = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(op)) {
        axis = IE::getFQAxisIndex(fqOp);
    } else if (mlir::isa<IE::DequantizeOp, IE::QuantizeOp>(op)) {
        if (const auto perAxisQType = getPerAxisQType(op->getOperand(0))) {
            axis = perAxisQType.getQuantizedDimension();
        }
        if (const auto perAxisQType = getPerAxisQType(op->getResult(0))) {
            axis = perAxisQType.getQuantizedDimension();
        }
    }

    return axis;
}
