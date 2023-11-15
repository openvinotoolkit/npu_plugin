//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/propagate_quantize_dequantize_utils.hpp"

void vpux::IE::propagateElementTypeDown(IE::LayerDataInfo<mlir::Type>& info) {
    const auto inputElemType = info.getInput(0);

    if (inputElemType.isa<mlir::quant::UniformQuantizedPerAxisType>()) {
        // Do not propagate element type down in per channel case.
        return;
    }

    for (size_t outputInd = 0; outputInd < info.getNumOutputs(); ++outputInd) {
        info.setOutput(outputInd, inputElemType);
    }
}

void vpux::IE::propagateElementTypeUp(IE::LayerDataInfo<mlir::Type>& info) {
    const auto outputElemType = info.getOutput(0);

    if (outputElemType.isa<mlir::quant::UniformQuantizedPerAxisType>()) {
        // Do not propagate element type up in per channel case.
        return;
    }

    for (size_t inputInd = 0; inputInd < info.getNumInputs(); ++inputInd) {
        info.setInput(inputInd, outputElemType);
    }
}
