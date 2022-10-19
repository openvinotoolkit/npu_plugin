//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/VPUIPRegMapped/host_parsing/host_parsed_inference.h"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

//
// KernelParamsOp
//

void vpux::VPUIPRegMapped::KernelParamsOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    // input memref dims
    const auto inputMemrefVal = input();

    const auto inputShape = getShape(inputMemrefVal);
    const auto inOrderInput = DimsOrder::fromValue(inputMemrefVal);
    const auto inputMemShape = inOrderInput.toMemoryOrder(inputShape);

    std::vector<uint8_t> inputDimsVector;

    for (auto& dim : inputMemShape | reversed) {
        auto input_dim = checked_cast<int32_t>(dim);
        ArrayRef<uint8_t> valueAsArray(reinterpret_cast<const uint8_t*>(&input_dim), sizeof(input_dim));
        inputDimsVector.insert(inputDimsVector.end(), valueAsArray.begin(), valueAsArray.end());
    }

    // output memref dims
    const auto outputMemrefVal = output();

    const auto outputShape = getShape(outputMemrefVal);
    const auto inOrderOutput = DimsOrder::fromValue(outputMemrefVal);
    const auto outputMemShape = inOrderOutput.toMemoryOrder(outputShape);

    std::vector<uint8_t> outputDimsVector;

    for (auto& dim : outputMemShape | reversed) {
        auto output_dim = checked_cast<int32_t>(dim);
        ArrayRef<uint8_t> valueAsArray(reinterpret_cast<const uint8_t*>(&output_dim), sizeof(output_dim));
        outputDimsVector.insert(outputDimsVector.end(), valueAsArray.begin(), valueAsArray.end());
    }

    // input memref strides
    std::vector<uint8_t> inputStridesVector;

    const auto inputMemrefStrides = getMemStrides(inputMemrefVal);
    for (auto&& stride : inputMemrefStrides | reversed) {
        ArrayRef<uint8_t> valueAsArray(reinterpret_cast<const uint8_t*>(&stride), sizeof(stride));
        inputStridesVector.insert(inputStridesVector.end(), valueAsArray.begin(), valueAsArray.end());
    }

    // output memref strides
    std::vector<uint8_t> outputStridesVector;

    const auto outputMemrefStrides = getMemStrides(outputMemrefVal);
    for (auto&& stride : outputMemrefStrides | reversed) {
        ArrayRef<uint8_t> valueAsArray(reinterpret_cast<const uint8_t*>(&stride), sizeof(stride));
        outputStridesVector.insert(outputStridesVector.end(), valueAsArray.begin(), valueAsArray.end());
    }

    auto params = kernel_params();

    auto dense_elem_data = params.getValues<uint8_t>();

    auto data_vector = std::vector<uint8_t>(dense_elem_data.begin(), dense_elem_data.end());

    // serialize actual kernel params
    binDataSection.appendData(data_vector.data(), data_vector.size());

    // serialize IO dims/strides
    binDataSection.appendData(inputDimsVector.data(), inputDimsVector.size());
    binDataSection.appendData(outputDimsVector.data(), outputDimsVector.size());
    binDataSection.appendData(inputStridesVector.data(), inputStridesVector.size());
    binDataSection.appendData(outputStridesVector.data(), outputStridesVector.size());
}

size_t vpux::VPUIPRegMapped::KernelParamsOp::getBinarySize() {
    auto params = kernel_params();
    auto dense_elem_data = params.getValues<uint8_t>();
    auto data_vector = std::vector<uint8_t>(dense_elem_data.begin(), dense_elem_data.end());

    const auto inputMemrefVal = input();
    const auto outputMemrefVal = output();

    const auto inputDimsSize = sizeof(int32_t) * getShape(inputMemrefVal).size();
    const auto outputDimsSize = sizeof(int32_t) * getShape(outputMemrefVal).size();

    const auto inputStridesSize = sizeof(int64_t) * getMemStrides(inputMemrefVal).size();
    const auto outputStridesSize = sizeof(int64_t) * getMemStrides(outputMemrefVal).size();

    return data_vector.size() + inputDimsSize + outputDimsSize + inputStridesSize + outputStridesSize;
}

size_t vpux::VPUIPRegMapped::KernelParamsOp::getParamsStructSize() {
    auto params = kernel_params();
    auto dense_elem_data = params.getValues<uint8_t>();
    auto data_vector = std::vector<uint8_t>(dense_elem_data.begin(), dense_elem_data.end());

    return data_vector.size();
}
