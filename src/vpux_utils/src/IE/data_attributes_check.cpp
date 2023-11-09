//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/IE/data_attributes_check.hpp"

namespace ie = InferenceEngine;
namespace {

/**
 * @brief A simple function building a string representation for a vector of "size_t" typed elements.
 * @param vector The vector which shall be represented as a string.
 * @returns The string representation of the given argument in the form "[x[0], x[1], ...]"
 */
std::string vectorToStringRepresentation(const ie::SizeVector& vector) {
    std::ostringstream stringRepresentation;
    stringRepresentation << '[';

    for (auto iterator = vector.begin(); iterator != vector.end(); iterator++) {
        if (iterator != vector.begin()) {
            stringRepresentation << ", ";
        }
        stringRepresentation << *iterator;
    }
    stringRepresentation << ']';
    return stringRepresentation.str();
}

}  // namespace

void vpux::checkDataAttributesMatch(const ie::TensorDesc& userTensorDesc, const ie::TensorDesc& deviceTensorDesc) {
    if (userTensorDesc.getPrecision() == deviceTensorDesc.getPrecision() &&
        userTensorDesc.getBlockingDesc().getBlockDims() == deviceTensorDesc.getBlockingDesc().getBlockDims()) {
        return;
    }

    // Build the error message
    std::ostringstream errorMessage;
    errorMessage << "The data attributes corresponding to the inference request structure are not matching the ones "
                    "expected by the device:"
                 << std::endl;

    const ie::SizeVector& userDimensions = userTensorDesc.getDims();
    const ie::SizeVector& deviceDimensions = deviceTensorDesc.getDims();
    const std::string& stringUserDimensions = vectorToStringRepresentation(userDimensions);
    const std::string& stringDeviceDimensions = vectorToStringRepresentation(deviceDimensions);

    errorMessage << "Inference request data attributes: "
                 << "Precision " << userTensorDesc.getPrecision().name() << ", "
                 << "Layout " << userTensorDesc.getLayout() << ", "
                 << "Dimensions " << stringUserDimensions << std::endl;
    errorMessage << "The data attributes expected by the device: "
                 << "Precision " << deviceTensorDesc.getPrecision().name() << ", "
                 << "Layout " << deviceTensorDesc.getLayout() << ", "
                 << "Dimensions " << stringDeviceDimensions;
    IE_THROW() << errorMessage.str();
}
