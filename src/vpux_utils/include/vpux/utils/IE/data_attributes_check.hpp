//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once
#include <ie_layouts.h>

namespace vpux {

/**
 * @brief Checks the contents of the two data attribute structures and throws an error if these are not compatible.
 * @details This function is mainly used for verifying that the precision and shape of the data allow running
 * inferences for the corresponding compiled model in a manner which produces the correct results.
 * The check is performed by comparing the metadata objects using the precision and in-memory shape values. The
 * "in-memory shape" refers to the shape of the data after the layout value is applied.
 * In order for this to succeed, the size of the shape lists should also match.
 *
 * As an example, consider two metadata ("ie::TensorDesc") objects:
 * - "U8" precision, "NHWC" layout, "[1, 3, 224, 224]" dimensions, "[1, 224, 224, 3]" "BlockingDescriptor"
 * dimensions
 * - "U8" precision, "NCHW" layout, "[1, 224, 224, 3]" dimensions, "[1, 224, 224, 3]" "BlockingDescriptor"
 * dimensions
 *
 * where "BlockingDescriptor" dimensions denotes the shape of the data after considering the layout value. The
 * checks performed in this case shall pass since the precision and in-memory shape ("BlockingDescriptor"
 * dimensions) values match.
 * @param userTensorDesc Usually corresponds to the data attributes associated with the "inference request"
 * structure.
 * @param deviceTensorDesc The data attributes expected by the device.
 * @throw The "IE_THROW" function is used for throwing an error in case of incompatibility.
 */
void checkDataAttributesMatch(const InferenceEngine::TensorDesc& userTensorDesc,
                              const InferenceEngine::TensorDesc& deviceTensorDesc);

}  // namespace vpux
