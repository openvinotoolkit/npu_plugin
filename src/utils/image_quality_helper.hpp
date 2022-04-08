//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include <ie_blob.h>

#include <sstream>

namespace utils {

std::vector<std::vector<float>> parseBlobAsFP32(const InferenceEngine::BlobMap& outputBlob);

float runPSNRMetric(std::vector<std::vector<float>>& actOutput, std::vector<std::vector<float>>& refOutput,
                    const size_t imgHeight, const size_t imgWidth, int scaleBorder, bool normalizedImage);
}  // namespace utils
