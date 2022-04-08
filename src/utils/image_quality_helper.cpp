//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "image_quality_helper.hpp"

#include "vpux/utils/IE/blob.hpp"

#include <cmath>

namespace ie = InferenceEngine;

float utils::runPSNRMetric(std::vector<std::vector<float>>& actOutput, std::vector<std::vector<float>>& refOutput,
                           const size_t imgHeight, const size_t imgWidth, int scaleBorder, bool normalizedImage) {
    size_t colorScale;
    float imageDiff;
    float sum = 0;

    if (!normalizedImage) {
        colorScale = 255;
    } else {
        colorScale = 1;
    }

    for (size_t iout = 0; iout < actOutput.size(); ++iout) {
        for (size_t h = scaleBorder; h < imgHeight - scaleBorder; h++) {
            for (size_t w = scaleBorder; w < imgWidth - scaleBorder; w++) {
                imageDiff =
                        ((actOutput[iout][h * imgWidth + w] - refOutput[iout][h * imgWidth + w]) / (float)colorScale);

                sum = sum + (imageDiff * imageDiff);
            }
        }
    }

    auto mse = sum / (imgWidth * imgHeight);
    auto psnr = -10 * log10(mse);

    std::cout << "psnr: " << psnr << " Db" << std::endl;

    return psnr;
}

std::vector<std::vector<float>> utils::parseBlobAsFP32(const ie::BlobMap& outputBlob) {
    std::vector<std::vector<float>> results;

    for (auto blob : outputBlob) {
        auto blobFP32 = vpux::toFP32(ie::as<ie::MemoryBlob>(blob.second));
        auto ptr = blobFP32->cbuffer().as<float*>();
        IE_ASSERT(ptr != nullptr);

        const auto size = blobFP32->size();
        std::vector<float> result(size);
        std::copy_n(ptr, size, result.begin());

        results.push_back(result);
    }

    return results;
}
