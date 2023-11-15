//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "semantic_segmentation_helpers.hpp"

#include "vpux/utils/IE/blob.hpp"

namespace ie = InferenceEngine;

void utils::argMax_channels(const InferenceEngine::MemoryBlob::Ptr blob, std::vector<uint8_t>& resultArgmax) {
    const auto blobFP32 = vpux::toFP32(vpux::toDefLayout(blob));
    const auto blobMem = blobFP32->rmap();
    const auto blobPtr = blobMem.as<const float*>();

    size_t C = blobFP32->getTensorDesc().getDims()[1];
    size_t H = blobFP32->getTensorDesc().getDims()[2];
    size_t W = blobFP32->getTensorDesc().getDims()[3];

    for (size_t h = 0; h < H; h++) {
        for (size_t w = 0; w < W; w++) {
            float argMax = 0.0f;
            uint8_t clsIdx = std::numeric_limits<uint8_t>::max();
            for (size_t c = 0; c < C; c++) {
                if (argMax < blobPtr[c * H * W + h * W + w]) {
                    argMax = blobPtr[c * H * W + h * W + w];
                    clsIdx = c;
                }
            }
            resultArgmax.push_back(clsIdx);
        }
    }
}
