//
// Copyright 2020 Intel Corporation.
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

#pragma once

#include <ie_blob.h>

#include <sstream>

namespace utils {
struct Box final {
    float x, y, w, h;
};

struct BoundingBox final {
    int idx;
    float left, right, top, bottom;
    float prob;
    BoundingBox(int idx, float xmin, float ymin, float xmax, float ymax, float prob)
            : idx(idx), left(xmin), right(xmax), top(ymin), bottom(ymax), prob(prob) {
    }
};

std::vector<BoundingBox> parseYoloOutput(const InferenceEngine::Blob::Ptr& blob, size_t imgWidth, size_t imgHeight,
                                         float confThresh, bool isTiny);

std::vector<BoundingBox> parseYoloV3Output(const InferenceEngine::BlobMap& blobs, size_t imgWidth, size_t imgHeight,
                                           int classes, int coords, int num, const std::vector<float>& anchors,
                                           float confThresh, InferenceEngine::Layout layout);

std::vector<BoundingBox> parseSSDOutput(const InferenceEngine::Blob::Ptr& blob, size_t imgWidth, size_t imgHeight,
                                        float confThresh);

void printDetectionBBoxOutputs(std::vector<BoundingBox>& actualOutput, std::ostringstream& outputStream,
                               const std::vector<std::string>& labels = {});

float boxIntersectionOverUnion(const Box& a, const Box& b);
}  // namespace utils
