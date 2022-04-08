//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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

std::vector<BoundingBox> parseYoloV4Output(const InferenceEngine::BlobMap& blobs, size_t imgWidth, size_t imgHeight,
                                           int classes, int coords, int num, const std::vector<float>& anchors,
                                           float confThresh, InferenceEngine::Layout layout);

std::vector<BoundingBox> parseYoloV3V4Output(const InferenceEngine::BlobMap& blobs, size_t imgWidth, size_t imgHeight,
                                             int classes, int coords, int num, const std::vector<float>& anchors,
                                             float confThresh, InferenceEngine::Layout layout,
                                             const std::function<float(const float)>& transformationFunc,
                                             const std::function<int(const size_t, const int)>& anchorFunc);

std::vector<BoundingBox> parseSSDOutput(const InferenceEngine::Blob::Ptr& blob, size_t imgWidth, size_t imgHeight,
                                        float confThresh);

void printDetectionBBoxOutputs(std::vector<BoundingBox>& actualOutput, std::ostringstream& outputStream,
                               const std::vector<std::string>& labels = {});

float boxIntersectionOverUnion(const Box& a, const Box& b);
}  // namespace utils
