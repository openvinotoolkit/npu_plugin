//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
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
        : idx(idx), left(xmin), right(xmax), top(ymin), bottom(ymax), prob(prob) {}
};

std::vector<BoundingBox> parseYoloOutput(
    const InferenceEngine::Blob::Ptr& blob, size_t imgWidth, size_t imgHeight, float confThresh, bool isTiny);

std::vector<BoundingBox> parseYoloV3Output(const InferenceEngine::BlobMap& blobs, size_t imgWidth, size_t imgHeight,
    int classes, int coords, int num, const std::vector<float>& anchors, float confThresh,
    InferenceEngine::Layout layout);

std::vector<BoundingBox> parseSSDOutput(
    const InferenceEngine::Blob::Ptr& blob, size_t imgWidth, size_t imgHeight, float confThresh);

void printDetectionBBoxOutputs(std::vector<BoundingBox>& actualOutput, std::ostringstream& outputStream,
    const std::vector<std::string>& labels = {});

float boxIntersectionOverUnion(const Box& a, const Box& b);
}  // namespace utils
