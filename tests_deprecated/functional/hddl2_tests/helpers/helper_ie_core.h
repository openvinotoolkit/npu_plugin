//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "ie_core.hpp"
#include <yolo_helpers.hpp>

class IE_Core_Helper {
public:
    InferenceEngine::Core ie;
    IE_Core_Helper();
    const std::string pluginName;

    // TODO Separate load image helpers to separate helper both for KMB and HDDL2
    // Will return RGB image
    static InferenceEngine::Blob::Ptr loadCatImage(const InferenceEngine::Layout& targetImageLayout = InferenceEngine::Layout::NHWC);
    // TODO Duplicate of kmb_test_base.cpp function
    static InferenceEngine::Blob::Ptr loadImage(const std::string &imageName, const size_t width, const size_t height,
                                                const InferenceEngine::Layout targetImageLayout, const bool isBGR);
    static void checkBBoxOutputs(std::vector<utils::BoundingBox> &actualOutput, std::vector<utils::BoundingBox> &refOutput,
        const int imgWidth, const int imgHeight, const float boxTolerance, const float probTolerance);
};
