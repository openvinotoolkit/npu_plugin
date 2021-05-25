//
// Copyright 2019-2020 Intel Corporation.
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
