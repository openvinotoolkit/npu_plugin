//
// Copyright 2019-2020 Intel Corporation.
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
