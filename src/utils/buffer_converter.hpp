//
// Copyright 2019 Intel Corporation.
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

#include <ie_data.h>

#include <ie_parallel.hpp>
#include <unordered_set>
#include <vector>

namespace vpu {

namespace KmbPlugin {

template <typename T>
void kchw_to_khwc(const T* src, T* dst, const InferenceEngine::TensorDesc& desc) {
    IE_ASSERT(desc.getDims().size() > 3 && desc.getLayout() == InferenceEngine::Layout::NCHW);

    auto& dims = desc.getDims();
    auto W = dims[3];
    auto H = dims[2];
    auto C = dims[1];
    auto K = dims[0];

    InferenceEngine::parallel_for4d(K, W, H, C, [=](int k, int w, int h, int c) {
        auto inInd = w + h * W + c * H * W + k * C * H * W;
        auto outInd = c + w * C + h * W * C + k * H * W * C;

        dst[outInd] = src[inInd];
    });
}

}  // namespace KmbPlugin

}  // namespace vpu
