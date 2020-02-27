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

#include "ie_compound_blob.h"

namespace NV12Blob_Creator {
    InferenceEngine::NV12Blob::Ptr createBlob(const std::size_t &width, const std::size_t &height);
    InferenceEngine::NV12Blob::Ptr createBlob(const InferenceEngine::TensorDesc& tensorDesc);
    InferenceEngine::NV12Blob::Ptr createFromMemory(const std::size_t &width, const std::size_t &height, uint8_t *data);
    void descriptorsFromFrameSize(const size_t &width, const size_t &height,
                                  InferenceEngine::TensorDesc &uvDesc, InferenceEngine::TensorDesc &yDesc);

    inline InferenceEngine::NV12Blob::Ptr
    createBlob(const std::size_t &width, const std::size_t &height) {
        InferenceEngine::TensorDesc uvDesc;
        InferenceEngine::TensorDesc yDesc;
        NV12Blob_Creator::descriptorsFromFrameSize(width, height, uvDesc, yDesc);

        auto yPlane = InferenceEngine::make_shared_blob<uint8_t>(yDesc);
        auto uvPlane = InferenceEngine::make_shared_blob<uint8_t>(uvDesc);

        return InferenceEngine::make_shared_blob<InferenceEngine::NV12Blob>(yPlane, uvPlane);
    }

    inline InferenceEngine::NV12Blob::Ptr
    createFromMemory(const std::size_t &width, const std::size_t &height, uint8_t *data) {
        InferenceEngine::TensorDesc uvDesc;
        InferenceEngine::TensorDesc yDesc;
        NV12Blob_Creator::descriptorsFromFrameSize(width, height, uvDesc, yDesc);

        auto yPlane = InferenceEngine::make_shared_blob<uint8_t>(yDesc, data);
        auto uvPlane = InferenceEngine::make_shared_blob<uint8_t>(uvDesc, data + height * width);

        return InferenceEngine::make_shared_blob<InferenceEngine::NV12Blob>(yPlane, uvPlane);
    }

    inline void descriptorsFromFrameSize(const size_t &width, const size_t &height,
                                         InferenceEngine::TensorDesc &uvDesc, InferenceEngine::TensorDesc &yDesc) {
        uvDesc = {InferenceEngine::Precision::U8, {1, 2, height / 2, width / 2}, InferenceEngine::Layout::NHWC};
        yDesc = {InferenceEngine::Precision::U8, {1, 1, height, width}, InferenceEngine::Layout::NHWC};
    }

    inline InferenceEngine::NV12Blob::Ptr createBlob(const InferenceEngine::TensorDesc& tensorDesc) {
        if (tensorDesc.getLayout() != InferenceEngine::NCHW) {
            THROW_IE_EXCEPTION << "Only NCHW Layout supported in nv12 blob creator!";
        }
        if (tensorDesc.getPrecision() != InferenceEngine::Precision::U8) {
            THROW_IE_EXCEPTION << "Only U8 Precision supported in nv12 blob creator!";
        }
        const InferenceEngine::SizeVector& dims = tensorDesc.getDims();
        const uint N_index = 0;
        const uint C_index = 1;
        const uint H_index = 2;
        const uint W_index = 3;

        if (dims[N_index] != 1 || dims[C_index] != 3) {
            THROW_IE_EXCEPTION << "Only batch 1 and channel == 3 supported for nv12 creator!";
        }

        return createBlob(dims[W_index], dims[H_index]);
    }
}
