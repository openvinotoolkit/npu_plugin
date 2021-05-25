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

#include "ie_blob.h"

namespace Blob_Creator {
    InferenceEngine::Blob::Ptr createBlob(const InferenceEngine::SizeVector& dims,
            const InferenceEngine::Layout& layout = InferenceEngine::Layout::NHWC);

    inline InferenceEngine::Blob::Ptr createBlob(const InferenceEngine::SizeVector &dims, const InferenceEngine::Layout &layout) {
        if (dims.size() != 4) {
            IE_THROW() << "Dims size must be 4 for CreateBlob method";
        }
        InferenceEngine::TensorDesc desc = {InferenceEngine::Precision::U8, dims, layout};

        auto blob = InferenceEngine::make_shared_blob<uint8_t>(desc);
        blob->allocate();

        return blob;
    }
}
