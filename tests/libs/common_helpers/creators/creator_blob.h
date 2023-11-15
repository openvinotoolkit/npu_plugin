//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "ie_blob.h"

namespace Blob_Creator {
InferenceEngine::Blob::Ptr createBlob(const InferenceEngine::SizeVector& dims,
                                      const InferenceEngine::Layout& layout = InferenceEngine::Layout::NHWC);

inline InferenceEngine::Blob::Ptr createBlob(const InferenceEngine::SizeVector& dims,
                                             const InferenceEngine::Layout& layout) {
    if (dims.size() != 4) {
        IE_THROW() << "Dims size must be 4 for CreateBlob method";
    }
    InferenceEngine::TensorDesc desc = {InferenceEngine::Precision::U8, dims, layout};

    auto blob = InferenceEngine::make_shared_blob<uint8_t>(desc);
    blob->allocate();

    return blob;
}
}  // namespace Blob_Creator
