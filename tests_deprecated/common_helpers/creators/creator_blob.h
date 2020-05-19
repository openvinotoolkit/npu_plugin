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

#include "ie_blob.h"

namespace Blob_Creator {
    InferenceEngine::Blob::Ptr createBlob(const InferenceEngine::SizeVector& dims,
            const InferenceEngine::Layout& layout = InferenceEngine::Layout::NHWC);

    InferenceEngine::Blob::Ptr createBlob(const InferenceEngine::SizeVector &dims, const InferenceEngine::Layout &layout) {
        if (dims.size() != 4) {
            THROW_IE_EXCEPTION << "Dims size must be 4 for CreateBlob method";
        }
        InferenceEngine::TensorDesc desc = {InferenceEngine::Precision::U8, dims, layout};

        auto blob = InferenceEngine::make_shared_blob<uint8_t>(desc);
        blob->allocate();

        return blob;
    }
}
