// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"

void ref_copy_wrap(InferenceEngine::Blob::Ptr src,
                   InferenceEngine::Blob::Ptr dst,
                   const ParamsStruct& params) {
    ASSERT_TRUE(params.empty());
    ref_copy(src, dst);
}

void ref_copy(const InferenceEngine::Blob::Ptr src,
              InferenceEngine::Blob::Ptr dst) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    ASSERT_EQ(src->dims().size(), dst->dims().size());
    uint16_t *srcData = src->buffer();
    uint16_t *dstData = dst->buffer();
    ASSERT_NE(srcData, nullptr);
    ASSERT_NE(dstData, nullptr);
    std::memcpy(dstData, srcData, src->byteSize());
}
