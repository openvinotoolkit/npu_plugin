//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include <ie_api.h>
#include <ie_layouts.h>
#include <ie_algorithm.hpp>

//------------------------------------------------------------------------------
//      class TensorDescription_Helper
//------------------------------------------------------------------------------
class TensorDescription_Helper {
public:
    TensorDescription_Helper();

    InferenceEngine::TensorDesc tensorDesc;
    size_t tensorSize;

protected:
    // ResNet configuration
    const InferenceEngine::Precision _precision = InferenceEngine::Precision::U8;
    const InferenceEngine::SizeVector _sizeVector = {1, 3, 224, 224};
    const InferenceEngine::Layout _layout = InferenceEngine::Layout::NCHW;
};

//------------------------------------------------------------------------------
//      class TensorDescription_Helper Implementation
//------------------------------------------------------------------------------
inline TensorDescription_Helper::TensorDescription_Helper() {
    tensorDesc = InferenceEngine::TensorDesc(_precision, _sizeVector, _layout);
    tensorSize = InferenceEngine::details::product(tensorDesc.getDims().begin(),
                                                   tensorDesc.getDims().end());
}
