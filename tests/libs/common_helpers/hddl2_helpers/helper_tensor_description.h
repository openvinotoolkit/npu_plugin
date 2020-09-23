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
