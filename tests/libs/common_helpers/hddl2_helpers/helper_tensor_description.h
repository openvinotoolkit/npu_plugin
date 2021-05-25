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
