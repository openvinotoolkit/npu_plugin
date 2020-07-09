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

#include "ie_core.hpp"
#include "model_loader.h"

class ModelHelper {
public:
    using Ptr = std::shared_ptr<ModelHelper>;

    explicit ModelHelper(const bool u8Input = true) : _u8Input(u8Input) {}
    InferenceEngine::CNNNetwork getNetwork() const;

protected:
    void loadModel();

    InferenceEngine::CNNNetwork _network;
    std::string _modelRelatedPath;
    bool _u8Input;
};

//------------------------------------------------------------------------------
inline void ModelHelper::loadModel() {
    _network = ModelLoader_Helper::LoadModel(_modelRelatedPath);

    if (_u8Input) {
        // For VPUX device input should be U8 and network layout NHWC
        InferenceEngine::InputsDataMap inputInfo(_network.getInputsInfo());
        auto inputInfoItem = *inputInfo.begin();
        inputInfoItem.second->setPrecision(InferenceEngine::Precision::U8);
    }
}

inline InferenceEngine::CNNNetwork ModelHelper::getNetwork() const {
    if (_network.getName().empty()) THROW_IE_EXCEPTION << "Network not loaded";
    return _network;
}