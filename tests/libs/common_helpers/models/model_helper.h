//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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
    if (_network.getName().empty()) IE_THROW() << "Network not loaded";
    return _network;
}