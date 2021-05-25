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