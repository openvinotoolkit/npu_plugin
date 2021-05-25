//
// Copyright 2019 Intel Corporation.
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

#include <cpp/ie_cnn_network.h>
#include <ie_core.hpp>

#include <sstream>

namespace ModelLoader_Helper {

inline std::string getTestModelsBasePath() {
    if (const auto envVar = std::getenv("MODELS_PATH")) {
        return envVar;
    }

#ifdef MODELS_PATH
    return MODELS_PATH;
#else
    return {};
#endif
}

inline std::string getTestModelsPath() {
    return getTestModelsBasePath() + "/src/models";
}

inline InferenceEngine::CNNNetwork LoadModel(const std::string& modelName) {
    std::ostringstream modelFile;
    modelFile << "/" << modelName << ".xml";

    std::ostringstream weightsFile;
    weightsFile << "/" << modelName << ".bin";

    std::string modelFilePath = getTestModelsPath() + modelFile.str();
    std::string weightsFilePath = getTestModelsPath() + weightsFile.str();

    InferenceEngine::Core ie;
    return ie.ReadNetwork(modelFilePath, weightsFilePath);
}

}
