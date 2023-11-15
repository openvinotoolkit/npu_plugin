//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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

}  // namespace ModelLoader_Helper
