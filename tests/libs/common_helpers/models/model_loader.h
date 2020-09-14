//
// Copyright 2019 Intel Corporation.
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
