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
#include <ie_api.h>

#include <sstream>
#include <test_model_path.hpp>

namespace ModelLoader_Helper {

static bool LoadModel(const std::string& modelName, InferenceEngine::CNNNetwork& network) {
    std::ostringstream modelFile;
    modelFile << "/" << modelName << ".xml";

    std::ostringstream weightsFile;
    weightsFile << "/" << modelName << ".bin";

    std::string modelFilePath = ModelsPath() + modelFile.str();
    std::string weightsFilePath = ModelsPath() + weightsFile.str();

    try {
        InferenceEngine::Core ie;
        network = ie.ReadNetwork(modelFilePath, weightsFilePath);
    } catch (const std::exception& ex) {
        std::cout << "Failed to read network! Error: " << ex.what() << std::endl;
        return false;
    }
    return true;
}

}
