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

#include "executable_network_factory.h"

#ifndef __WIN32__
    #define BLOB_SAVING_IMPLEMENTED
#endif

#ifdef BLOB_SAVING_IMPLEMENTED
// TODO Copy of kmb_tests_base, remove after moving to vpuxFuncTests
const std::string DUMP_PATH = []() -> std::string {
  if (const auto var = std::getenv("IE_KMB_TESTS_DUMP_PATH")) {
      return var;
  }

  return "/tmp/";
}();

const bool GENERATE_BLOBS = []() -> bool {
  if (const auto var = std::getenv("IE_KMB_TESTS_GENERATE_BLOBS")) {
      try {
          const auto intVal = std::stoi(var);
          return intVal;
      } catch (...) {
          throw std::invalid_argument("Only 0 and 1 values are supported");
      }
  }

  return false;
}();
#endif

std::map<std::string, CachedModel> ExecutableNetworkFactory::cachedModels;

InferenceEngine::ExecutableNetwork ExecutableNetworkFactory::createExecutableNetwork(
        const std::string& modelRelativePath, const ModelParams& modelParams) {

    InferenceEngine::Core ie;
    std::istringstream graphBlobStream = getGraphBlob(modelRelativePath, modelParams);
    const auto network = ie.ImportNetwork(graphBlobStream, "VPUX");
    return network;
}

InferenceEngine::CNNNetwork ExecutableNetworkFactory::createCNNNetwork(const std::string& modelRelativePath,
                                                                       const ModelParams& modelParams) {
    auto network = ModelLoader_Helper::LoadModel(modelRelativePath);

    const auto inputsInfo = network.getInputsInfo();
    for (auto& input : inputsInfo) {
        auto inputData = input.second;
        inputData->setLayout(modelParams.inputLayout);
        inputData->setPrecision(modelParams.inputPrecision);
    }
    return network;
}

#ifdef BLOB_SAVING_IMPLEMENTED
static std::string extractModelNameFromPath(const std::string& modelPath) {
    const auto pos = modelPath.rfind('/');
    if(pos == std::string::npos) {
        IE_THROW() << "[ExecutableNetworkFactory] Failed to find model name in path: " << modelPath;
    }
    return modelPath.substr(pos);
}
#endif

std::istringstream ExecutableNetworkFactory::getGraphBlob(const std::string& modelRelativePath,
                                                          const ModelParams& modelParams) {
    // Try to search in cache
    try {
        const auto cachedModel = cachedModels.at(modelRelativePath);
        if (cachedModel.modelParams == modelParams) {
            std::cout << "[ExecutableNetworkFactory] Cached model found and will be used." << std::endl;
            return std::istringstream(cachedModel.graphBlob);
        }
    } catch (...) {
        std::cout << "[ExecutableNetworkFactory] Cached model was not found." << std::endl;
    }

    // Try to search graph blob on filesystem
#ifdef BLOB_SAVING_IMPLEMENTED
    const std::string temporaryFileName = DUMP_PATH + extractModelNameFromPath(modelRelativePath);

    // If caching (IE_KMB_TESTS_GENERATE_BLOBS) is on, can try to find saved one
    bool saveAndUseBlobsFromFilesystem;

    const ModelParams defaultParams;
    if (modelParams == defaultParams) {
        saveAndUseBlobsFromFilesystem = GENERATE_BLOBS;
    } else {
        std::cout << "[ExecutableNetworkFactory] Not default params caching not supported!" << std::endl;
        saveAndUseBlobsFromFilesystem = false;
    }

    if (saveAndUseBlobsFromFilesystem) {
        const std::ifstream graphBlobStream(temporaryFileName, std::ios::binary);
        std::stringstream stringstream;
        stringstream << graphBlobStream.rdbuf();
        if (graphBlobStream) {
            std::istringstream graphBlobStreamContentStream(stringstream.str());
            std::cout << "[ExecutableNetworkFactory] Blob " << temporaryFileName << " loaded from filesystem"
                      << std::endl;
            return graphBlobStreamContentStream;
        } else {
            std::cout << "[ExecutableNetworkFactory] Can not open blob file " << temporaryFileName
                      << ". It was not created!" << std::endl;
        }
    }
#endif

    // If nothing worked out, well, need to compile it
    InferenceEngine::Core ie;
    std::cout << "[ExecutableNetworkFactory] Compilation of model " << modelRelativePath << " will be completed." << std::endl;
    const auto network = createCNNNetwork(modelRelativePath, modelParams);
    auto executableNetwork = ie.LoadNetwork(network, "VPUX");

    // Save blob
#ifdef BLOB_SAVING_IMPLEMENTED
    if (saveAndUseBlobsFromFilesystem) {
        executableNetwork.Export(temporaryFileName);
        std::cout << "[ExecutableNetworkFactory] Save temporary blob: " << temporaryFileName << std::endl;
    }
#endif
    std::stringstream stringstream;
    executableNetwork.Export(stringstream);

    // Save to cache
    CachedModel cachedModel;
    cachedModel.modelParams = modelParams;
    cachedModel.graphBlob = stringstream.str();
    cachedModels[modelRelativePath] = cachedModel;
    std::cout << "Model: " << modelRelativePath << " was cached!" << std::endl;

    std::istringstream graphBlobStreamContentStream(stringstream.str());
    return graphBlobStreamContentStream;
}
