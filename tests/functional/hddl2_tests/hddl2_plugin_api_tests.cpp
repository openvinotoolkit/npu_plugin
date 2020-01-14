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

#include <Inference.h>
#include <cpp/ie_cnn_net_reader.h>
#include <gtest/gtest.h>

#include <fstream>
#include <ie_core.hpp>
#include <test_model_path.hpp>
#include <vector>

//------------------------------------------------------------------------------
//      class HDDL2_Plugin_API Declaration
//------------------------------------------------------------------------------
class HDDL2_Plugin_API : public ::testing::Test {
public:
    std::string device_name = "HDDL2";
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork network;
    InferenceEngine::ExecutableNetwork executableNetwork;
    InferenceEngine::InferRequest inferRequest;

    void LoadModel();

private:
    std::string _modelName = "googlenet/bvlc_googlenet_fp16";
    IE_SUPPRESS_DEPRECATED_START
    InferenceEngine::CNNNetReader _netReader;
    IE_SUPPRESS_DEPRECATED_END
};

struct modelBlobsInfo {
    std::string _graphPath, _inputPath, _outputPath;
};

const static std::vector<modelBlobsInfo> pathToPreCompiledGraph = {
    {
        ._graphPath = "/KMB_models/BLOBS/resnet/resnet.blob",
        ._inputPath = "/KMB_models/BLOBS/resnet/input.dat",
        ._outputPath = "/KMB_models/BLOBS/resnet/output.dat",
    },
};

class InferWithPath : public HDDL2_Plugin_API, public testing::WithParamInterface<modelBlobsInfo> {};

//------------------------------------------------------------------------------
//      Implementation of class HDDL2_Plugin_API
//------------------------------------------------------------------------------
void HDDL2_Plugin_API::LoadModel() {
    std::ostringstream modelFile;
    modelFile << "/" << _modelName << ".xml";

    std::ostringstream weightsFile;
    weightsFile << "/" << _modelName << ".bin";

    std::string modelFilePath = ModelsPath() + modelFile.str();
    std::string weightsFilePath = ModelsPath() + weightsFile.str();

    ASSERT_NO_THROW(_netReader.ReadNetwork(modelFilePath));
    ASSERT_TRUE(_netReader.isParseSuccess());
    ASSERT_NO_THROW(_netReader.ReadWeights(weightsFilePath));

    network = _netReader.getNetwork();
}

TEST_F(HDDL2_Plugin_API, getVpusmmDriver) {
    bool isVPUSMMDriverFound = false;
    std::ifstream modulesLoaded("/proc/modules");
    std::string line;
    while (std::getline(modulesLoaded, line)) {
        if (line.find("vpusmm_driver") != std::string::npos) {
            isVPUSMMDriverFound = true;
            std::cout << " [INFO] - Driver found: " << line << std::endl;
            break;
        }
    }
    ASSERT_TRUE(isVPUSMMDriverFound);
}

TEST_F(HDDL2_Plugin_API, getAvailableDevices) {
    std::vector<HddlUnite::Device> devices;
    HddlStatusCode code = getAvailableDevices(devices);
    std::cout << " [INFO] - Devices found: " << devices.size() << std::endl;
    ASSERT_EQ(code, HddlStatusCode::HDDL_OK);
}

TEST_F(HDDL2_Plugin_API, CanFindPlugin) {
    LoadModel();

    ASSERT_NO_THROW(ie.LoadNetwork(network, device_name));
}

TEST_F(HDDL2_Plugin_API, CanCreateExecutableNetworkLoadNetwork) {
    LoadModel();

    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(network, device_name));
}

TEST_F(HDDL2_Plugin_API, DISABLED_CanCreateInferRequestAfterLoadNetwork) {
    LoadModel();

    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(network, device_name));
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());
}

TEST_F(HDDL2_Plugin_API, DISABLED_CanCallInfer) {
    LoadModel();
    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(network, device_name));
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    ASSERT_NO_THROW(inferRequest.Infer());
}

TEST_P(InferWithPath, CanCreateExecutableNetworkImportMethod) {
    modelBlobsInfo blobsInfo = GetParam();
    std::string graphSuffix = blobsInfo._graphPath;
    std::string modelFilePath = ModelsPath() + graphSuffix;

    std::map<std::string, std::string> config = {};
    ASSERT_NO_THROW(executableNetwork = ie.ImportNetwork(modelFilePath, device_name, config));
}

TEST_P(InferWithPath, CanCreateInferRequestAfterImportNetwork) {
    modelBlobsInfo blobsInfo = GetParam();
    std::string graphSuffix = blobsInfo._graphPath;
    std::string modelFilePath = ModelsPath() + graphSuffix;

    ASSERT_NO_THROW(executableNetwork = ie.ImportNetwork(modelFilePath, device_name));
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());
}

TEST_P(InferWithPath, SyncInferenceTest) {
    modelBlobsInfo blobsInfo = GetParam();
    std::string graphSuffix = blobsInfo._graphPath;
    std::string modelFilePath = ModelsPath() + graphSuffix;

    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = ie.ImportNetwork(modelFilePath, device_name));

    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = importedNetwork.CreateInferRequest());

    ASSERT_NO_THROW(inferRequest.Infer());

    std::string outputName = importedNetwork.GetOutputsInfo().begin()->first;
    InferenceEngine::Blob::Ptr outputBlob;
    ASSERT_NO_THROW(outputBlob = inferRequest.GetBlob(outputName));
}

INSTANTIATE_TEST_CASE_P(inferenceWithParameters, InferWithPath, ::testing::ValuesIn(pathToPreCompiledGraph));
