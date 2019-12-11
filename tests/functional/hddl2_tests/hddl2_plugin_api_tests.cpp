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

#include <hddlunite/Inference.h>

#include <fstream>
#include <ie_core.hpp>
#include <vector>

#include "hddl2_plugin_api_test_cases.h"

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
