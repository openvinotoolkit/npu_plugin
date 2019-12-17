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

#include <ie_core.hpp>

#include "hddl2_plugin_api_test_cases.h"

TEST_F(HDDL2_Plugin_API, CanFindPlugin) {
    LoadModel();

    ASSERT_NO_THROW(core.LoadNetwork(network, device_name));
}

TEST_F(HDDL2_Plugin_API, CanCreateExecutableNetworkLoadNetwork) {
    LoadModel();

    ASSERT_NO_THROW(executableNetwork = core.LoadNetwork(network, device_name));
}

TEST_F(HDDL2_Plugin_API, CanCreateExecutableNetworkImportMethod) {
    std::string blob_name = "custom_network_name";

    std::map<std::string, std::string> config = {};
    ASSERT_NO_THROW(executableNetwork = core.ImportNetwork(blob_name, device_name, config));
}

TEST_F(HDDL2_Plugin_API, CanCreateInferRequestAfterLoadNetwork) {
    LoadModel();

    ASSERT_NO_THROW(executableNetwork = core.LoadNetwork(network, device_name));
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());
}

TEST_F(HDDL2_Plugin_API, CanCreateInferRequestAfterImportNetwork) {
    std::string blob_name = "custom_network_name";

    ASSERT_NO_THROW(executableNetwork = core.ImportNetwork(blob_name, device_name));
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());
}

TEST_F(HDDL2_Plugin_API, CanCallInfer) {
    LoadModel();
    ASSERT_NO_THROW(executableNetwork = core.LoadNetwork(network, device_name));
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    ASSERT_NO_THROW(inferRequest.Infer());
}
