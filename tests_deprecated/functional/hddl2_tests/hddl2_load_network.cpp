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

#include "hddl2_load_network.h"
#include <hddl2/hddl2_plugin_config.hpp>

#include <thread>

#include "helper_remote_context.h"

using namespace InferenceEngine;
using namespace InferenceEngine::HDDL2ConfigParams;

TEST_F(LoadNetwork_Tests, CanCreateExecutableNetwork) {
    ASSERT_NO_THROW(auto executableNetwork = ie.LoadNetwork(network, pluginName));
}

TEST_F(LoadNetwork_Tests, CanCreateWithContext) {
    Remote_Context_Helper contextHelper;

    auto contextParams = contextHelper.wrapWorkloadIdToMap(contextHelper.getWorkloadId());
    RemoteContext::Ptr remoteContext = ie.CreateContext(pluginName, contextParams);

    ASSERT_NO_THROW(auto executableNetwork = ie.LoadNetwork(network, remoteContext, {}));
}

TEST_F(LoadNetwork_Tests, CannotCreateWithNullContext) {
    RemoteContext::Ptr remoteContext = nullptr;

    ASSERT_ANY_THROW(auto executableNetwork = ie.LoadNetwork(network, remoteContext, {}));
}

TEST_F(LoadNetwork_Tests, CanSetCSRAMSize) {
    // TODO We need some way to distinguish KMB/TBH cases
    const uint64_t csram_size = 6 * 1024 * 1024;
    std::map<std::string, std::string> _config = {{VPU_HDDL2_CONFIG_KEY(CSRAM_SIZE), std::to_string(csram_size)}};

    ASSERT_NO_THROW(auto executableNetwork = ie.LoadNetwork(network, pluginName, _config));
}

TEST_F(LoadNetwork_Tests, CannotSetBadConfig) {
    std::map<std::string, std::string> _config = {{"BAD_KEY", "BAD_VALUE"}};

    ASSERT_ANY_THROW(auto executableNetwork = ie.LoadNetwork(network, pluginName, _config));
}

//------------------------------------------------------------------------------
InferenceEngine::ExecutableNetwork::Ptr ExecutableNetwork_Tests::_cacheExecNetwork = nullptr;

void ExecutableNetwork_Tests::SetUp() {
    if (_cacheExecNetwork == nullptr) {
        ASSERT_NO_THROW(_cacheExecNetwork = std::make_shared<InferenceEngine::ExecutableNetwork>(ie.LoadNetwork(network, pluginName)));
    }
    executableNetworkPtr = _cacheExecNetwork;
}

TEST_F(ExecutableNetwork_Tests, CanCreateInferRequest) {
    ASSERT_NO_THROW(inferRequest = executableNetworkPtr->CreateInferRequest());
}
