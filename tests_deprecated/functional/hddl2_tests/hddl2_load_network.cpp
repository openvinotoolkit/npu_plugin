//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "hddl2_load_network.h"
#include <vpux_private_config.hpp>
#include "hddl2_helpers/helper_remote_context.h"

using namespace InferenceEngine;
using namespace InferenceEngine::VPUXConfigParams;

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
    std::map<std::string, std::string> _config = {{VPUX_CONFIG_KEY(CSRAM_SIZE), std::to_string(csram_size)}};

    ASSERT_NO_THROW(auto executableNetwork = ie.LoadNetwork(network, pluginName, _config));
}

TEST_F(LoadNetwork_Tests, CannotSetBadConfig) {
    std::map<std::string, std::string> _config = {{"BAD_KEY", "BAD_VALUE"}};

    ASSERT_ANY_THROW(auto executableNetwork = ie.LoadNetwork(network, pluginName, _config));
}

//------------------------------------------------------------------------------
TEST_F(ExecutableNetwork_Tests, CanCreateInferRequest) {
    ASSERT_NO_THROW(inferRequest = executableNetworkPtr->CreateInferRequest());
}
