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

#include <thread>

#include "helper_remote_context.h"

using namespace InferenceEngine;

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

//------------------------------------------------------------------------------
InferenceEngine::ExecutableNetwork::Ptr ExecutableNetwork_Tests::_cacheExecNetwork = nullptr;

void ExecutableNetwork_Tests::SetUp() {
    if (_cacheExecNetwork == nullptr) {
        _cacheExecNetwork = std::make_shared<InferenceEngine::ExecutableNetwork>(ie.LoadNetwork(network, pluginName));
    }
    executableNetworkPtr = _cacheExecNetwork;
}

TEST_F(ExecutableNetwork_Tests, CanCreateInferRequest) {
    ASSERT_NO_THROW(inferRequest = executableNetworkPtr->CreateInferRequest());
}
