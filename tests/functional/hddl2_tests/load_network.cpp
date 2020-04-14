//
// Copyright 2019-2020 Intel Corporation.
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

#include "load_network.h"

#include <Inference.h>
#include <helper_remote_context.h>

#include "models/model_loader.h"

using namespace InferenceEngine;

// [Track number: S#30141]
TEST_F(LoadNetwork_Tests, DISABLED_CanFindPlugin) { ASSERT_NO_THROW(ie.LoadNetwork(network, pluginName)); }

// [Track number: S#30141]
TEST_F(LoadNetwork_Tests, DISABLED_CanCreateExecutable) {
    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(network, pluginName));
}

// [Track number: S#30141]
TEST_F(LoadNetwork_Tests, DISABLED_CanCreateWithContext) {
    Remote_Context_Helper contextHelper;

    auto contextParams = contextHelper.wrapWorkloadIdToMap(contextHelper.getWorkloadId());
    RemoteContext::Ptr remoteContext = ie.CreateContext(pluginName, contextParams);

    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(network, remoteContext, {}));
}

// TODO FAIL - IE side problem
TEST_F(LoadNetwork_Tests, DISABLED_CannotCreateWithNullContext) {
    RemoteContext::Ptr remoteContext = nullptr;

    ASSERT_ANY_THROW(executableNetwork = ie.LoadNetwork(network, remoteContext, {}));
}

// [Track number: S#30141]
TEST_F(LoadNetwork_Tests, DISABLED_CanCreateInferRequestAfterLoadNetwork) {
    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(network, pluginName));
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());
}

// TODO FAIL - SIGABRT on googlenet in mcm adapter
TEST_F(LoadNetwork_Tests, DISABLED_CanCreateInferRequestAfterLoadNetwork_GoogleNet) {
    const std::string _modelName = "googlenet/bvlc_googlenet_fp16";
    ModelLoader_Helper::LoadModel(_modelName, network);

    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(network, pluginName));
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());
}
