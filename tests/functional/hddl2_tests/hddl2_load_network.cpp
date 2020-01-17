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

#include "gtest/gtest.h"
#include "hddl2_core_api.h"
#include "hddl2_helpers/helper_model_loader.h"
#include "helper_remote_context.h"

using namespace InferenceEngine;

using ModelLoader_Helper::LoadModel;

//------------------------------------------------------------------------------
//      class HDDL2_LoadNetwork_Tests Declaration
//------------------------------------------------------------------------------
class HDDL2_LoadNetwork_Tests : public HDDL2_Core_API_Tests {
public:
    const std::string modelName = "googlenet/bvlc_googlenet_fp16";
};

//------------------------------------------------------------------------------
//      class HDDL2_LoadNetwork_Tests Initiations
//------------------------------------------------------------------------------
TEST_F(HDDL2_LoadNetwork_Tests, CanFindPlugin) {
    LoadModel(modelName, network);

    ASSERT_NO_THROW(ie.LoadNetwork(network, pluginName));
}

TEST_F(HDDL2_LoadNetwork_Tests, CanCreateExecutable) {
    LoadModel(modelName, network);

    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(network, pluginName));
}

// TODO Create LoadNetwork with context implementation required
TEST_F(HDDL2_LoadNetwork_Tests, DISABLED_CanCreateWithContext) {
    Remote_Context_Helper contextHelper;

    auto contextParams = contextHelper.wrapWorkloadIdToMap(contextHelper.getWorkloadId());
    RemoteContext::Ptr remoteContext = ie.CreateContext(pluginName, contextParams);
    LoadModel(modelName, network);

    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(network, remoteContext, {}));
}

// TODO FAIL - IE side problem
TEST_F(HDDL2_LoadNetwork_Tests, DISABLED_CannotCreateWithNullContext) {
    RemoteContext::Ptr remoteContext = nullptr;

    ASSERT_ANY_THROW(executableNetwork = ie.LoadNetwork(network, remoteContext, {}));
}

TEST_F(HDDL2_LoadNetwork_Tests, CanCreateInferRequestAfterLoadNetwork) {
    LoadModel(modelName, network);

    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(network, pluginName));
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());
}
