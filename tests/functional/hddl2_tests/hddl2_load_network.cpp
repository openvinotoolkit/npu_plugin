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
#include <helper_remote_context.h>

#include "hddl2_core_api.h"
#include "models/model_loader.h"
#include "models/model_pooling.h"
#include "ie_core.hpp"

using namespace InferenceEngine;

//------------------------------------------------------------------------------
//      class HDDL2_LoadNetwork_Tests Declaration
//------------------------------------------------------------------------------
class HDDL2_LoadNetwork_Tests : public HDDL2_Core_API_Tests {
public:
    void SetUp() override;
    InferenceEngine::CNNNetwork network;

protected:
    ModelPooling_Helper _modelPoolingHelper;
};

void HDDL2_LoadNetwork_Tests::SetUp() { network = _modelPoolingHelper.network; }

//------------------------------------------------------------------------------
//      class HDDL2_LoadNetwork_Tests Initiations
//------------------------------------------------------------------------------
TEST_F(HDDL2_LoadNetwork_Tests, CanFindPlugin) { ASSERT_NO_THROW(ie.LoadNetwork(network, pluginName)); }

TEST_F(HDDL2_LoadNetwork_Tests, CanCreateExecutable) {
    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(network, pluginName));
}

TEST_F(HDDL2_LoadNetwork_Tests, CanCreateWithContext) {
    Remote_Context_Helper contextHelper;

    auto contextParams = contextHelper.wrapWorkloadIdToMap(contextHelper.getWorkloadId());
    RemoteContext::Ptr remoteContext = ie.CreateContext(pluginName, contextParams);

    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(network, remoteContext, {}));
}

// TODO FAIL - IE side problem
TEST_F(HDDL2_LoadNetwork_Tests, DISABLED_CannotCreateWithNullContext) {
    RemoteContext::Ptr remoteContext = nullptr;

    ASSERT_ANY_THROW(executableNetwork = ie.LoadNetwork(network, remoteContext, {}));
}

TEST_F(HDDL2_LoadNetwork_Tests, CanCreateInferRequestAfterLoadNetwork) {
    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(network, pluginName));
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());
}

// TODO FAIL - SIGABRT on googlenet in mcm adapter
TEST_F(HDDL2_LoadNetwork_Tests, DISABLED_CanCreateInferRequestAfterLoadNetwork_GoogleNet) {
    const std::string _modelName = "googlenet/bvlc_googlenet_fp16";
    ModelLoader_Helper::LoadModel(_modelName, network);

    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(network, pluginName));
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());
}
