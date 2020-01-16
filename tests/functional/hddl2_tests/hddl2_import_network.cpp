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
#include "helper_precompiled_resnet.h"

namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
//      class HDDL2_ImportNetwork_Tests Declaration
//------------------------------------------------------------------------------
class HDDL2_ImportNetwork_Tests : public HDDL2_Core_API_Tests {
public:
    modelBlobInfo blobInfo = PrecompiledResNet_Helper::resnet;
};

//------------------------------------------------------------------------------
//      class HDDL2_ImportNetwork_Tests Initiation - create
//------------------------------------------------------------------------------
TEST_F(HDDL2_ImportNetwork_Tests, CanFindPlugin) { ASSERT_NO_THROW(ie.ImportNetwork(blobInfo.graphPath, pluginName)); }

TEST_F(HDDL2_ImportNetwork_Tests, CanCreateExecutableNetwork) {
    std::map<std::string, std::string> config = {};

    ASSERT_NO_THROW(executableNetwork = ie.ImportNetwork(blobInfo.graphPath, pluginName));
}

TEST_F(HDDL2_ImportNetwork_Tests, CanCreateExecutableNetworkWithConfig) {
    std::map<std::string, std::string> config = {};

    ASSERT_NO_THROW(executableNetwork = ie.ImportNetwork(blobInfo.graphPath, pluginName, config));
}

TEST_F(HDDL2_ImportNetwork_Tests, CanCreateInferRequest) {
    ASSERT_NO_THROW(executableNetwork = ie.ImportNetwork(blobInfo.graphPath, pluginName));

    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());
}
