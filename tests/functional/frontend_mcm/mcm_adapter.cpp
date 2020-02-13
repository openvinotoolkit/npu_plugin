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

#include <gtest/gtest.h>
#include "models/model_pooling.h"

#include "mcm_adapter.hpp"

using namespace vpu;

class MCMAdapter_Tests : public ::testing::Test {
public:
    InferenceEngine::CNNNetwork network;
    MCMConfig mcmConfig;

protected:
    void SetUp() override;
};

void MCMAdapter_Tests::SetUp() {
    ModelPooling_Helper modelPoolingHelper;
    network = modelPoolingHelper.network;
}

using MCMAdapter_compileNetwork = MCMAdapter_Tests;
TEST_F(MCMAdapter_compileNetwork, canCompile) {
    std::vector<char> blobFile;

    ASSERT_NO_THROW(MCMAdapter::compileNetwork(network, mcmConfig, blobFile));
    ASSERT_GT(blobFile.size(), 0);
}
