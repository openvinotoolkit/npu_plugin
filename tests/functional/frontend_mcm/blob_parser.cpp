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

#include <blob_parser.hpp>
#include <fstream>
#include <string>

#include "models/precompiled_resnet.h"

using namespace vpu;

class BlobParser_Tests : public ::testing::Test {
public:
    InferenceEngine::InputsDataMap networkInputs;
    InferenceEngine::OutputsDataMap networkOutputs;

    std::string blobContentString;

protected:
    void SetUp() override;
};

void BlobParser_Tests::SetUp() {
    const std::string graphPath = PrecompiledResNet_Helper::resnet50.graphPath;

    std::ifstream blobFile(graphPath, std::ios::binary);
    if (!blobFile.is_open()) {
        THROW_IE_EXCEPTION << "[ERROR] *Could not open file: " << graphPath;
    }

    std::ostringstream blobContentStream;
    blobContentStream << blobFile.rdbuf();
    blobContentString = blobContentStream.str();
}

TEST_F(BlobParser_Tests, CanParseBlob) {
    ASSERT_NO_THROW(MCMAdapter::getNetworkInputs(blobContentString.c_str(), networkInputs));
    ASSERT_NO_THROW(MCMAdapter::getNetworkOutputs(blobContentString.c_str(), networkOutputs));
}

TEST_F(BlobParser_Tests, CanGetInputsOutputsDimensions) {
    InferenceEngine::SizeVector expectedInput = {1, 3, 224, 224};
    InferenceEngine::SizeVector expectedOutput = {1, 1024, 1, 1};

    ASSERT_NO_THROW(MCMAdapter::getNetworkInputs(blobContentString.c_str(), networkInputs));
    ASSERT_NO_THROW(MCMAdapter::getNetworkOutputs(blobContentString.c_str(), networkOutputs));

    for (auto& networkInput : networkInputs) {
        InferenceEngine::SizeVector input = networkInput.second->getTensorDesc().getDims();
        ASSERT_EQ(expectedInput, input);
    }
    for (auto& networkOutput : networkOutputs) {
        InferenceEngine::SizeVector output = networkOutput.second->getTensorDesc().getDims();
        ASSERT_EQ(expectedOutput, output);
    }
}

// TODO: cannot parse input and output name correctly
TEST_F(BlobParser_Tests, DISABLED_CanGetInputsOutputsNames) {
    std::string expectedInputName = "input";
    std::string expectedOutputName = "output";

    ASSERT_NO_THROW(MCMAdapter::getNetworkInputs(blobContentString.c_str(), networkInputs));
    ASSERT_NO_THROW(MCMAdapter::getNetworkOutputs(blobContentString.c_str(), networkOutputs));

    for (auto& networkInput : networkInputs) {
        auto input = networkInput.first;
        ASSERT_EQ(expectedInputName, input);
    }
    for (auto& networkOutput : networkOutputs) {
        auto output = networkOutput.first;
        ASSERT_EQ(expectedOutputName, output);
    }
}
