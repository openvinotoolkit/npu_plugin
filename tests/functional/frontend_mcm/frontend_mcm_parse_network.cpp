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

#include "frontend_mcm_core.h"

using namespace vpu;

TEST_F(FrontendMCM_Core_Tests, CanParseBlob) {
    std::ifstream blobFile(resnetModel.graphPath, std::ios::binary);
    if (!blobFile.is_open()) {
        THROW_IE_EXCEPTION << "[ERROR] *Could not open file: " << resnetModel.graphPath;
    }

    std::ostringstream blobContentStream;
    blobContentStream << blobFile.rdbuf();
    const std::string& blobContentString = blobContentStream.str();
    ASSERT_NO_THROW(MCMAdapter::getNetworkInputs(blobContentString.c_str(), networkInputs));
    ASSERT_NO_THROW(MCMAdapter::getNetworkOutputs(blobContentString.c_str(), networkOutputs));
}

TEST_F(FrontendMCM_Core_Tests, CanGetInputsOutputsDimensions) {
    InferenceEngine::SizeVector expectedInput = {1, 3, 224, 224};
    InferenceEngine::SizeVector expectedOutput = {1, 1024, 1, 1};

    std::ifstream blobFile(resnetModel.graphPath, std::ios::binary);
    if (!blobFile.is_open()) {
        THROW_IE_EXCEPTION << "[ERROR] *Could not open file: " << resnetModel.graphPath;
    }

    std::ostringstream blobContentStream;
    blobContentStream << blobFile.rdbuf();
    const std::string& blobContentString = blobContentStream.str();
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
TEST_F(FrontendMCM_Core_Tests, DISABLED_CanGetInputsOutputsNames) {
    std::string expectedInputName = "input";
    std::string expectedOutputName = "output";

    std::ifstream blobFile(resnetModel.graphPath, std::ios::binary);
    if (!blobFile.is_open()) {
        THROW_IE_EXCEPTION << "[ERROR] *Could not open file: " << resnetModel.graphPath;
    }

    std::ostringstream blobContentStream;
    blobContentStream << blobFile.rdbuf();
    const std::string& blobContentString = blobContentStream.str();
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
