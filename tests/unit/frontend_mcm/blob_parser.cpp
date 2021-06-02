//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include <gtest/gtest.h>

#include <blob_parser.hpp>
#include <fstream>
#include <string>

#include <schema/graphfile/graphfile_generated.h>

#include "models/precompiled_resnet.h"

using namespace vpu;

class BlobParser_Tests : public ::testing::Test {
public:
    InferenceEngine::InputsDataMap networkInputs;
    InferenceEngine::OutputsDataMap networkOutputs;

    std::string blobContentString;
    MVCNN::GraphFileT graphFileInstance;

protected:
    void SetUp() override;
};

void BlobParser_Tests::SetUp() {
    const std::string graphPath = PrecompiledResNet_Helper::resnet50.graphPath;

    std::ifstream blobFile(graphPath, std::ios::binary);
    if (!blobFile.is_open()) {
        IE_THROW() << "[ERROR] *Could not open file: " << graphPath;
    }

    std::ostringstream blobContentStream;
    blobContentStream << blobFile.rdbuf();
    blobContentString = blobContentStream.str();

    const auto* graphFilePtr = MVCNN::GetGraphFile(blobContentString.c_str());
    graphFilePtr->UnPackTo(&graphFileInstance);
}

TEST_F(BlobParser_Tests, CanParseBlob) {
    ASSERT_NO_THROW(networkInputs = MCMAdapter::getNetworkInputs(graphFileInstance));
    ASSERT_NO_THROW(networkOutputs = MCMAdapter::getNetworkOutputs(graphFileInstance));
}

TEST_F(BlobParser_Tests, CanGetInputsOutputsDimensions) {
    InferenceEngine::SizeVector expectedInput = {1, 3, 224, 224};
    InferenceEngine::SizeVector expectedOutput = {1, 1000, 1, 1};

    ASSERT_NO_THROW(networkInputs = MCMAdapter::getNetworkInputs(graphFileInstance));
    ASSERT_NO_THROW(networkOutputs = MCMAdapter::getNetworkOutputs(graphFileInstance));

    for (const auto& networkInput : networkInputs) {
        InferenceEngine::SizeVector input = networkInput.second->getTensorDesc().getDims();
        ASSERT_EQ(expectedInput, input);
    }
    for (const auto& networkOutput : networkOutputs) {
        InferenceEngine::SizeVector output = networkOutput.second->getTensorDesc().getDims();
        ASSERT_EQ(expectedOutput, output);
    }
}

TEST_F(BlobParser_Tests, CanGetInputsOutputsNames) {
    std::string expectedInputName = "data";
    std::string expectedOutputName = "Output_0";

    ASSERT_NO_THROW(networkInputs = MCMAdapter::getNetworkInputs(graphFileInstance));
    ASSERT_NO_THROW(networkOutputs = MCMAdapter::getNetworkOutputs(graphFileInstance));

    for (const auto& networkInput : networkInputs) {
        auto input = networkInput.first;
        ASSERT_EQ(expectedInputName, input);
    }
    for (const auto& networkOutput : networkOutputs) {
        auto output = networkOutput.first;
        ASSERT_EQ(expectedOutputName, output);
    }
}
