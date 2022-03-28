//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <gtest/gtest.h>

#include <blob_parser.hpp>
#include <fstream>
#include <string>

#include <schema/graphfile/graphfile_generated.h>
#include "simple_graph.hpp"

using namespace vpu;

class BlobParser_Tests : public ::testing::Test {
public:
    InferenceEngine::InputsDataMap networkInputs;
    InferenceEngine::OutputsDataMap networkOutputs;

    MVCNN::SummaryHeader* graphHeader = nullptr;
    MCMAdapter::graphTensors* graphInputs = nullptr;
    MCMAdapter::graphTensors* graphOutputs = nullptr;
    InferenceEngine::SizeVector _dims;
    std::string _inputName;
    std::string _outputName;
    std::string _outputDevName;
    std::string _deviceId;
    std::string _compiledModel;

protected:
    void SetUp() override;
};

void BlobParser_Tests::SetUp() {
    _deviceId = 
        std::getenv("IE_KMB_TESTS_DEVICE_NAME") != nullptr ? std::getenv("IE_KMB_TESTS_DEVICE_NAME") : "VPUX";
    _dims = {1, 3, 224, 224};
    _inputName = "input_0";
    _outputName = "output_0";
    _outputDevName = "output_dev0";

    std::stringstream blobStream(std::ios_base::binary | std::ios_base::in | std::ios_base::out);
    utils::simpleGraph::getExeNetwork(_deviceId, _dims, _inputName, _outputName, _outputDevName)->Export(blobStream);

    _compiledModel = blobStream.str();
    const auto graphFilePtr = MVCNN::GetGraphFile(_compiledModel.data());
    ASSERT_NE(graphFilePtr, nullptr);
    graphHeader = const_cast<MVCNN::SummaryHeader*>(graphFilePtr->header());
    ASSERT_NE(graphHeader, nullptr);
    graphInputs = const_cast<MCMAdapter::graphTensors*>(graphHeader->net_input());
    ASSERT_NE(graphInputs, nullptr);
    graphOutputs = const_cast<MCMAdapter::graphTensors*>(graphHeader->net_output());
    ASSERT_NE(graphOutputs, nullptr);
}

TEST_F(BlobParser_Tests, CanParseBlob) {
    ASSERT_NO_THROW(networkInputs = MCMAdapter::getNetworkInputs(*graphInputs));
    ASSERT_NO_THROW(networkOutputs = MCMAdapter::getNetworkOutputs(*graphOutputs));
}

TEST_F(BlobParser_Tests, CanGetInputsOutputsDimensions) {
    const auto expectedInput = _dims;
    const auto expectedOutput = _dims;

    ASSERT_NO_THROW(networkInputs = MCMAdapter::getNetworkInputs(*graphInputs));
    ASSERT_NO_THROW(networkOutputs = MCMAdapter::getNetworkOutputs(*graphOutputs));

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
    const auto expectedInputName = _inputName;
    const auto expectedOutputName = _outputDevName;

    ASSERT_NO_THROW(networkInputs = MCMAdapter::getNetworkInputs(*graphInputs));
    ASSERT_NO_THROW(networkOutputs = MCMAdapter::getNetworkOutputs(*graphOutputs));

    for (const auto& networkInput : networkInputs) {
        auto input = networkInput.first;
        ASSERT_EQ(expectedInputName, input);
    }
    for (const auto& networkOutput : networkOutputs) {
        auto output = networkOutput.first;
        ASSERT_EQ(expectedOutputName, output);
    }
}
