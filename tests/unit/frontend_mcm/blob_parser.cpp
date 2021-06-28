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
#include "simple_graph.hpp"

using namespace vpu;

class BlobParser_Tests : public ::testing::Test {
public:
    InferenceEngine::InputsDataMap networkInputs;
    InferenceEngine::OutputsDataMap networkOutputs;

    MVCNN::SummaryHeader* graphHeader = nullptr;
    MCMAdapter::graphTensors* graphInputs = nullptr;
    MCMAdapter::graphTensors* graphOutputs = nullptr;
    std::stringstream _blobStream;
    InferenceEngine::SizeVector _dims;
    std::string _inputName;
    std::string _outputName;
    std::string _outputDevName;
    std::string _deviceId;

protected:
    void SetUp() override;
};

void BlobParser_Tests::SetUp() {
    _deviceId = "VPUX";
    _dims = {1, 3, 224, 224};
    _inputName = "input_0";
    _outputName = "output_0";
    _outputDevName = "output_dev0";
    utils::simpleGraph::getExeNetwork(_deviceId, _dims, _inputName, _outputName, _outputDevName)->Export(_blobStream);

    const auto graphFilePtr = MVCNN::GetGraphFile(_blobStream.str().data());
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
