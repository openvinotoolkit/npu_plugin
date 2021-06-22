//
// Copyright 2019 Intel Corporation.
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

#include <Inference.h>

#include <fstream>

#include "core_api.h"
#include "gtest/gtest.h"
#include "hddl2_helpers/helper_workload_context.h"
#include "helper_remote_context.h"
#include "simple_graph.hpp"

namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
class ImportNetwork_Tests : public CoreAPI_Tests {
public:
    IE::ParamMap params;

protected:
    void SetUp() override;
    WorkloadContext_Helper _workloadContextHelper;
    WorkloadID _workloadId = 0;
    std::stringstream blobStream;
    InferenceEngine::SizeVector _dims;
    std::string _inputName;
    std::string _outputName;
    std::string _outputDevName;
    std::string _deviceId;
};

void ImportNetwork_Tests::SetUp() {
    _workloadId = _workloadContextHelper.getWorkloadId();
    params = Remote_Context_Helper::wrapWorkloadIdToMap(_workloadId);
    _deviceId = "VPUX";
    _dims = {1, 3, 416, 416};
    _inputName = "input_0";
    _outputName = "output_0";
    _outputDevName = "output_dev_0";
    utils::simpleGraph::getExeNetwork(_deviceId, _dims, _inputName, _outputName, _outputDevName)->Export(blobStream);
}

//------------------------------------------------------------------------------
TEST_F(ImportNetwork_Tests, CanFindPlugin) {
    ASSERT_NO_THROW(ie.ImportNetwork(blobStream, pluginName));
}

TEST_F(ImportNetwork_Tests, CanCreateExecutableNetwork) {
    ASSERT_NO_THROW(auto executableNetwork = ie.ImportNetwork(blobStream, pluginName));
}

TEST_F(ImportNetwork_Tests, CanCreateExecutableNetworkWithConfig) {
    std::map<std::string, std::string> config = {};
    ASSERT_NO_THROW(auto executableNetwork = ie.ImportNetwork(blobStream, pluginName, config));
}

TEST_F(ImportNetwork_Tests, CanCreateInferRequest) {
    IE::ExecutableNetwork executableNetwork;
    ASSERT_NO_THROW(executableNetwork = ie.ImportNetwork(blobStream, pluginName));

    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());
}

TEST_F(ImportNetwork_Tests, CanCreateExecutableNetworkWithStream) {
    const std::map<std::string, std::string> config = {};

    IE::RemoteContext::Ptr remoteContextPtr = ie.CreateContext(pluginName, params);

    ASSERT_NO_THROW(auto executableNetwork = ie.ImportNetwork(blobStream, remoteContextPtr, config));
}

TEST_F(ImportNetwork_Tests, canParseInputAndOutput) {
    const std::string expected_input_name = _inputName;
    const IE::Precision expected_input_precision = IE::Precision::U8;
    const IE::SizeVector expected_input_dims = _dims;
    const IE::Layout expected_input_layout = IE::Layout::NCHW;

    const std::string expected_output_name = _outputName;
    const IE::Precision expected_output_precision = IE::Precision::FP32;
    const IE::SizeVector expected_output_dims = _dims;
    const IE::Layout expected_output_layout = IE::Layout::NCHW;

    IE::ExecutableNetwork executableNetwork = ie.ImportNetwork(blobStream, "VPUX");

    IE::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    auto inputBlobName = executableNetwork.GetInputsInfo().begin()->first;
    const auto inputBlob = inferRequest.GetBlob(inputBlobName);
    EXPECT_EQ(expected_input_name, inputBlobName);
    EXPECT_EQ(expected_input_precision, inputBlob->getTensorDesc().getPrecision());
    EXPECT_EQ(expected_input_dims, inputBlob->getTensorDesc().getDims());
    EXPECT_EQ(expected_input_layout, inputBlob->getTensorDesc().getLayout());

    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    const auto outputBlob = inferRequest.GetBlob(outputBlobName);
    EXPECT_EQ(expected_output_name, outputBlobName);
    EXPECT_EQ(expected_output_precision, outputBlob->getTensorDesc().getPrecision());
    EXPECT_EQ(expected_output_dims, outputBlob->getTensorDesc().getDims());
    EXPECT_EQ(expected_output_layout, outputBlob->getTensorDesc().getLayout());
}
