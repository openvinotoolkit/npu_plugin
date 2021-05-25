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
#include "models/precompiled_resnet.h"

namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
class ImportNetwork_Tests : public CoreAPI_Tests {
public:
    modelBlobInfo blobInfo = PrecompiledResNet_Helper::resnet50;
    IE::ParamMap params;

protected:
    void SetUp() override;
    WorkloadContext_Helper _workloadContextHelper;
    WorkloadID _workloadId = 0;
};

void ImportNetwork_Tests::SetUp() {
    _workloadId = _workloadContextHelper.getWorkloadId();
    params = Remote_Context_Helper::wrapWorkloadIdToMap(_workloadId);
}

//------------------------------------------------------------------------------
TEST_F(ImportNetwork_Tests, CanFindPlugin) {
    ASSERT_NO_THROW(ie.ImportNetwork(blobInfo.graphPath, pluginName));
}

TEST_F(ImportNetwork_Tests, CanCreateExecutableNetwork) {
    ASSERT_NO_THROW(auto executableNetwork = ie.ImportNetwork(blobInfo.graphPath, pluginName));
}

TEST_F(ImportNetwork_Tests, CanCreateExecutableNetworkWithConfig) {
    std::map<std::string, std::string> config = {};
    ASSERT_NO_THROW(auto executableNetwork = ie.ImportNetwork(blobInfo.graphPath, pluginName, config));
}

TEST_F(ImportNetwork_Tests, CanCreateInferRequest) {
    IE::ExecutableNetwork executableNetwork;
    ASSERT_NO_THROW(executableNetwork = ie.ImportNetwork(blobInfo.graphPath, pluginName));

    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());
}

TEST_F(ImportNetwork_Tests, CanCreateExecutableNetworkWithStream) {
    const std::map<std::string, std::string> config = {};

    std::filebuf blobFile;
    if (!blobFile.open(blobInfo.graphPath, std::ios::in | std::ios::binary)) {
        IE_THROW() << "Could not open file: " << blobInfo.graphPath;
    }
    std::istream tmp_stream(&blobFile);

    IE::RemoteContext::Ptr remoteContextPtr = ie.CreateContext(pluginName, params);

    ASSERT_NO_THROW(auto executableNetwork = ie.ImportNetwork(tmp_stream, remoteContextPtr, config));
    blobFile.close();
}

TEST_F(ImportNetwork_Tests, canParseInputAndOutput) {
    const std::string expected_input_name = "data";
    const IE::Precision expected_input_precision = IE::Precision::U8;
    const IE::SizeVector expected_input_dims = {1, 3, 224, 224};
    const IE::Layout expected_input_layout = IE::Layout::NCHW;

    const std::string expected_output_name = "prob";
    const IE::Precision expected_output_precision = IE::Precision::FP32;
    const IE::SizeVector expected_output_dims = {1, 1000};
    const IE::Layout expected_output_layout = IE::Layout::NC;

    IE::ExecutableNetwork executableNetwork = ie.ImportNetwork(blobInfo.graphPath, "VPUX");

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
