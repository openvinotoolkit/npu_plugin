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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "kmb_infer_request.h"

namespace ie = InferenceEngine;

using namespace ::testing;
using namespace vpu::KmbPlugin;

class kmbInferRequestUnitTests : public ::testing::Test {
protected:
    InferenceEngine::InputsDataMap setupInputsWithSingleElement() {
        std::string inputName = "input";
        ie::TensorDesc inputDescription = ie::TensorDesc(ie::Precision::U8, {1, 3, 224, 224}, ie::Layout::NHWC);
        ie::DataPtr inputData = std::make_shared<ie::Data>(inputName, inputDescription);
        ie::InputInfo::Ptr inputInfo = std::make_shared<ie::InputInfo>();
        inputInfo->setInputData(inputData);
        InferenceEngine::InputsDataMap inputs = {{inputName, inputInfo}};

        return inputs;
    }

    InferenceEngine::OutputsDataMap setupOutputsWithSingleElement() {
        std::string outputName = "output";
        ie::TensorDesc outputDescription = ie::TensorDesc(ie::Precision::U8, {1000}, ie::Layout::C);
        ie::DataPtr outputData = std::make_shared<ie::Data>(outputName, outputDescription);
        InferenceEngine::OutputsDataMap outputs = {{outputName, outputData}};

        return outputs;
    }
};

class MockExecutor : public KmbExecutor {
public:
    MockExecutor(const KmbConfig& config): KmbExecutor(config) {}
    MOCK_METHOD0(deallocateGraph, void());
    MOCK_METHOD1(allocateGraph, void(const std::vector<char>& graphFileContent));
    MOCK_METHOD2(getResult, void(void* result_data, unsigned int result_bytes));
    MOCK_METHOD4(queueInference, void(void* input_data, size_t input_bytes, void* result_data, size_t result_bytes));
    MOCK_CONST_METHOD0(getNetworkInputs, ie::InputsDataMap&());
    MOCK_CONST_METHOD0(getNetworkOutputs, ie::OutputsDataMap&());
};

TEST_F(kmbInferRequestUnitTests, cannotCreateInferRequestWithEmptyInputAndOutput) {
    KmbConfig config;
    auto executor = std::make_shared<MockExecutor>(config);
    KmbInferRequest::Ptr inferRequest;

    ASSERT_THROW(inferRequest = std::make_shared<KmbInferRequest>(InferenceEngine::InputsDataMap(),
                     InferenceEngine::OutputsDataMap(), std::vector<vpu::StageMetaInfo>(), config, executor),
        InferenceEngine::details::InferenceEngineException);
}

TEST_F(kmbInferRequestUnitTests, canCreateInferRequestWithValidParameters) {
    KmbConfig config;
    auto executor = std::make_shared<MockExecutor>(config);

    KmbInferRequest::Ptr inferRequest;

    auto inputs = setupInputsWithSingleElement();
    auto outputs = setupOutputsWithSingleElement();

    ASSERT_NO_THROW(inferRequest = std::make_shared<KmbInferRequest>(
                        inputs, outputs, std::vector<vpu::StageMetaInfo>(), config, executor));
}
