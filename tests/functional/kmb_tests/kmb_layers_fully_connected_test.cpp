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

#include "kmb_layers_tests.hpp"

#define ERROR_BOUND (.1f)

using namespace InferenceEngine;

typedef std::tuple<tensor_test_params, tensor_test_params, size_t> fully_connected_test_params;
typedef kmbLayerTestBaseWithParam< fully_connected_test_params > kmbLayersTestsFullyConnectedParams;

TEST_P(kmbLayersTestsFullyConnectedParams, TestsFullyConnected) {
    auto param = GetParam();
    tensor_test_params inputTensor = std::get<0>(param);
    tensor_test_params outputTensor = std::get<1>(param);
    size_t outSize = std::get<2>(param);

    size_t weightsSize = outSize * inputTensor.n * inputTensor.c * inputTensor.h * inputTensor.w * sizeof(uint16_t);
    size_t biasesSize = 0;

    const ::testing::TestInfo* const test_info =
      ::testing::UnitTest::GetInstance()->current_test_info();

    std::cout << ::testing::UnitTest::GetInstance()->current_test_info()->name() << " test_info->name()=" <<
            test_info->name() << " test_info->test_case_name() " << test_info->test_case_name() << std::endl;

    std::map<std::string, std::string> params;
    params["out-size"] = std::to_string(outSize);

    TBlob<uint8_t>::Ptr weightsBlob(GenWeights(weightsSize + biasesSize));

    SetInputTensor(inputTensor);
    SetOutputTensor(outputTensor);
    NetworkInit("FullyConnected",
                &params,
                weightsSize,
                biasesSize,
                weightsBlob,
                InferenceEngine::Precision::FP16 // output precision
    );
}

static const fully_connected_test_params paramsTable[] = {
    std::make_tuple<tensor_test_params, tensor_test_params, size_t>(
        {1, 128, 2, 2},  // input tensor
        {1, 1024, 1, 1}, // output tensor
        128              // out-size
    ),
    std::make_tuple<tensor_test_params, tensor_test_params, size_t>(
        {1, 64, 2, 2}, // input tensor
        {1, 2048, 1, 1}, // output tensor
        64               // out-size
    ),
};

INSTANTIATE_TEST_CASE_P(loadNetworkNoThrow, kmbLayersTestsFullyConnectedParams,
    ::testing::ValuesIn(paramsTable)
);
