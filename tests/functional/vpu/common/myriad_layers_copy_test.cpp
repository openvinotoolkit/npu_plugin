// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include <algorithm>

using std::tr1::tuple;
using std::tr1::get;

using namespace InferenceEngine;

const int MAX_DIMS = 8;

struct nd_tensor_test_params {
    size_t dims[MAX_DIMS];
};

PRETTY_PARAM(NDims, nd_tensor_test_params);

typedef myriadLayerTestBaseWithParam<tuple<NDims, int>> myriadLayerCopy_nightly;

TEST_P(myriadLayerCopy_nightly, Copy) {

    nd_tensor_test_params input_dims = get<0>(GetParam());
    int ndims = get<1>(GetParam());

    IN_OUT_desc inputTensors;
    IN_OUT_desc outputTensors;
    outputTensors.resize(1);
    inputTensors.resize(1);
    inputTensors[0].resize(ndims);
    outputTensors[0].resize(ndims);

    for (int i = 0; i < ndims; i++)
    {
        inputTensors[0][i] = input_dims.dims[i];
        outputTensors[0][i] = input_dims.dims[i];
    }

    SetInputTensors(inputTensors);
    SetOutputTensors(outputTensors);

    _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);

    NetworkInit("Copy", 0, 0, 0, nullptr, InferenceEngine::Precision::FP16);
    SetFirstInputToRange(1.0f, 100.0f);

    ASSERT_TRUE(Infer());
    auto inputBlob = _inputMap.begin()->second;
    auto outputBlob = _outputMap.begin()->second;

    Compare(outputBlob, inputBlob, 0);
}

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayerCopy_nightly,
        ::testing::Combine(
            ::testing::Values<NDims>(MAKE_STRUCT(nd_tensor_test_params, {36, 19, 20, 21})
                                   , MAKE_STRUCT(nd_tensor_test_params, {7, 8, 5, 12})
                                   , MAKE_STRUCT(nd_tensor_test_params, {196, 12, 20, 5}))
          , ::testing::Values<int>(2, 3, 4)
                        ));
