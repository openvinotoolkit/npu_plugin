// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"

#define BOUND (10.0f)
#define ERROR_BOUND (1.2e-3f)
#define ERROR_BOUND_WITH_TANH (1.0e-3f)
using namespace InferenceEngine;

class myriadLayersTestsTanh_nightly: public myriadLayersTests_nightly,
                             public testing::WithParamInterface<Dims> {
};

TEST_P(myriadLayersTestsTanh_nightly, TestsTanh)
{
    auto p = ::testing::WithParamInterface<Dims>::GetParam();
    SetInputTensor(p);
    SetOutputTensor(p);

    NetworkInit("TanH",
                nullptr,
                0,
                0,
                nullptr,
                InferenceEngine::Precision::FP16 // output precision
        );
    SetFirstInputToRange(-BOUND, BOUND);
    ASSERT_TRUE(Infer());
    /* output check */
    ref_tanh(_inputMap.begin()->second, _refBlob);
    Compare(_outputMap.begin()->second, _refBlob, ERROR_BOUND);
}

static std::vector<Dims> s_tanhParams = {
    {{1, 1, 16, 16}},
    {{1, 2, 16, 16}},
    {{1, 3, 16, 16}},
    {{1, 1, 53, 16}},
    {{1, 2, 53, 16}},
    {{1, 3, 53, 16}},
    {{1, 1, 224, 224}},
    {{1, 2, 224, 224}},
    {{1, 3, 224, 224}},
    {{1, 1, 224, 235}},
    {{1, 2, 224, 235}},
    {{1, 3, 224, 235}},
//  {{1, 1, 277, 230}},   /* BUG: https://gitlab-icv.inn.intel.com/inference-engine/inference-engine/issues/9 */
//  {{1, 2, 277, 230}},   /* BUG: https://gitlab-icv.inn.intel.com/inference-engine/inference-engine/issues/9 */
//  {{1, 3, 277, 230}})); /* BUG: https://gitlab-icv.inn.intel.com/inference-engine/inference-engine/issues/9 */

};

INSTANTIATE_TEST_CASE_P(
        accuracy, myriadLayersTestsTanh_nightly,
        ::testing::ValuesIn(s_tanhParams));

static std::vector<InferenceEngine::SizeVector> s_convolutionTensors = {
    {{1, 8, 4, 16}, {16, 8, 16}}  //NCHW
};

/* tests subset to check 2 layers operation invocation */
/* additional tests for 2D and 3D tensors added        */
static std::vector<int32_t> s_dimensionsFC = {
    4, 3
};

static std::vector<int32_t> s_addBiasFC = {
    1, 0
};

/* to decrease tests duration and tests amount */
static std::vector<fcon_test_params> s_fcTestParamsSubset = {
    {{1, 1, 16, 8},     8, 0.02f},
    {{1, 1, 8, 40},     8, 0.02f},
    {{1, 4, 8, 16},     4, 0.065f},
    {{1, 16, 16, 16},  16, 0.36f},
    {{1, 16, 8, 8},    8, 0.065f}
};

class myriadLayerConvolutionWithTanH_nightly: public ConvolutionTest<>{
};

TEST_P(myriadLayerConvolutionWithTanH_nightly, Convolution) {
    AddLayer("TanH",
             nullptr,
             {_output_tensor},
             {_output_tensor},
             ref_tanh_wrap);

    float maxerr = 0;
    if (group == 1)
        maxerr = 0.00055 * IC * kernel.x * kernel.y;
    else // TODO: currently dephConv is slightly less accurate
        maxerr = 0.00066 * (IC / group) * kernel.x * kernel.y;
    ASSERT_TRUE(GenerateNetAndInfer());
    Compare(_outputMap.begin()->second, GenReferenceOutput(), maxerr);
}

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayerConvolutionWithTanH_nightly,
        ::testing::Combine(
            ::testing::ValuesIn(g_convolutionTensors)
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<uint32_t>(16)
          , ::testing::Values<uint32_t>(1)
          )
);

class myriadLayersTestsMaxPoolingWithTanh_nightly: public PoolingTest<POOLING_MAX>{
};

class myriadLayersTestsAvgPoolingWithTanh_nightly: public PoolingTest<POOLING_AVG>{
};

TEST_P(myriadLayersTestsMaxPoolingWithTanh_nightly, TestsMaxPoolingWithTanh)
{
    AddLayer("TanH",
             nullptr,
             {_output_tensor},
             {_output_tensor},
             ref_tanh_wrap);
    ASSERT_TRUE(GenerateNetAndInfer());
    Compare(_outputMap.begin()->second, GenReferenceOutput(), ERROR_BOUND_WITH_TANH);
}

TEST_P(myriadLayersTestsAvgPoolingWithTanh_nightly, TestsAvgPoolingWithTanh)
{
    AddLayer("TanH",
             nullptr,
             {_output_tensor},
             {_output_tensor},
             ref_tanh_wrap);
    ASSERT_TRUE(GenerateNetAndInfer());
    Compare(_outputMap.begin()->second, GenReferenceOutput(), ERROR_BOUND_WITH_TANH);
}

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsMaxPoolingWithTanh_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(g_poolingInput),
        ::testing::ValuesIn(g_poolingLayerParamsLite),
        ::testing::ValuesIn(g_poolingLayout))
);

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsAvgPoolingWithTanh_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(g_poolingInput),
        ::testing::ValuesIn(g_poolingLayerParamsLite),
        ::testing::ValuesIn(g_poolingLayout))
);

class myriadLayerFullyConnectedWithTanH_nightly: public FCTest<>{
};

TEST_P(myriadLayerFullyConnectedWithTanH_nightly, TestsFullyConnected)
{
    AddLayer("TanH",
             nullptr,
             {_output_tensor},
             {_output_tensor},
             ref_tanh_wrap);
    ASSERT_TRUE(GenerateNetAndInfer());
    Compare(_outputMap.begin()->second, GenReferenceOutput(), _par.error_bound);
}

INSTANTIATE_TEST_CASE_P(
    accuracy, myriadLayerFullyConnectedWithTanH_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(g_fcTestParamsSubset),
        ::testing::Values(g_dimensionsFC[0]),
        ::testing::ValuesIn(g_addBiasFC)
    )
);
