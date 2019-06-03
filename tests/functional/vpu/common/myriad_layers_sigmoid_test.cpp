// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"
#include <cmath>

#define BOUND (10.0f)
#define ERROR_BOUND (1.e-3f)
#define ERROR_BOUND_WITH_SIGMOID (1.e-3f)

using namespace InferenceEngine;

class myriadLayersTestsSigmoid_nightly: public myriadLayersTests_nightly,
                           public testing::WithParamInterface<Dims> {
public:
};

TEST_P(myriadLayersTestsSigmoid_nightly, TestsSigmoid)
{
    auto p = ::testing::WithParamInterface<Dims>::GetParam();
    SetInputTensor(p);
    SetOutputTensor(p);
    NetworkInit("Sigmoid",
                    nullptr,
                    0,
                    0,
                    nullptr,
                    InferenceEngine::Precision::FP16 // output precision
    );
    SetFirstInputToRange(-BOUND, BOUND);
    ASSERT_TRUE(Infer());

    /* output check */
    ref_sigmoid(_inputMap.begin()->second, _refBlob);
    Compare(_outputMap.begin()->second, _refBlob, ERROR_BOUND);
}

static std::vector<Dims> s_sigmoidParams = {
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
    {{1, 3, 224, 235}}
};
//  {{1, 1, 277, 230}},   /* https://gitlab-icv.inn.intel.com/inference-engine/inference-engine/issues/10 */
//  {{1, 2, 277, 230}},   /* https://gitlab-icv.inn.intel.com/inference-engine/inference-engine/issues/10 */
//  {{1, 3, 277, 230}})); /* https://gitlab-icv.inn.intel.com/inference-engine/inference-engine/issues/10 */

INSTANTIATE_TEST_CASE_P(
        accuracy, myriadLayersTestsSigmoid_nightly,
        ::testing::ValuesIn(s_sigmoidParams));

class myriadLayersTestsMaxPoolingWithSigmoid_nightly: public PoolingTest<POOLING_MAX>{
};

class myriadLayersTestsAvgPoolingWithSigmoid_nightly: public PoolingTest<POOLING_AVG>{
};

TEST_P(myriadLayersTestsMaxPoolingWithSigmoid_nightly, TestsMaxPoolingWithSigmoid)
{
    AddLayer("Sigmoid",
             nullptr,
             {_output_tensor},
             {_output_tensor},
             ref_sigmoid_wrap);
    ASSERT_TRUE(GenerateNetAndInfer());
    Compare(_outputMap.begin()->second, GenReferenceOutput(), ERROR_BOUND_WITH_SIGMOID);
}

TEST_P(myriadLayersTestsAvgPoolingWithSigmoid_nightly, TestsAvgPoolingWithSigmoid)
{
    AddLayer("Sigmoid",
             nullptr,
             {_output_tensor},
             {_output_tensor},
             ref_sigmoid_wrap);
    ASSERT_TRUE(GenerateNetAndInfer());
    Compare(_outputMap.begin()->second, GenReferenceOutput(), ERROR_BOUND_WITH_SIGMOID);
}

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsMaxPoolingWithSigmoid_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(g_poolingInput),
        ::testing::ValuesIn(g_poolingLayerParamsLite),
        ::testing::ValuesIn(g_poolingLayout))
);

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsAvgPoolingWithSigmoid_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(g_poolingInput),
        ::testing::ValuesIn(g_poolingLayerParamsLite),
        ::testing::ValuesIn(g_poolingLayout))
);

class myriadLayerConvolutionWithSigmoid_nightly: public ConvolutionTest<>{
};

TEST_P(myriadLayerConvolutionWithSigmoid_nightly, Convolution) {
    AddLayer("Sigmoid",
             nullptr,
             {_output_tensor},
             {_output_tensor},
             ref_sigmoid_wrap);

    float maxerr = 0;
    if (group == 1)
        maxerr = 0.00055 * IC * kernel.x * kernel.y;
    else // TODO: currently dephConv is slightly less accurate
        maxerr = 0.00066 * (IC / group) * kernel.x * kernel.y;
    ASSERT_TRUE(GenerateNetAndInfer());
    Compare(_outputMap.begin()->second, GenReferenceOutput(), maxerr);
}

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayerConvolutionWithSigmoid_nightly,
        ::testing::Combine(
            ::testing::ValuesIn(g_convolutionTensors)
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<uint32_t>(16)
          , ::testing::Values<uint32_t>(1)
          )
);

class myriadLayerFullyConnectedWithSigmoid_nightly: public FCTest<>{
};

TEST_P(myriadLayerFullyConnectedWithSigmoid_nightly, TestsFullyConnected)
{
    AddLayer("Sigmoid",
             nullptr,
             {_output_tensor},
             {_output_tensor},
             ref_sigmoid_wrap);
    ASSERT_TRUE(GenerateNetAndInfer());
    Compare(_outputMap.begin()->second, GenReferenceOutput(), _par.error_bound);
}

INSTANTIATE_TEST_CASE_P(
    accuracy, myriadLayerFullyConnectedWithSigmoid_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(g_fcTestParamsSubset),
        ::testing::Values(g_dimensionsFC[0]),
        ::testing::ValuesIn(g_addBiasFC)
    )
);
