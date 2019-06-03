// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"
#include <algorithm>

using std::tuple;
using std::get;

using namespace InferenceEngine;

PRETTY_PARAM(ChannelShared, int);
typedef myriadLayerTestBaseWithParam<tuple<Dims, ChannelShared >> myriadLayerPReLU_nightly;

TEST_P(myriadLayerPReLU_nightly, PReLU) {
    tensor_test_params dims = get<0>(GetParam());
    int channel_shared = get<1>(GetParam());

    SetInputTensor(dims);
    SetOutputTensor(dims);

    int num_weights = channel_shared ? 1 : dims.c;
    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(GenWeights(num_weights));
    uint16_t* weights = weights_ptr->data().as<uint16_t*>();

    std::map<std::string, std::string> layer_params = {{"channel_shared", std::to_string(channel_shared)}};
    NetworkInit("PReLU", &layer_params, weights_ptr->byteSize(), 0, weights_ptr, InferenceEngine::Precision::FP16);
    SetFirstInputToRange(0, 5.0f);
    ASSERT_TRUE(Infer());

    auto inputBlob = _inputMap.begin()->second;
    auto outputBlob = _outputMap.begin()->second;

    ref_PReLU(inputBlob, _refBlob, weights, num_weights);
    Compare(outputBlob, _refBlob, 0);
}

INSTANTIATE_TEST_CASE_P(accuracy_PReLU, myriadLayerPReLU_nightly,
        ::testing::Combine(
            ::testing::Values<Dims>(MAKE_STRUCT(tensor_test_params, 1, 13, 77,  99)
                                  , MAKE_STRUCT(tensor_test_params, 1,  9, 68,  83)
                                  , MAKE_STRUCT(tensor_test_params, 1,  4, 64,  64)
                                  , MAKE_STRUCT(tensor_test_params, 1,  1, 11,   8))
          , ::testing::Values<ChannelShared>(0, 1)
                        ));

struct  PReLULayerDef {
    ParamsStruct list;
}PReLULayer;

static std::vector<PReLULayerDef> s_PReluLayerParams = {
    {{{PRELU_PARAM, "0"}}},
    {{{PRELU_PARAM, "1"}}}
};

class myriadLayerFullyConnectedWithPReLU_nightly: public FCTest<PReLULayerDef>{
};

#define TEST_BODY \
    int channel_shared = 0;\
    if (!extraLayerParams.list.empty()) {\
        auto iter = extraLayerParams.list.find(PRELU_PARAM);\
        if (iter != extraLayerParams.list.end()) {\
             channel_shared = std::stof(iter->second);\
        }\
    }\
    size_t weightsSize = 1;\
    if (channel_shared == 0) {\
        int32_t OW;\
        int32_t OH;\
        int32_t OC;\
        get_dims(_output_tensor, OW, OH, OC);\
        weightsSize = OC;\
    }\
    AddLayer("PReLU",\
             &extraLayerParams.list,\
             weightsSize,\
             0,\
             defaultWeightsRange,\
             {_output_tensor},\
             {_output_tensor},\
             ref_PReLU_wrap);\
    ASSERT_TRUE(GenerateNetAndInfer());

TEST_P(myriadLayerFullyConnectedWithPReLU_nightly, TestsFullyConnected)
{
    auto p = ::testing::WithParamInterface<std::tuple<fcon_test_params, int32_t, int32_t, PReLULayerDef>>::GetParam();
    auto extraLayerParams = std::get<3>(p);
    TEST_BODY;
    Compare(_outputMap.begin()->second, GenReferenceOutput(), _par.error_bound);
}

INSTANTIATE_TEST_CASE_P(
    accuracy, myriadLayerFullyConnectedWithPReLU_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(g_fcTestParamsSubset),
        ::testing::Values(g_dimensionsFC[0]),
        ::testing::ValuesIn(g_addBiasFC),
        ::testing::ValuesIn(s_PReluLayerParams)
    )
);

#define ERROR_BOUND_WITH_RELU (4.e-3f)

class myriadLayersTestsMaxPoolingWithPReLU_nightly: public PoolingTest<POOLING_MAX, PReLULayerDef>{
};

class myriadLayersTestsAvgPoolingWithPReLU_nightly: public PoolingTest<POOLING_AVG, PReLULayerDef>{
};

TEST_P(myriadLayersTestsMaxPoolingWithPReLU_nightly, TestsMaxPoolingWithPReLU)
{
    auto p = ::testing::WithParamInterface<std::tuple<InferenceEngine::SizeVector, pooling_layer_params, const char*, PReLULayerDef>>::GetParam();
    auto extraLayerParams = std::get<3>(p);
    TEST_BODY;
    Compare(_outputMap.begin()->second, GenReferenceOutput(), ERROR_BOUND_WITH_RELU);
}

TEST_P(myriadLayersTestsAvgPoolingWithPReLU_nightly, TestsAvgPoolingWithPReLU)
{
    auto p = ::testing::WithParamInterface<std::tuple<InferenceEngine::SizeVector, pooling_layer_params, const char*, PReLULayerDef>>::GetParam();
    auto extraLayerParams = std::get<3>(p);
    TEST_BODY;
    Compare(_outputMap.begin()->second, GenReferenceOutput(), ERROR_BOUND_WITH_RELU);
}

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsMaxPoolingWithPReLU_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(g_poolingInput),
        ::testing::ValuesIn(g_poolingLayerParamsLite),
        ::testing::ValuesIn(g_poolingLayout),
        ::testing::ValuesIn(s_PReluLayerParams))
);

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsAvgPoolingWithPReLU_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(g_poolingInput),
        ::testing::ValuesIn(g_poolingLayerParamsLite),
        ::testing::ValuesIn(g_poolingLayout),
        ::testing::ValuesIn(s_PReluLayerParams))
);

INSTANTIATE_TEST_CASE_P(accuracy_postop, myriadLayersTestsMaxPoolingWithPReLU_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(g_poolingInput_postOp),
        ::testing::Values<pooling_layer_params>(MAKE_STRUCT(pooling_layer_params, {3, 3}, {1, 1}, {1, 1})),
        ::testing::ValuesIn(g_poolingLayout),
        ::testing::Values<PReLULayerDef>(MAKE_STRUCT(PReLULayerDef, {{{PRELU_PARAM, "0"}}})))
);

INSTANTIATE_TEST_CASE_P(accuracy_postop, myriadLayersTestsAvgPoolingWithPReLU_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(g_poolingInput_postOp),
        ::testing::Values<pooling_layer_params>(MAKE_STRUCT(pooling_layer_params, {3, 3}, {1, 1}, {1, 1})),
        ::testing::ValuesIn(g_poolingLayout),
        ::testing::Values<PReLULayerDef>(MAKE_STRUCT(PReLULayerDef, {{{PRELU_PARAM, "0"}}})))
);

class myriadLayerConvolutionWithPReLU_nightly: public ConvolutionTest<PReLULayerDef>{
};

TEST_P(myriadLayerConvolutionWithPReLU_nightly, Convolution) {
    auto p = ::testing::WithParamInterface<std::tuple<InferenceEngine::SizeVector, param_size, param_size, param_size, uint32_t, uint32_t, PReLULayerDef>>::GetParam();
    auto extraLayerParams = std::get<6>(p);
    TEST_BODY;
    float maxerr = 0;
    if (group == 1)
        maxerr = 0.00055 * IC * kernel.x * kernel.y;
    else // TODO: currently dephConv is slightly less accurate
        maxerr = 0.00066 * (IC / group) * kernel.x * kernel.y;
    Compare(_outputMap.begin()->second, GenReferenceOutput(), maxerr);
}

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayerConvolutionWithPReLU_nightly,
        ::testing::Combine(
            ::testing::ValuesIn(g_convolutionTensors)
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<uint32_t>(16)
          , ::testing::Values<uint32_t>(1)
          , ::testing::ValuesIn(s_PReluLayerParams)
          )
);

INSTANTIATE_TEST_CASE_P(DISABLED_accuracy_postop, myriadLayerConvolutionWithPReLU_nightly,
        ::testing::Combine(
            ::testing::ValuesIn(g_poolingInput_postOp)
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1)/*, MAKE_STRUCT(param_size, 2, 2)*/)
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<uint32_t>(32)
          , ::testing::Values<uint32_t>(1)
          , ::testing::Values<PReLULayerDef>(MAKE_STRUCT(PReLULayerDef, {{{PRELU_PARAM, "0"}}}))
          )
);

