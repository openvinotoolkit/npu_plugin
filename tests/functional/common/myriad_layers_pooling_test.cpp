// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"


#define ERROR_BOUND (1.2e-2f)

using namespace InferenceEngine;

extern const char POOLING_MAX[] = "max";
extern const char POOLING_AVG[] = "avg";


class myriadLayersTestsMax_nightly: public PoolingTest<POOLING_MAX>
{
};

class myriadLayersTestsMaxPad4_nightly: public PoolingTestPad4<POOLING_MAX>
{
};

class myriadLayersTestsGlobalMax_nightly: public GlobalPoolingTest<POOLING_MAX>
{
};

class myriadLayersTestsAvg_nightly: public PoolingTest<POOLING_AVG>
{
};

class myriadLayersTestsAvgPad4_nightly: public PoolingTestPad4<POOLING_AVG>
{
};

class myriadLayersTestsGlobalAvg_nightly: public GlobalPoolingTest<POOLING_AVG>
{
};

/* IR version 3 tests, main difference is a changes in padding parameters definitions */
/*                   input tensor,               kernel,     stride,    pads_begin, pads_end,  auto_pad,     exclude_pad  method */
typedef std::tuple<InferenceEngine::SizeVector, param_size, param_size, param_size, param_size, const char*, const char*, const char*> IR3_PoolParams;

class myriadLayers_IR3_PoolingTests_nightly: public myriadLayersTests_nightly, /*input tensor, kernel, stride, pads_begin, pads_end, out_channel, group */
                                     public testing::WithParamInterface<IR3_PoolParams> {
};

static void genTestData(InferenceEngine::Blob::Ptr blob) {
    ASSERT_NE(blob, nullptr);
    Layout layout = blob->layout();
    SizeVector dims = blob->getTensorDesc().getDims();

    ie_fp16* ptr = blob->buffer().as<ie_fp16*>();
    if (layout == NCHW || layout == NHWC) {
        size_t N = dims[0];
        size_t C = dims[1];
        size_t H = dims[2];
        size_t W = dims[3];

        float counter = 0.125f;
        for (size_t n = 0; n < N; n++) {
            for (size_t c = 0; c < C; c++) {
                for (size_t h = 0; h < H; h++) {
                    for (size_t w = 0; w < W; w++) {
                        size_t actualIdx = layout == NCHW ?
                                           w + h * W + c * W * H + n * W * H * C : c + w * C + h * C * W +
                                                                                   n * W * H * C;
                        ptr[actualIdx] = PrecisionUtils::f32tof16(counter);
                        counter += 0.025f;
                        if (counter > 5.0f) {
                            counter = -0.5f;
                        }
                    }
                }
            }
        }
    } else {
        ASSERT_TRUE(false);
    }
}


TEST_P(myriadLayers_IR3_PoolingTests_nightly, Pooling) {
    std::map<std::string, std::string> params;
    InferenceEngine::SizeVector output_tensor;
    int32_t IW = 0;
    int32_t IH = 0;
    int32_t IC = 0;
    int32_t I_N = 0;
    size_t  group = 0;
    /*input tensor,               kernel,     stride,    pads_begin, pads_end,  auto_pad,     exclude_pad  method */
    auto p = ::testing::WithParamInterface<IR3_PoolParams>::GetParam();
    auto input_tensor       = std::get<0>(p);
    param_size kernel       = std::get<1>(p);
    param_size stride       = std::get<2>(p);
    param_size pads_begin   = std::get<3>(p);
    param_size pads_end     = std::get<4>(p);
    const char* auto_pad    = std::get<5>(p);
    const std::string exclude_pad = std::get<6>(p);
    const std::string method      = std::get<7>(p);

    get_dims(input_tensor, IW, IH, IC, I_N);
    if (I_N > 1)
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);
    else
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(YES);
    int32_t OW = 1;
    int32_t OH = 1;
    int32_t OC = 1;
    int32_t ON = 1;
    if (strncmp(auto_pad, "same_upper", strlen(auto_pad)) == 0) {
        OW = input_tensor[3]/2;
        OH = input_tensor[2]/2;
        OC = input_tensor[1];
        ON = input_tensor[0];
    } else {
        ASSERT_TRUE(false);
    }
    if (kernel.x == 4 && kernel.y == 4) {
        /* particular case  for Faster-RCNN */
        OW = input_tensor[3] / kernel.x;
        OH = input_tensor[2] / kernel.y;
        OC = input_tensor[1];
        ON = input_tensor[0];
    }

    gen_dims(output_tensor, input_tensor.size(), OW, OH, OC, ON);

    std::string padsB   = gen_param(pads_begin);
    std::string padsE   = gen_param(pads_end);
    std::string strides = gen_param(stride);
    std::string kern    = gen_param(kernel);

    std::map<std::string, std::string> layer_params = {
              {"kernel",      kern}
            , {"strides",     strides}
            , {"pads_begin",  padsB}
            , {"pads_end",    padsE}
            , {"auto_pad",    auto_pad}
            , {"exclude_pad", exclude_pad}
            , {"pool-method",      method}
    };
    if (kernel.x == 4 && kernel.y == 4) {
        layer_params.erase("auto_pad");
        layer_params["rounding-type"] = "ceil";
    }
    _genDataCallback = genTestData;
    /*
    */
    AddLayer("Pooling",
             &layer_params,
             {input_tensor},
             {output_tensor},
             ref_pooling_wrap);
    ASSERT_TRUE(GenerateNetAndInfer(CheckMyriadX(), true, 3));
    float maxerr = 0.0001f;
    Compare(_outputMap.begin()->second, GenReferenceOutput(), maxerr);
}

class myriadLayers_IR3_BatchPoolingTests_nightly: public myriadLayersTests_nightly, /*input tensor, kernel, stride, pads_begin, pads_end, out_channel, group */
                                     public testing::WithParamInterface<IR3_PoolParams> {
};

TEST_P(myriadLayers_IR3_BatchPoolingTests_nightly, Pooling) {
    std::map<std::string, std::string> params;
    InferenceEngine::SizeVector output_tensor;
    int32_t IW = 0;
    int32_t IH = 0;
    int32_t IC = 0;
    int32_t I_N = 0;
    size_t  group = 0;
    /*input tensor,               kernel,     stride,    pads_begin, pads_end,  auto_pad,     exclude_pad  method */
    auto p = ::testing::WithParamInterface<IR3_PoolParams>::GetParam();
    auto input_tensor       = std::get<0>(p);
    param_size kernel       = std::get<1>(p);
    param_size stride       = std::get<2>(p);
    param_size pads_begin   = std::get<3>(p);
    param_size pads_end     = std::get<4>(p);
    const char* auto_pad    = std::get<5>(p);
    const std::string exclude_pad = std::get<6>(p);
    const std::string method      = std::get<7>(p);

    get_dims(input_tensor, IW, IH, IC, I_N);
    if (I_N > 1)
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);
    else
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(YES);
    int32_t OW = 1;
    int32_t OH = 1;
    int32_t OC = 1;
    int32_t ON = 1;
    if (strncmp(auto_pad, "same_upper", strlen(auto_pad)) == 0) {
        OW = input_tensor[3]/2;
        OH = input_tensor[2]/2;
        OC = input_tensor[1];
        ON = input_tensor[0];
    }
    gen_dims(output_tensor, input_tensor.size(), OW, OH, OC, ON);

    std::string padsB   = gen_param(pads_begin);
    std::string padsE   = gen_param(pads_end);
    std::string strides = gen_param(stride);
    std::string kern    = gen_param(kernel);

    std::map<std::string, std::string> layer_params = {
              {"kernel",      kern}
            , {"strides",     strides}
            , {"pads_begin",  padsB}
            , {"pads_end",    padsE}
            , {"auto_pad",    auto_pad}
            , {"exclude_pad", exclude_pad}
            , {"pool-method",      method}
    };
    _genDataCallback = genTestData;
    /*
    */
    AddLayer("Pooling",
             &layer_params,
             {input_tensor},
             {output_tensor},
             ref_pooling_wrap);
    ASSERT_TRUE(GenerateNetAndInfer(CheckMyriadX(), true, 3));
    float maxerr = 0.0001f;
    Compare(_outputMap.begin()->second, GenReferenceOutput(), maxerr);
}

static const std::vector<const char*> s_poolingAutoPad = {
    "same_upper"
};

static const std::vector<const char*> s_poolingExcludePad = {
    "true"
};

static const std::vector<const char*> s_poolingMethod = {
    "max"
};

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayers_IR3_BatchPoolingTests_nightly,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({10, 192, 56, 56})
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 3, 3)) /* kernel     */
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 2, 2)) /* stride     */
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 0, 0)) /* pads_begin */
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1)) /* pads_end   */
          , ::testing::ValuesIn(s_poolingAutoPad)
          , ::testing::ValuesIn(s_poolingExcludePad)
          , ::testing::ValuesIn(s_poolingMethod)
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_1, myriadLayers_IR3_BatchPoolingTests_nightly,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({10, 576, 14, 14})
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 2, 2)) /* kernel     */
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 2, 2)) /* stride     */
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 0, 0)) /* pads_begin */
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 0, 0)) /* pads_end   */
          , ::testing::ValuesIn(s_poolingAutoPad)
          , ::testing::ValuesIn(s_poolingExcludePad)
          , ::testing::ValuesIn(s_poolingMethod)
          )
);


INSTANTIATE_TEST_CASE_P(accuracy_4X4, myriadLayers_IR3_PoolingTests_nightly,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({10, 1024, 4, 4})
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 4, 4)) /* kernel     */
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1)) /* stride     */
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 0, 0)) /* pads_begin */
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 0, 0)) /* pads_end   */
          , ::testing::ValuesIn(s_poolingAutoPad)
          , ::testing::ValuesIn(s_poolingExcludePad)
          , ::testing::ValuesIn(s_poolingMethod)
          )
);

TEST_P(myriadLayersTestsMax_nightly, MaxPooling)
{
    ASSERT_TRUE(GenerateNetAndInfer());
    Compare(_outputMap.begin()->second, GenReferenceOutput(), ERROR_BOUND);
}

TEST_P(myriadLayersTestsMaxPad4_nightly, MaxPoolingPad4)
{
    ASSERT_TRUE(GenerateNetAndInfer());
    auto refBlob = GenReferenceOutput();
    Compare(_outputMap.begin()->second, refBlob, ERROR_BOUND);
}

TEST_P(myriadLayersTestsAvg_nightly, AvgPooling)
{
    ASSERT_TRUE(GenerateNetAndInfer());
    Compare(_outputMap.begin()->second, GenReferenceOutput(), ERROR_BOUND);
}

TEST_P(myriadLayersTestsAvgPad4_nightly, AvgPoolingPad4)
{
    ASSERT_TRUE(GenerateNetAndInfer());
    auto refBlob = GenReferenceOutput();
    Compare(_outputMap.begin()->second, refBlob, ERROR_BOUND);
}

TEST_P(myriadLayersTestsGlobalMax_nightly, GlobalMaxPooling)
{
    ASSERT_TRUE(GenerateNetAndInfer());
    auto refBlob = GenReferenceOutput();
    Compare(_outputMap.begin()->second, refBlob, ERROR_BOUND);
}

TEST_P(myriadLayersTestsGlobalAvg_nightly, GlobalAvgPooling)
{
    ASSERT_TRUE(GenerateNetAndInfer());
    auto refBlob = GenReferenceOutput();
    Compare(_outputMap.begin()->second, refBlob, ERROR_BOUND);
}

static std::vector<pooling_layer_params> s_poolingLayerParams_k3x3 = {
    {{3, 3}, {1, 1}, {1, 1}},
};

const std::vector<InferenceEngine::SizeVector> g_poolingInputPad4 = {
    {{1, 3,  224,  224}}
};

const std::vector<param_size> g_poolingKernelPad4 = {
    {4, 4},
    {6, 6},
    {8, 8},
};

const std::vector<param_size> g_poolingStridePad4 = {
    {1, 1},
};

const std::vector<paddings4> g_poolingPad4 = {
    {0, 0, 2, 0},
    {1, 2, 3, 2},
    {2, 2, 0, 0},
};

const std::vector<InferenceEngine::SizeVector> g_GlobalPoolingInput = {
#if 0 // temporary OFF because of HACKS for rfcn #ifdef MORE_DIMENSIONS // 4DGP
    {{2,  8,    7,  7}},
#endif
    {{1, 1024, 64, 32}},
    {{1, 2048,  8,  8}},
    {{1, 2048,  7,  7}},
    {{1, 1000, 15, 15}},
    {{1, 1000, 14, 14}},
    {{1, 1000, 12, 12}},
    {{1,  8,    7,  7}},
    {{1,  2,    7,  7}},
    {{1,  8,    7,  7}},
    {{1,  1000, 2,  3}},
};

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsMax_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(g_poolingInput),
        ::testing::ValuesIn(g_poolingLayerParamsFull),
        ::testing::ValuesIn(g_poolingLayout))
);

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsMaxPad4_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(g_poolingInputPad4),
        ::testing::ValuesIn(g_poolingKernelPad4),
        ::testing::ValuesIn(g_poolingStridePad4),
        ::testing::ValuesIn(g_poolingPad4),
        ::testing::ValuesIn(g_poolingLayout))
);

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsAvgPad4_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(g_poolingInputPad4),
        ::testing::ValuesIn(g_poolingKernelPad4),
        ::testing::ValuesIn(g_poolingStridePad4),
        ::testing::ValuesIn(g_poolingPad4),
        ::testing::ValuesIn(g_poolingLayout))
);

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsGlobalMax_nightly,
        ::testing::ValuesIn(g_GlobalPoolingInput ));

INSTANTIATE_TEST_CASE_P(accuracy_3x3, myriadLayersTestsMax_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(g_poolingInput),
        ::testing::ValuesIn(s_poolingLayerParams_k3x3),
        ::testing::ValuesIn(g_poolingLayout))
);

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsAvg_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(g_poolingInput),
        ::testing::ValuesIn(g_poolingLayerParamsFull),
        ::testing::ValuesIn(g_poolingLayout))
);

INSTANTIATE_TEST_CASE_P(accuracy_3x3, myriadLayersTestsAvg_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(g_poolingInput),
        ::testing::ValuesIn(s_poolingLayerParams_k3x3),
        ::testing::ValuesIn(g_poolingLayout))
);

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsGlobalAvg_nightly,
        ::testing::ValuesIn(g_GlobalPoolingInput));
