// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <../common_single_layer_tests/deconv_ref.hpp>
#include "myriad_layers_tests.hpp"

using std::tr1::tuple;
using std::tr1::get;

using namespace InferenceEngine;


static void refDeconvolution(const Blob::Ptr src, Blob::Ptr dst
        , const ie_fp16* weights_data, const ie_fp16* bias_data
        , param_size &kernel, param_size &stride, param_size &pad, size_t group) {
    conv_common_params params;
    params.kernel.insert(X_AXIS, kernel.x);
    params.kernel.insert(Y_AXIS, kernel.y);
    params.stride.insert(X_AXIS, stride.x);
    params.stride.insert(Y_AXIS, stride.y);
    params.pads_begin.insert(X_AXIS, pad.x);
    params.pads_begin.insert(Y_AXIS, pad.y);
    params.group = group;
    ref_deconv_common<ie_fp16>({ src }, *dst.get(), weights_data, 0, bias_data, 0, params);
}

PRETTY_PARAM(kernel, param_size)
PRETTY_PARAM(stride, param_size)
PRETTY_PARAM(pad, param_size)
PRETTY_PARAM(out_channels, int)
PRETTY_PARAM(group, int)
PRETTY_PARAM(layout, const char*)
PRETTY_PARAM(hw_optimization, bool)

typedef myriadLayerTestBaseWithParam<tuple<DimsInput, kernel, stride, pad
        , out_channels, group, layout, hw_optimization >> myriadLayerDeconvolution_nightly;


TEST_P(myriadLayerDeconvolution_nightly, Deconvolution) {

    tensor_test_params input_dims = get<0>(GetParam());
    param_size kernel = get<1>(GetParam());
    param_size stride = get<2>(GetParam());
    param_size pad = get<3>(GetParam());
    size_t out_channels = get<4>(GetParam());
    size_t group = get<5>(GetParam());
    const char* layout = get<6>(GetParam());
    bool hw_optimization = get<7>(GetParam());

    _config[VPU_CONFIG_KEY(COMPUTE_LAYOUT)] = layout;
    if (input_dims.n > 1)
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);
    else
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(YES);

    size_t out_w = stride.x * (input_dims.w - 1) + kernel.x - 2 * pad.x;
    size_t out_h = stride.y * (input_dims.h - 1) + kernel.y - 2 * pad.y;

    tensor_test_params output_dims = {input_dims.n, out_channels, out_h, out_w};

    SetInputTensor(input_dims);
    SetOutputTensor(output_dims);

    size_t num_weights = kernel.x * kernel.y * (input_dims.c / group) * output_dims.c;
    size_t num_bias = output_dims.c;

    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr =
            InferenceEngine::TBlob<uint8_t>::Ptr(GenWeights(num_weights + num_bias));
    ie_fp16* weights = weights_ptr->data().as<ie_fp16*>();
    ie_fp16* bias = weights + num_weights;

    std::map<std::string, std::string> layer_params = {
              {"kernel-x", std::to_string(kernel.x)}
            , {"kernel-y", std::to_string(kernel.y)}
            , {"stride-x", std::to_string(stride.x)}
            , {"stride-y", std::to_string(stride.y)}
            , {"pad-x", std::to_string(pad.x)}
            , {"pad-y", std::to_string(pad.y)}
            , {"output", std::to_string(out_channels)}
            , {"group", std::to_string(group)}
    };

    ASSERT_NO_FATAL_FAILURE(
        NetworkInit("Deconvolution",
            &layer_params,
            num_weights * sizeof(uint16_t),
            num_bias * sizeof(uint16_t),
            weights_ptr,
            InferenceEngine::Precision::FP16,
            InferenceEngine::Precision::FP16,
            hw_optimization));

    auto inputBlob = _inputMap.begin()->second;
    SetFirstInputToRange(-0.9f, 0.9f);

    ASSERT_TRUE(Infer());

    auto outputBlob = _outputMap.begin()->second;

    refDeconvolution(inputBlob, _refBlob, weights, bias, kernel, stride, pad, group);

    float maxerr = 0.00075 * (input_dims.c / group) * kernel.x * kernel.y;
    Compare(outputBlob, _refBlob, maxerr);
}

INSTANTIATE_TEST_CASE_P(accuracy_deconv_to_conv, myriadLayerDeconvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 6, 5, 6))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 1), MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 0), MAKE_STRUCT(param_size, 0, 1), MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<out_channels>(4)
          , ::testing::Values<group>(1)
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NCHW))
          , ::testing::Values<hw_optimization>(true)
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_deconv_to_conv_2, myriadLayerDeconvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 2, 256, 14, 14))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 2, 2), MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 0), MAKE_STRUCT(param_size, 0, 1), MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<out_channels>(256)
          , ::testing::Values<group>(1)
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NCHW))
          , ::testing::Values<hw_optimization>(true)
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_group, myriadLayerDeconvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 384, 4, 2))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 2, 2)
                                    , MAKE_STRUCT(param_size, 3, 3)
                                    , MAKE_STRUCT(param_size, 3, 4)
                                    , MAKE_STRUCT(param_size, 4, 4)
                                     )
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<out_channels>(384)
          , ::testing::Values<group>(2, 4)
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC), VPU_CONFIG_VALUE(NCHW))
          , ::testing::Values<hw_optimization>(false)
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_depthDeconv, myriadLayerDeconvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 384, 4, 2))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 2, 2)
                                    , MAKE_STRUCT(param_size, 3, 3)
                                    , MAKE_STRUCT(param_size, 3, 4)
                                    , MAKE_STRUCT(param_size, 4, 4)
                                     )
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<out_channels>(384)
          , ::testing::Values<group>(384)
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NCHW), VPU_CONFIG_VALUE(NHWC))
          , ::testing::Values<hw_optimization>(false)
          )
);

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayerDeconvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 2, 37, 59)
                                       , MAKE_STRUCT(tensor_test_params, 1, 21, 16, 16)
                                       , MAKE_STRUCT(tensor_test_params, 1, 512, 11, 13)
                                         )
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 2, 2)
                                    , MAKE_STRUCT(param_size, 3, 3)
                                    , MAKE_STRUCT(param_size, 4, 4)
                                    , MAKE_STRUCT(param_size, 5, 5)
                                      )
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)
                                    , MAKE_STRUCT(param_size, 2, 2)
                                      )
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0)
                                 , MAKE_STRUCT(param_size, 1, 1)
                                 , MAKE_STRUCT(param_size, 2, 2)
                                  )
          , ::testing::Values<out_channels>(1, 21)
          , ::testing::Values<group>(1)
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC))
          , ::testing::Values<hw_optimization>(false)
          )
);

INSTANTIATE_TEST_CASE_P(extra3x3s1, myriadLayerDeconvolution_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 256, 1, 1))
                              , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                              , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                              , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                              , ::testing::Values<out_channels>(256)
                              , ::testing::Values<group>(1)
                              , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC))
                              , ::testing::Values<hw_optimization>(false)
                              )
);
