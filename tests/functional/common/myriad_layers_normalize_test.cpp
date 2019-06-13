// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include "myriad_layers_tests.hpp"

#include <sstream>

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

using namespace InferenceEngine;

#define ERROR_BOUND 1e-3f

static void refNormalize(const Blob::Ptr src,
                         Blob::Ptr dst,
                         ie_fp16* weights_data,
                         int across_spatial,
                         int channel_shared,
                         float eps) {
    ASSERT_EQ(Layout::NHWC, src->layout());

    auto src_data = src->buffer().as<const uint16_t*>();
    auto dst_data = dst->buffer().as<uint16_t*>();

    auto N = src->dims()[3];
    auto C = src->dims()[2];
    auto H = src->dims()[1];
    auto W = src->dims()[0];

    for (size_t n = 0; n < N; ++n) {
        auto psrc = src_data + n * (C * H * W);
        auto pdst = dst_data + n * (C * H * W);

        if (across_spatial) {
            float norm = eps;
            for (size_t i = 0; i < C * H * W; ++i) {
                auto src_val = PrecisionUtils::f16tof32(psrc[i]);
                norm += src_val * src_val;
            }
            norm = 1.0f / std::sqrt(norm);

            for (size_t hw = 0; hw < H * W; ++hw) {
                for (size_t c = 0 ; c < C; ++c) {
                    auto ind = hw * C + c;

                    if (channel_shared) {
                        auto w = PrecisionUtils::f16tof32(weights_data[0]);
                        auto dst_val = PrecisionUtils::f16tof32(psrc[ind]) * norm * w;
                        pdst[ind] = PrecisionUtils::f32tof16(dst_val);
                    }
                    else {
                        auto w = PrecisionUtils::f16tof32(weights_data[c]);
                        auto dst_val = PrecisionUtils::f16tof32(psrc[ind]) * norm * w;
                        pdst[ind] = PrecisionUtils::f32tof16(dst_val);
                    }
                }
            }
        }
        else {
            for (int hw = 0; hw < H * W; ++hw) {
                float norm = eps;
                for (size_t c = 0; c < C; ++c) {
                    auto ind = hw * C + c;
                    auto src_val = PrecisionUtils::f16tof32(psrc[ind]);
                    norm += src_val * src_val;
                }
                norm = 1.0f / std::sqrt(norm);

                for (size_t c = 0; c < C; ++c) {
                    auto ind = hw * C + c;

                    if (channel_shared) {
                        auto w = PrecisionUtils::f16tof32(weights_data[0]);
                        auto dst_val = PrecisionUtils::f16tof32(psrc[ind]) * norm * w;
                        pdst[ind] = PrecisionUtils::f32tof16(dst_val);
                    }
                    else {
                        auto w = PrecisionUtils::f16tof32(weights_data[c]);
                        auto dst_val = PrecisionUtils::f16tof32(psrc[ind]) * norm * w;
                        pdst[ind] = PrecisionUtils::f32tof16(dst_val);
                    }
                }
            }
        }
    }
}

PRETTY_PARAM(AcrossSpatial, bool)
PRETTY_PARAM(ChannelShared, bool)
PRETTY_PARAM(EPS, float)

typedef myriadLayerTestBaseWithParam<std::tr1::tuple<Dims, AcrossSpatial, ChannelShared, EPS>> myriadLayersTestsNormalize_nightly;

TEST_P(myriadLayersTestsNormalize_nightly, Normalize) {
    tensor_test_params dims = std::tr1::get<0>(GetParam());
    int across_spatial = std::tr1::get<1>(GetParam());
    int channel_shared = std::tr1::get<2>(GetParam());
    float eps = std::tr1::get<3>(GetParam());

    SetInputTensor(dims);
    SetOutputTensor(dims);

    std::map<std::string, std::string> layer_params = {
        {"across_spatial",  std::to_string(across_spatial)},
        {"channel_shared",  std::to_string(channel_shared)},
        {"eps",             to_string_with_precision(eps, 10)}
    };

    size_t num_weights = 0;
    if (channel_shared) {
        num_weights = 1;
    }
    else {
        num_weights = dims.c;
    }
    TBlob<uint8_t>::Ptr weights(GenWeights(num_weights));

    NetworkInit("Normalize",
                    &layer_params,
                    weights->byteSize(),
                    0, // biases_size
                    weights,
                    InferenceEngine::Precision::FP16, // output precision
                    InferenceEngine::Precision::FP16  // input precision
                    );
    ASSERT_TRUE(Infer());

    auto src = _inputMap.begin()->second;
    auto dst = _outputMap.begin()->second;
    auto weights_data = weights->data().as<ie_fp16*>();

    refNormalize(src, _refBlob, weights_data, across_spatial, channel_shared, eps);

    Compare(dst, _refBlob, ERROR_BOUND);
}

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsNormalize_nightly, ::testing::Combine(
    ::testing::Values<Dims>(
        // small size, num_channels is not divisible by 8
        MAKE_STRUCT(tensor_test_params, 1, 33, 1, 1),

        // size used in SSD_VGG topology
        MAKE_STRUCT(tensor_test_params, 1, 512, 38, 38),

        // size used in a customer topology
        MAKE_STRUCT(tensor_test_params, 1, 128, 1, 1)
    ),
    ::testing::Values<AcrossSpatial>(false, true),
    ::testing::Values<ChannelShared>(false, true),
    ::testing::Values<EPS>(1e-10f, 1e-9f, 1e-8f, 1e-7f, 1.192093e-07, 1e-6f, 1e-5f, 1e-4f, 1e-3f, 0.0f)
));


INSTANTIATE_TEST_CASE_P(DISABLED_accuracy, myriadLayersTestsNormalize_nightly, ::testing::Combine(
    ::testing::Values<Dims>(
        //more tests
        MAKE_STRUCT(tensor_test_params, 1, 1, 38, 38),
        MAKE_STRUCT(tensor_test_params, 1, 1, 1, 1),
        MAKE_STRUCT(tensor_test_params, 1, 1, 8, 8),
        MAKE_STRUCT(tensor_test_params, 1, 3, 17, 17),
        MAKE_STRUCT(tensor_test_params, 1, 1, 17, 17),
        MAKE_STRUCT(tensor_test_params, 1, 1, 32, 32),
        MAKE_STRUCT(tensor_test_params, 1, 8, 38, 38),
        MAKE_STRUCT(tensor_test_params, 1, 512, 1, 1),
        MAKE_STRUCT(tensor_test_params, 1, 512, 8, 8)
    ),
    ::testing::Values<AcrossSpatial>(false, true),
    ::testing::Values<ChannelShared>(false, true),
    ::testing::Values<EPS>(1e-10f, 1e-9f, 1e-8f, 1e-7f, 1.192093e-07, 1e-6f, 1e-5f, 1e-4f, 1e-3f, 0.0f)
));
