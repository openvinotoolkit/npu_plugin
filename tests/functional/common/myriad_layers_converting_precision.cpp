// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <inference_engine/blob_factory.hpp>
#include "myriad_layers_tests.hpp"

#define ERROR_BOUND (1.e-3f)

using namespace InferenceEngine;

void ref_convertPrecision(const InferenceEngine::Blob::Ptr &src, InferenceEngine::Blob::Ptr &dst)
{
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    ASSERT_TRUE(((src->precision() == InferenceEngine::Precision::U8 || src->precision() == InferenceEngine::Precision::FP32)
                && dst->precision() == InferenceEngine::Precision::FP16)
                || (src->precision() == InferenceEngine::Precision::FP16 && dst->precision() == InferenceEngine::Precision::FP32));

    uint16_t *dst_data = dst->buffer();
    for (size_t i = 0; i < dst->size(); i++) {
        if (src->precision() == Precision::U8 && dst->precision() == Precision::FP16) {
            dst->buffer().as<int16_t *>()[i] = PrecisionUtils::f32tof16(src->cbuffer().as<uint8_t *>()[i] * 1.f);
        } else if (src->precision() == Precision::FP32 && dst->precision() == Precision::FP16) {
            dst->buffer().as<int16_t *>()[i] = PrecisionUtils::f32tof16(src->cbuffer().as<float *>()[i] * 1.f);
        } else if (src->precision() == Precision::FP16 && dst->precision() == Precision::FP32) {
            dst->buffer().as<float *>()[i] = PrecisionUtils::f16tof32(src->cbuffer().as<int16_t *>()[i] * 1.f);
        } else {
            THROW_IE_EXCEPTION << "Unsupported input or output precision";
        }
    }
}

typedef std::tuple<tensor_test_params, std::pair<Precision, Precision>> TestParam;

class myriadLayersTestsConvertPrecision_nightly: public myriadLayersTests_nightly,
                           public testing::WithParamInterface<TestParam> {
};

TEST_P(myriadLayersTestsConvertPrecision_nightly, TestsConvertPrecision)
{
    TestParam param = ::testing::WithParamInterface<TestParam>::GetParam();
    tensor_test_params inputDims = std::get<0>(param);
    std::pair<Precision, Precision> precisions = std::get<1>(param);
    Precision inputPrecision = precisions.first, outputPrecision = precisions.second;

    SetInputTensors({{inputDims.n, inputDims.c, inputDims.h, inputDims.w}});
    SetOutputTensors({{inputDims.n, inputDims.c, inputDims.h, inputDims.w}});

    NetworkInit("Copy",
                nullptr,
                0,
                0,
                nullptr,
                outputPrecision,
                inputPrecision);
    ASSERT_TRUE(Infer());

    InferenceEngine::Blob::Ptr refBlob = make_blob_with_precision(outputPrecision, _outputMap.begin()->second->getTensorDesc().getLayout(), _outputMap.begin()->second->dims());
    refBlob->allocate();
    ref_convertPrecision(_inputMap.begin()->second, refBlob);

    Compare(_outputMap.begin()->second, refBlob, ERROR_BOUND);
}

std::vector<tensor_test_params> inputsDims = {{1,1,224,224}, {1, 1, 416, 416}, {1, 1, 62, 62}, {1, 1, 227, 227}, {1,3,224,224},};
std::vector<std::pair<Precision, Precision>> precisions = {{Precision::U8, Precision::FP16},
                                                           {Precision::FP32, Precision::FP16},
                                                           {Precision::FP16, Precision::FP32}};
INSTANTIATE_TEST_CASE_P(
        accuracy, myriadLayersTestsConvertPrecision_nightly,
        ::testing::Combine(::testing::ValuesIn(inputsDims), ::testing::ValuesIn(precisions)));
