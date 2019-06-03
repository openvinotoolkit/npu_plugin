// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "myriad_layers_tests.hpp"

using std::tr1::tuple;
using std::tr1::get;

using namespace InferenceEngine;

static void reorg_calculate(short *inp, int w, int h, int c, int batch, int stride, float *out)
{
    int out_c = c / (stride*stride);

    int oc = c * (stride*stride);
    int oh = h / stride;
    int ow = w / stride;

    for(int b = 0; b < batch; ++b)
    {
        for(int k = 0; k < c; ++k)
        {
            for(int j = 0; j < h; ++j)
            {
                for(int i = 0; i < w; ++i)
                {
                    int in_index = i + w * (j + h * (k + c * b));

                    int new_z = in_index / (oh*ow);
                    int new_y = (in_index %(oh*ow)) / ow;
                    int new_x = (in_index %(oh*ow)) % ow;
                    int new_index = new_z + new_x * oc + new_y * oc * ow;

                    int c2 = k % out_c;
                    int offset = k / out_c;
                    int w2 = i*stride + offset % stride;
                    int h2 = j*stride + offset / stride;
                    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));

                    out[new_index] = PrecisionUtils::f16tof32(inp[out_index]);
                }
            }
        }
    }
}

PRETTY_PARAM(Stride, int);
PRETTY_PARAM(ScaleOutput, int);


typedef myriadLayerTestBaseWithParam<tuple<DimsInput, ScaleOutput, Stride, std::string, std::string >> myriadLayersTestsReorg_nightly;

TEST_P(myriadLayersTestsReorg_nightly, TestsReorg) {

    // TODO: M2 mode is not working for OpenCL compiler
    if(!get<4>(GetParam()).empty() && !CheckMyriadX()) {
        GTEST_SKIP()<<"Custom layers for MYRIAD2 not supported";
    }

    tensor_test_params dimsInput = get<0>(GetParam());

    int scaleOutput = get<1>(GetParam());
    tensor_test_params dimsOutput = {dimsInput.n, dimsInput.c * (scaleOutput * scaleOutput), dimsInput.h / scaleOutput, dimsInput.w / scaleOutput};

    int stride = get<2>(GetParam());
    std::string layout = get<3>(GetParam());
    std::map<std::string, std::string> params;
    std::string type =  "ReorgYolo";

    params["stride"] = std::to_string(stride);
    SetInputTensor(dimsInput);
    SetOutputTensor(dimsOutput);
    _config[VPU_CONFIG_KEY(COMPUTE_LAYOUT)] = layout;
    _config[VPU_CONFIG_KEY(CUSTOM_LAYERS)] = get<4>(GetParam());
    NetworkInit(type, &params);
    /* input data preparation */
    SetInputInOrder();
    ASSERT_TRUE(Infer());
    InferenceEngine::SizeVector inputDims = _inputsInfo.begin()->second->getDims();
    InferenceEngine::Blob::Ptr inputBlobRef =
            InferenceEngine::make_shared_blob<short, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP16, InferenceEngine::NHWC, inputDims);
    inputBlobRef->allocate();
    short *inputBlobRefRawData = inputBlobRef->buffer();

    int c = inputDims[2];
    int h = inputDims[1];
    int w = inputDims[0];

    auto inputBlob =_inputMap[_inputsInfo.begin()->first];
    short * inputBlob_data = inputBlob->buffer();

    /* Preliminary repacking */
    for(int k = 0; k < c; k++)
    {
        for(int j = 0; j < h; j++)
        {
            for(int i = 0; i < w; i++)
            {
                int dst_index = i + w * j + w * h * k;
                int src_index = k + c * i + c * w * j;

                inputBlobRefRawData[dst_index] = inputBlob_data[src_index];
            }
        }
    }

    auto outputBlob =_outputMap[_outputsInfo.begin()->first];
    InferenceEngine::SizeVector outputDims = _outputsInfo.begin()->second->dims;

    InferenceEngine::TBlob<float>::Ptr outputBlobRef =
                InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, InferenceEngine::NCHW, outputDims);
    outputBlobRef->allocate();
    float *outputBlobRefRawData = outputBlobRef->buffer();

    reorg_calculate(inputBlobRefRawData, w, h, c, 1, stride, outputBlobRefRawData);

    compare(outputBlob->buffer(), outputBlobRef->buffer(), outputBlob->size(), 0.0);
}


INSTANTIATE_TEST_CASE_P(
        accuracy, myriadLayersTestsReorg_nightly,
        ::testing::Combine(
                            ::testing::Values<DimsInput >(MAKE_STRUCT(tensor_test_params, 1, 64, 26, 26),
                                                          MAKE_STRUCT(tensor_test_params, 1,  4,  6,  6)
                                                         ),
                            ::testing::Values<ScaleOutput>(2),
                            ::testing::Values<Stride>(2),
                            ::testing::Values<std::string>(VPU_CONFIG_VALUE(NHWC),
                                                           VPU_CONFIG_VALUE(NCHW)),
                            ::testing::Values<std::string>("", TestsCommon::get_data_path() + "/vpu/mvcl/customLayers_m7.xml")
                          )
                       );
