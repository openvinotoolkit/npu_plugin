// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"

using namespace InferenceEngine;

#define ERROR_BOUND 1e-3f

static void refGRN(const Blob::Ptr src,
                         Blob::Ptr dst,
                   float bias, int isCHW) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    const uint16_t *src_data = src->buffer();
          uint16_t *dst_data = dst->buffer();
    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(dst_data, nullptr);
    int32_t IW = 1;
    int32_t IH = 1;
    int32_t IC = 1;
    get_dims(src, IW, IH, IC);
    for (uint32_t h = 0; h < IH; h++) {
        for (uint32_t w = 0; w < IW; w++) {
            float variance = 1e-9f;
            for (uint32_t c = 0; c < IC; c++) {
                int ind = isCHW ? c*IH*IW + h*IW + w : h*IW*IC + w*IC + c;
                float s = PrecisionUtils::f16tof32(src_data[ind]);
                variance += powf(s, 2);
            }
            variance = sqrtf(variance + bias);
            for (uint32_t c = 0; c < IC; c++) {
                int ind = isCHW ? c*IH*IW + h*IW + w : h*IW*IC + w*IC + c;

                float s = PrecisionUtils::f16tof32(src_data[ind]);
                float result = s / variance;

                dst_data[ind] = PrecisionUtils::f32tof16(result);
            }
        }
    }
}

PRETTY_PARAM(Bias, float)

typedef myriadLayerTestBaseWithParam<std::tuple<Dims, Bias, std::string>> myriadLayersTestsGRN_nightly;

TEST_P(myriadLayersTestsGRN_nightly, GRN) {
    tensor_test_params dims  = std::get<0>(GetParam());
    float bias               = std::get<1>(GetParam());
    std::string customConfig = std::get<2>(GetParam());

    if(!customConfig.empty() && !CheckMyriadX()) {
        GTEST_SKIP()<<"Custom layers for MYRIAD2 not supported";
    }
    _config[VPU_CONFIG_KEY(CUSTOM_LAYERS)] = customConfig;

    SetInputTensor(dims);
    SetOutputTensor(dims);

    std::map<std::string, std::string> params;
    params["bias"] = std::to_string(bias);

    ASSERT_NO_FATAL_FAILURE(NetworkInit("GRN", &params, 0, 0, nullptr, Precision::FP16, Precision::FP16));

    ASSERT_TRUE(Infer());

    ASSERT_NO_FATAL_FAILURE(refGRN(_inputMap.begin()->second, _refBlob, bias, false));

    Compare(_outputMap.begin()->second, _refBlob, ERROR_BOUND);
}

static std::vector<Dims> s_GRNTensors = {
        {{1, 3, 16, 224}},
        {{1, 24, 128, 224}},
};

static std::vector<Bias> s_GRN_bias = {
        0.5f, 10.f
};

static std::vector<std::string> s_MVNCustomConfig = {
        {"" , TestsCommon::get_data_path() + "/vpu/mvcl/customLayers_m7.xml"}
};

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsGRN_nightly,
        ::testing::Combine(
        ::testing::ValuesIn(s_GRNTensors),
        ::testing::ValuesIn(s_GRN_bias),
        ::testing::ValuesIn(s_MVNCustomConfig)));


TEST_F(myriadLayersTests_nightly, GRN_CHW_Input)
{
    std::string model = R"V0G0N(
        <net name="GRN" version="2" batch="1">
            <layers>
                <layer name="data" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>24</dim>
                            <dim>128</dim>
                            <dim>224</dim>
                        </port>
                    </output>
                </layer>
                <layer name="grn" type="GRN" precision="FP16" id="2">
                    <data bias="0.5"/>
                    <input>
                        <port id="2">
                            <dim>1</dim>
                            <dim>24</dim>
                            <dim>128</dim>
                            <dim>224</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>24</dim>
                            <dim>128</dim>
                            <dim>224</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="1" to-layer="2" to-port="2"/>
            </edges>
        </net>
    )V0G0N";

    StatusCode st;

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());

    auto network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["data"]->setInputPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["grn"]->setPrecision(Precision::FP16);

    ASSERT_NO_THROW(st = myriadPluginPtr->LoadNetwork(_exeNetwork, network,
                                                      {{VPU_CONFIG_KEY(COMPUTE_LAYOUT), VPU_CONFIG_VALUE(NCHW)}}, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    ASSERT_NE(_exeNetwork, nullptr) << _resp.msg;

    ASSERT_NO_THROW(st = _exeNetwork->CreateInferRequest(_inferRequest, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    auto dims = _inputsInfo["data"]->getDims();

    auto inputNCHW = make_shared_blob<ie_fp16, const SizeVector>(Precision::FP16, Layout::NCHW, dims);
    ASSERT_NO_THROW(inputNCHW->allocate());

    auto outputNCHW = make_shared_blob<ie_fp16, const SizeVector>(Precision::FP16, Layout::NCHW, dims);
    ASSERT_NO_THROW(outputNCHW->allocate());

    auto output_ref = make_shared_blob<ie_fp16, const SizeVector>(Precision::FP16, Layout::NCHW, dims);
    ASSERT_NO_THROW(output_ref->allocate());

    ASSERT_NO_THROW(GenRandomData(inputNCHW));

    ASSERT_NO_THROW(st = _inferRequest->SetBlob("data", inputNCHW, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = _inferRequest->SetBlob("grn", outputNCHW, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = _inferRequest->Infer(&_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_FATAL_FAILURE(refGRN(inputNCHW, output_ref, 0.5f, true));

    Compare(outputNCHW, output_ref, 0.003);
}
