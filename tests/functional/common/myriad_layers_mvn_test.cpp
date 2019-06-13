// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"

using namespace InferenceEngine;

#define ERROR_BOUND 0.2f

static void refMVN(const Blob::Ptr src,
                   Blob::Ptr dst,
                   int across_channels, int normalize_variance, int isCHW) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    const uint16_t *src_data = src->buffer();
    uint16_t *dst_data = dst->buffer();
    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(dst_data, nullptr);

    int32_t IW = 1;
    int32_t IH = 1;
    int32_t IC = 1;
    int32_t IB = 1;
    get_dims(src, IW, IH, IC);

    const float esp = 1e-9f;

    float* mean_buf = new float[IW*IH*IC];

    for (int b = 0; b < IB; b++)
    {
        // Calculate mean value
        if (across_channels)
        {
            float mean = 0;
            for (int c = 0; c < IC; c++) {
                for (int h = 0; h < IH; h++) {
                    for (int w = 0; w < IW; w++) {
                        int ind = isCHW ? c*IH*IW + h*IW + w : h*IW*IC + w*IC + c;
                        float s = PrecisionUtils::f16tof32(src_data[ind]);
                        mean += s;
                    }
                }
            }
            mean /= IC*IH*IW;
            for (int c = 0; c < IC; c++) {
                for (int h = 0; h < IH; h++) {
                    for (int w = 0; w < IW; w++) {
                        int ind = isCHW ? c*IH*IW + h*IW + w : h*IW*IC + w*IC + c;
                        float s = PrecisionUtils::f16tof32(src_data[ind]);
                        mean_buf[ind] = s - mean;
                        dst_data[ind] = PrecisionUtils::f32tof16(s - mean);
                    }
                }
            }
        }
        else {
            for (int c = 0; c < IC; c++)
            {
                float mean = 0;
                for (int h = 0; h < IH; h++) {
                    for (int w = 0; w < IW; w++) {
                        int ind = isCHW ? c*IH*IW + h*IW + w : h*IW*IC + w*IC + c;
                        float s = PrecisionUtils::f16tof32(src_data[ind]);
                        mean += s;
                    }
                }
                mean /= IH*IW;
                for (int h = 0; h < IH; h++) {
                    for (int w = 0; w < IW; w++) {
                        int ind = isCHW ? c*IH*IW + h*IW + w : h*IW*IC + w*IC + c;
                        float s = PrecisionUtils::f16tof32(src_data[ind]);
                        mean_buf[ind] = s - mean;
                        dst_data[ind] = PrecisionUtils::f32tof16(s - mean);
                    }
                }
            }
        }
    }

    if (normalize_variance)
    {
        for (int b = 0; b < IB; b++)
        {
            // Calculate variances value
            if (across_channels)
            {
                float variance = 0;
                for (int c = 0; c < IC; c++) {
                    for (int h = 0; h < IH; h++) {
                        for (int w = 0; w < IW; w++) {
                            int ind = isCHW ? c*IH*IW + h*IW + w : h*IW*IC + w*IC + c;
                            variance += mean_buf[ind] * mean_buf[ind];
                        }
                    }
                }
                variance /= IC*IH*IW;
                variance = sqrtf(variance);//std::pow(variance, 0.5f);
                variance += esp;
                for (int c = 0; c < IC; c++) {
                    for (int h = 0; h < IH; h++) {
                        for (int w = 0; w < IW; w++) {
                            int ind = isCHW ? c*IH*IW + h*IW + w : h*IW*IC + w*IC + c;
                            dst_data[ind] = PrecisionUtils::f32tof16(mean_buf[ind] / variance);
                        }
                    }
                }
            }
            else {
                for (int c = 0; c < IC; c++)
                {
                    float variance = 0;
                    for (int h = 0; h < IH; h++) {
                        for (int w = 0; w < IW; w++) {
                            int ind = isCHW ? c*IH*IW + h*IW + w : h*IW*IC + w*IC + c;
                            variance += mean_buf[ind] * mean_buf[ind];
                        }
                    }
                    variance /= IH*IW;
                    variance = sqrtf(variance);
                    variance += esp;
                    for (int h = 0; h < IH; h++) {
                        for (int w = 0; w < IW; w++) {
                            int ind = isCHW ? c*IH*IW + h*IW + w : h*IW*IC + w*IC + c;
                            dst_data[ind] = PrecisionUtils::f32tof16(mean_buf[ind] / variance);
                        }
                    }
                }
            }
        }
    }

    delete[] mean_buf;
}

PRETTY_PARAM(AcrossChannels, int)
PRETTY_PARAM(Normalize, int)

typedef myriadLayerTestBaseWithParam<std::tuple<Dims, AcrossChannels, Normalize, std::string>> myriadLayersTestsMVN_nightly;

TEST_P(myriadLayersTestsMVN_nightly, MVN)
{
    tensor_test_params dims  = std::get<0>(GetParam());
    int acrossChannels       = std::get<1>(GetParam());
    int normalize            = std::get<2>(GetParam());
    std::string customConfig = std::get<3>(GetParam());

    if(!customConfig.empty() && !CheckMyriadX()) {
        GTEST_SKIP()<<"Custom layers for MYRIAD2 not supported";
    }
    _config[VPU_CONFIG_KEY(CUSTOM_LAYERS)] = customConfig;

    SetInputTensor(dims);
    SetOutputTensor(dims);

    std::map<std::string, std::string> params;
    params["across_channels"] = std::to_string(acrossChannels);
    params["normalize_variance"] = std::to_string(normalize);

    ASSERT_NO_FATAL_FAILURE(NetworkInit("MVN", &params, 0, 0, nullptr, Precision::FP16, Precision::FP16));
    ASSERT_NO_FATAL_FAILURE(SetFirstInputToRange(0, 256));

    ASSERT_TRUE(Infer());

    ASSERT_NO_FATAL_FAILURE(refMVN(_inputMap.begin()->second, _refBlob, acrossChannels, normalize, false));

    Compare(_outputMap.begin()->second, _refBlob, ERROR_BOUND);
}

static std::vector<Dims> s_MVNTensors = {
        {{1, 3, 512, 896}}
};

static std::vector<AcrossChannels> s_MVN_acrossChannels = { 0, 1};
static std::vector<Normalize> s_MVN_normalize = { 0, 1};
static std::vector<std::string> s_MVNCustomConfig = {
        {"", TestsCommon::get_data_path() + "/vpu/mvcl/customLayers_m7.xml"}
};

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsMVN_nightly,
        ::testing::Combine(
        ::testing::ValuesIn(s_MVNTensors),
        ::testing::ValuesIn(s_MVN_acrossChannels),
        ::testing::ValuesIn(s_MVN_normalize),
        ::testing::ValuesIn(s_MVNCustomConfig)));

TEST_F(myriadLayersTests_nightly, MVN_CHW_Input)
{
    std::string model = R"V0G0N(
        <net name="MVN" version="2" batch="1">
            <layers>
                <layer name="data" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>512</dim>
                            <dim>896</dim>
                        </port>
                    </output>
                </layer>
                <layer name="mvn" type="MVN" precision="FP16" id="2">
                    <data across_channels="1" eps="9.999999717180685e-10" normalize_variance="1"/>
                    <input>
                        <port id="2">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>512</dim>
                            <dim>896</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>512</dim>
                            <dim>896</dim>
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
    _outputsInfo["mvn"]->setPrecision(Precision::FP16);

    ASSERT_NO_THROW(st = myriadPluginPtr->LoadNetwork(_exeNetwork, network,
                                                      {{VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), CONFIG_VALUE(YES)}}, &_resp));
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

    ASSERT_NO_THROW(st = _inferRequest->SetBlob("mvn", outputNCHW, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = _inferRequest->Infer(&_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_FATAL_FAILURE(refMVN(inputNCHW, output_ref, 1, 1, true));

    Compare(outputNCHW, output_ref, 0.003);
}
