// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"

using namespace InferenceEngine;

using myriadInferTests_nightly = myriadLayersTests_nightly;

TEST_F(myriadInferTests_nightly, NCHW_Input) {
    std::string model = R"V0G0N(
        <net name="Power" version="2" batch="1">
            <layers>
                <layer name="data" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>600</dim>
                            <dim>800</dim>
                        </port>
                    </output>
                </layer>
                <layer name="power" type="Power" precision="FP16" id="2">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="2">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>600</dim>
                            <dim>800</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>600</dim>
                            <dim>800</dim>
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
    _outputsInfo["power"]->setPrecision(Precision::FP16);

    ASSERT_NO_THROW(st = myriadPluginPtr->LoadNetwork(_exeNetwork, network, {}, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    ASSERT_NE(_exeNetwork, nullptr) << _resp.msg;

    ASSERT_NO_THROW(st = _exeNetwork->CreateInferRequest(_inferRequest, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    auto dims = _inputsInfo["data"]->getDims();

    auto inputNHWC = make_shared_blob<ie_fp16, const SizeVector>(Precision::FP16, Layout::NHWC, dims);
    ASSERT_NO_THROW(inputNHWC->allocate());

    auto outputNHWC = make_shared_blob<ie_fp16, const SizeVector>(Precision::FP16, Layout::NHWC, dims);
    ASSERT_NO_THROW(outputNHWC->allocate());

    auto inputNCHW = make_shared_blob<ie_fp16, const SizeVector>(Precision::FP16, Layout::NCHW, dims);
    ASSERT_NO_THROW(inputNCHW->allocate());

    auto outputNCHW = make_shared_blob<ie_fp16, const SizeVector>(Precision::FP16, Layout::NHWC, dims);
    ASSERT_NO_THROW(outputNCHW->allocate());

    ASSERT_NO_THROW(GenRandomData(inputNHWC));
    ASSERT_NO_THROW(
        ConvertLayout<ie_fp16>(inputNHWC->layout(), inputNCHW->layout(),
                               inputNHWC->cbuffer().as<const ie_fp16*>(),
                               inputNCHW->buffer().as<ie_fp16*>(),
                               inputNCHW->dims()));

    ASSERT_NO_THROW(st = _inferRequest->SetBlob("data", inputNHWC, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = _inferRequest->SetBlob("power", outputNHWC, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = _inferRequest->Infer(&_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = _inferRequest->SetBlob("data", inputNCHW, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = _inferRequest->SetBlob("power", outputNCHW, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = _inferRequest->Infer(&_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    Compare(outputNHWC, outputNCHW, 0.0);
}

TEST_F(myriadInferTests_nightly, AddOutputToConvWithReLU) {
    const std::string conv_model = R"V0G0N(
        <Net name="conv_model" version="2" batch="1">
            <layers>
                <layer name="input" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv" type="Convolution" precision="FP16" id="2">
                    <convolution_data
                        stride-x="1" stride-y="1"
                        pad-x="0" pad-y="0"
                        kernel-x="1" kernel-y="1"
                        output="64"
                        group="1"/>
                    <input>
                        <port id="2">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                    <weights offset="0" size="8192"/>
                    <biases offset="8192" size="128"/>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="1" to-layer="2" to-port="2"/>
            </edges>
        </Net>
    )V0G0N";

    const std::string full_model = R"V0G0N(
        <Net name="full_model" version="2" batch="1">
            <layers>
                <layer name="input" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv" type="Convolution" precision="FP16" id="2">
                    <convolution_data
                        stride-x="1" stride-y="1"
                        pad-x="0" pad-y="0"
                        kernel-x="1" kernel-y="1"
                        output="64"
                        group="1"/>
                    <input>
                        <port id="2">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                    <weights offset="0" size="8192"/>
                    <biases offset="8192" size="128"/>
                </layer>
                <layer name="relu" type="ReLU" precision="FP16" id="3">
                    <input>
                        <port id="4">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </input>
                    <output>
                        <port id="5">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="1" to-layer="2" to-port="2"/>
                <edge from-layer="2" from-port="3" to-layer="3" to-port="4"/>
            </edges>
        </Net>
    )V0G0N";

    StatusCode st;

    TBlob<uint8_t>::Ptr weights(GenWeights(8320 / sizeof(ie_fp16)));

    CNNNetReader conv_net_reader;
    ASSERT_NO_THROW(conv_net_reader.ReadNetwork(conv_model.c_str(), conv_model.length()));
    ASSERT_TRUE(conv_net_reader.isParseSuccess());
    ASSERT_NO_THROW(conv_net_reader.SetWeights(weights));

    auto conv_network = conv_net_reader.getNetwork();

    auto conv_inputs_info = conv_network.getInputsInfo();
    conv_inputs_info["input"]->setInputPrecision(Precision::FP16);

    auto conv_outputs_info = conv_network.getOutputsInfo();
    conv_outputs_info["conv"]->setPrecision(Precision::FP16);

    Blob::Ptr input = make_shared_blob<ie_fp16, const SizeVector>(Precision::FP16, Layout::NCHW, conv_inputs_info["input"]->getDims());
    input->allocate();
    GenRandomData(input);

    Blob::Ptr conv_output;
    {
        IExecutableNetwork::Ptr conv_exe;
        ASSERT_NO_THROW(st = myriadPluginPtr->LoadNetwork(conv_exe, conv_network, {}, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
        ASSERT_NE(conv_exe, nullptr) << _resp.msg;

        IInferRequest::Ptr conv_req;
        ASSERT_NO_THROW(st = conv_exe->CreateInferRequest(conv_req, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        ASSERT_NO_THROW(st = conv_req->SetBlob("input", input, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        ASSERT_NO_THROW(st = conv_req->GetBlob("conv", conv_output, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        ASSERT_NO_THROW(st = conv_req->Infer(&_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    }

    CNNNetReader full_net_reader;
    ASSERT_NO_THROW(full_net_reader.ReadNetwork(full_model.c_str(), full_model.length()));
    ASSERT_TRUE(full_net_reader.isParseSuccess());
    ASSERT_NO_THROW(full_net_reader.SetWeights(weights));

    auto full_network = full_net_reader.getNetwork();
    full_network.addOutput("conv", 0);

    auto full_inputs_info = full_network.getInputsInfo();
    full_inputs_info["input"]->setInputPrecision(Precision::FP16);

    auto full_outputs_info = full_network.getOutputsInfo();
    full_outputs_info["conv"]->setPrecision(Precision::FP16);
    full_outputs_info["relu"]->setPrecision(Precision::FP16);

    Blob::Ptr full_output;
    {
        IExecutableNetwork::Ptr full_exe;
        ASSERT_NO_THROW(st = myriadPluginPtr->LoadNetwork(full_exe, full_network, {}, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
        ASSERT_NE(full_exe, nullptr) << _resp.msg;

        IInferRequest::Ptr full_req;
        ASSERT_NO_THROW(st = full_exe->CreateInferRequest(full_req, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        ASSERT_NO_THROW(st = full_req->SetBlob("input", input, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        ASSERT_NO_THROW(st = full_req->GetBlob("conv", full_output, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        ASSERT_NO_THROW(st = full_req->Infer(&_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    }

    Compare(full_output, conv_output, 0.0f);
}
