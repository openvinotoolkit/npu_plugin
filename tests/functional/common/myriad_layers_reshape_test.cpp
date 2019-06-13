// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "ie_layouts.h"
#include "myriad_layers_tests.hpp"
#include <vpu/private_plugin_config.hpp>
#include "myriad_layers_reference_functions.hpp"

using namespace InferenceEngine;

using myriadEliminateReshapeTests_nightly = myriadLayersTests_nightly;

TEST_F(myriadEliminateReshapeTests_nightly, TF_mobilenet_v1_0_25_128) {
    std::string modelName = "MobileNet/tf_mobilenet_v1_0.25_128_fp16";
    std::ostringstream modelFile;
    modelFile << "/" << modelName << ".xml";

    std::ostringstream weightsFile;
    weightsFile << "/" << modelName << ".bin";

    std::string modelFilePath = ModelsPath() + modelFile.str();
    std::string weightsFilePath = ModelsPath() + weightsFile.str();

    ASSERT_NO_THROW(_net_reader.ReadNetwork(modelFilePath));
    ASSERT_TRUE(_net_reader.isParseSuccess());
    ASSERT_NO_THROW(_net_reader.ReadWeights(weightsFilePath));

    StatusCode st;

    ASSERT_NO_THROW(st = myriadPluginPtr->LoadNetwork(_exeNetwork, _net_reader.getNetwork(),
                                                      {
                                                          {
                                                              VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION),
                                                              CONFIG_VALUE(NO)
                                                          },
                                                          {
                                                              CONFIG_KEY(PERF_COUNT),
                                                              CONFIG_VALUE(YES)
                                                          }
                                                      },
                                                      &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = _exeNetwork->CreateInferRequest(_inferRequest, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = _inferRequest->Infer(&_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    std::map<std::string, InferenceEngineProfileInfo> perfMap;
    ASSERT_NO_THROW(st = _inferRequest->GetPerformanceCounts(perfMap, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    auto layerInfo = perfMap["MobilenetV1/Logits/SpatialSqueeze"];
    ASSERT_EQ(InferenceEngineProfileInfo::OPTIMIZED_OUT, layerInfo.status);
}

TEST_F(myriadLayersTests_nightly, ReshapeAfterConcat_Eliminate) {
    std::string model = R"V0G0N(
        <net name="ReshapeAfterConcat_Eliminate" version="2" batch="1">
            <layers>
                <layer name="input1" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>30</dim>
                        </port>
                    </output>
                </layer>
                <layer name="input2" type="Input" precision="FP16" id="2">
                    <output>
                        <port id="2">
                            <dim>1</dim>
                            <dim>20</dim>
                        </port>
                    </output>
                </layer>
                <layer name="input3" type="Input" precision="FP16" id="3">
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>10</dim>
                        </port>
                    </output>
                </layer>

                <layer name="input1_copy" type="Power" precision="FP16" id="4">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="4">
                            <dim>1</dim>
                            <dim>30</dim>
                        </port>
                    </input>
                    <output>
                        <port id="5">
                            <dim>1</dim>
                            <dim>30</dim>
                        </port>
                    </output>
                </layer>
                <layer name="input2_copy" type="Power" precision="FP16" id="5">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="6">
                            <dim>1</dim>
                            <dim>20</dim>
                        </port>
                    </input>
                    <output>
                        <port id="7">
                            <dim>1</dim>
                            <dim>20</dim>
                        </port>
                    </output>
                </layer>
                <layer name="input3_copy" type="Power" precision="FP16" id="6">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="8">
                            <dim>1</dim>
                            <dim>10</dim>
                        </port>
                    </input>
                    <output>
                        <port id="9">
                            <dim>1</dim>
                            <dim>10</dim>
                        </port>
                    </output>
                </layer>

                <layer name="concat" type="Concat" precision="FP16" id="7">
                    <concat_data axis="1"/>
                    <input>
                        <port id="10">
                            <dim>1</dim>
                            <dim>30</dim>
                        </port>
                        <port id="11">
                            <dim>1</dim>
                            <dim>20</dim>
                        </port>
                        <port id="12">
                            <dim>1</dim>
                            <dim>10</dim>
                        </port>
                    </input>
                    <output>
                        <port id="13">
                            <dim>1</dim>
                            <dim>60</dim>
                        </port>
                    </output>
                </layer>

                <layer name="reshape" type="Reshape" precision="FP16" id="8">
                    <data dim="0,-1,30" axis="0" num_axes="-1"/>
                    <input>
                        <port id="14">
                            <dim>1</dim>
                            <dim>60</dim>
                        </port>
                    </input>
                    <output>
                        <port id="15">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>30</dim>
                        </port>
                    </output>
                </layer>

                <layer name="reshape_copy" type="Power" precision="FP16" id="9">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="16">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>30</dim>
                        </port>
                    </input>
                    <output>
                        <port id="17">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>30</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="1" to-layer="4" to-port="4"/>
                <edge from-layer="2" from-port="2" to-layer="5" to-port="6"/>
                <edge from-layer="3" from-port="3" to-layer="6" to-port="8"/>

                <edge from-layer="4" from-port="5" to-layer="7" to-port="10"/>
                <edge from-layer="5" from-port="7" to-layer="7" to-port="11"/>
                <edge from-layer="6" from-port="9" to-layer="7" to-port="12"/>

                <edge from-layer="7" from-port="13" to-layer="8" to-port="14"/>

                <edge from-layer="8" from-port="15" to-layer="9" to-port="16"/>
            </edges>
        </net>
    )V0G0N";

    StatusCode st;

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());

    auto network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["input1"]->setInputPrecision(Precision::FP16);
    _inputsInfo["input2"]->setInputPrecision(Precision::FP16);
    _inputsInfo["input3"]->setInputPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["reshape_copy"]->setPrecision(Precision::FP16);

    ASSERT_NO_THROW(st = myriadPluginPtr->LoadNetwork(_exeNetwork, network, { {VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), CONFIG_VALUE(NO)},
                                                                              {CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES)} }, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    ASSERT_NE(_exeNetwork, nullptr) << _resp.msg;

    ASSERT_NO_THROW(st = _exeNetwork->CreateInferRequest(_inferRequest, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    Blob::Ptr input1;
    ASSERT_NO_THROW(st = _inferRequest->GetBlob("input1", input1, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    GenRandomData(input1);

    Blob::Ptr input2;
    ASSERT_NO_THROW(st = _inferRequest->GetBlob("input2", input2, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    GenRandomData(input2);

    Blob::Ptr input3;
    ASSERT_NO_THROW(st = _inferRequest->GetBlob("input3", input3, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    GenRandomData(input3);

    ASSERT_NO_THROW(st = _inferRequest->Infer(&_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    Blob::Ptr output;
    ASSERT_NO_THROW(st = _inferRequest->GetBlob("reshape_copy", output, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    _refBlob = make_shared_blob<ie_fp16, const SizeVector>(Precision::FP16, Layout::ANY, output->dims());
    _refBlob->allocate();
    {
        ie_fp16* dst_ptr = _refBlob->buffer().as<ie_fp16*>();
        int dst_offset = 0;

        auto concat = [&](const Blob::Ptr& src) {
            const ie_fp16* src_ptr = src->cbuffer().as<const ie_fp16*>();
            int num = src->dims()[0];
            std::copy_n(src_ptr, num, dst_ptr + dst_offset);
            dst_offset += num;
        };

        concat(input1);
        concat(input2);
        concat(input3);
    }

    Compare(output, _refBlob, 0);

    std::map<std::string, InferenceEngineProfileInfo> perfMap;
    ASSERT_NO_THROW(st = _inferRequest->GetPerformanceCounts(perfMap, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    auto layerInfo = perfMap["reshape"];
    EXPECT_EQ(InferenceEngineProfileInfo::OPTIMIZED_OUT, layerInfo.status);
}


typedef myriadLayerTestBaseWithParam<std::tuple<InferenceEngine::SizeVector, InferenceEngine::SizeVector>> myriadLayerReshape_nightly;

TEST_P(myriadLayerReshape_nightly, Reshape) {
    auto input_tensor = std::get<0>(GetParam());
    auto output_tensor = std::get<1>(GetParam());
    std::string shape = std::to_string(output_tensor[0]) + ",";
    shape += std::to_string(output_tensor[1]) + ",";
    shape += std::to_string(output_tensor[2]) + ",";
    shape += std::to_string(output_tensor[3]);
    std::map<std::string, std::string> params;
    params["dim"] = shape;
    /* IE requires these parameters */
    params["axis"] = std::to_string(1);
    params["num_axes"] = std::to_string(-1);
    AddLayer("Reshape",
            nullptr,
            {input_tensor},
            {output_tensor},
            ref_reshape_wrap);
    ASSERT_TRUE(GenerateNetAndInfer(CheckMyriadX(), true, 3));
}


typedef myriadLayersTests_nightly myriadLayerReshapeFasterRCNN_nightly;

TEST_F(myriadLayerReshapeFasterRCNN_nightly, Reshape) {
    InferenceEngine::SizeVector input_tensor = {1, 14, 14, 24};
    InferenceEngine::SizeVector output_tensor = {1, 2352, 2};
    std::map<std::string, std::string> layer_params = {
              {"axis", "0"}
            ,{"dim", "0,-1,2"}
            ,{"num_axes", std::to_string(-1)}
    };
    AddLayer("Reshape",
            &layer_params,
            {input_tensor},
            {output_tensor},
            ref_reshape_wrap);
    ASSERT_TRUE(GenerateNetAndInfer(CheckMyriadX(), true, 3));
}

static std::vector<InferenceEngine::SizeVector> s_reshapeInParams = {
    {{1, 4, 2, 16}},
    {{1, 2, 4, 16}},
    {{1, 4, 16, 2}},
    {{1, 16, 4, 2}},
    {{1, 8,  4,  4}},
};

static std::vector<InferenceEngine::SizeVector> s_reshapeOutParams = {
    {{1, 16, 2, 4}},
    {{1, 4, 16, 2}},
    {{1, 4, 2, 16}},
    {{1, 4, 4,  8}},
    {{1, 4, 8,  4}},
    {{1, 2, 4, 16}},
    {{1, 2, 16, 4}},
    {{1, 64, 2, 1}},
};

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayerReshape_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(s_reshapeInParams),
        ::testing::ValuesIn(s_reshapeOutParams))
);

std::string MODEL_WITH_FLATTEN = R"V0G0N(
    <net name="MODEL_WITH_FLATTEN" version="2" batch="1">
        <layers>
            <layer id="0" name="input" precision="FP16" type="Input">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>16</dim>
                        <dim>6</dim>
                        <dim>6</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="pool5" precision="FP16" type="Pooling">
                <data exclude-pad="false" kernel-x="2" kernel-y="2" pad-x="0" pad-y="0" pool-method="max" stride="1,1,2,2" stride-x="2" stride-y="2"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>16</dim>
                        <dim>6</dim>
                        <dim>6</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>16</dim>
                        <dim>3</dim>
                        <dim>3</dim>
                    </port>
                </output>
            </layer>
            <layer id="2" name="flatten_0" precision="FP16" type="Reshape">
                <data axis="1" dim="1,144" num_axes="-1" />
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>16</dim>
                        <dim>3</dim>
                        <dim>3</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>144</dim>
                    </port>
                </output>
            </layer>
            <layer id="3" name="fc6" precision="FP16" type="FullyConnected">
                <data out-size="32"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>144</dim>
                    </port>
                </input>
                <output>
                    <port id="3">
                        <dim>1</dim>
                        <dim>32</dim>
                    </port>
                </output>
                <blobs>
                    <weights offset="0" size="9216"/>
                    <biases offset="9216" size="64"/>
                </blobs>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
            <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
            <edge from-layer="2" from-port="1" to-layer="3" to-port="0"/>
        </edges>
    </net>
)V0G0N";

std::string MODEL_WITHOUT_FLATTEN = R"V0G0N(
    <net name="MODEL_WITHOUT_FLATTEN" version="2" batch="1">
        <layers>
            <layer id="0" name="input" precision="FP16" type="Input">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>16</dim>
                        <dim>6</dim>
                        <dim>6</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="pool5" precision="FP16" type="Pooling">
                <data exclude-pad="false" kernel-x="2" kernel-y="2" pad-x="0" pad-y="0" pool-method="max" stride="1,1,2,2" stride-x="2" stride-y="2"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>16</dim>
                        <dim>6</dim>
                        <dim>6</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>16</dim>
                        <dim>3</dim>
                        <dim>3</dim>
                    </port>
                </output>
            </layer>
            <layer id="2" name="fc6" precision="FP16" type="FullyConnected">
                <data out-size="32"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>16</dim>
                        <dim>3</dim>
                        <dim>3</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>32</dim>
                    </port>
                </output>
                <blobs>
                    <weights offset="0" size="9216"/>
                    <biases offset="9216" size="64"/>
                </blobs>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
            <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
        </edges>
    </net>
)V0G0N";


typedef myriadLayerTestBaseWithParam<std::string> myriadLayersTestsReshapeBeforeFC_nightly;

TEST_P(myriadLayersTestsReshapeBeforeFC_nightly, OptimizeReshapeIfItIsPlacedBeforeFC) {
    std::string HWConfigValue = GetParam();
    if (!CheckMyriadX() && HWConfigValue == CONFIG_VALUE(YES)) {
        std::cout << "Disable for non-MyriadX devices" << std::endl;
        return;
    }

    std::string outputName = "fc6";
    StatusCode st = InferenceEngine::OK;
    InferenceEngine::ResponseDesc resp;
    TBlob<uint8_t>::Ptr weights(GenWeights(9280 / sizeof(ie_fp16)));

    InferenceEngine::CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(MODEL_WITH_FLATTEN.data(), MODEL_WITH_FLATTEN.length()));
    ASSERT_NO_THROW(net_reader.SetWeights(weights));
    ASSERT_TRUE(net_reader.isParseSuccess());

    auto network = net_reader.getNetwork();

    auto inputsInfo = network.getInputsInfo();
    inputsInfo["input"]->setInputPrecision(Precision::FP16);

    auto outputsInfo = network.getOutputsInfo();
    outputsInfo[outputName]->setPrecision(Precision::FP16);

    InferenceEngine::IExecutableNetwork::Ptr exeNetwork;
    ASSERT_NO_THROW(st = myriadPluginPtr->LoadNetwork(exeNetwork, network,
                                                     { {VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), HWConfigValue},
                                                       {CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES) }}, &resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    InferenceEngine::IInferRequest::Ptr inferRequest;
    ASSERT_NO_THROW(st = exeNetwork->CreateInferRequest(inferRequest, &resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    Blob::Ptr input;
    ASSERT_NO_THROW(st = inferRequest->GetBlob("input", input, &resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = inferRequest->Infer(&_resp));
    ASSERT_EQ(StatusCode::OK, st) << resp.msg;

    std::map<std::string, InferenceEngineProfileInfo> perfMap;
    ASSERT_NO_THROW(st = inferRequest->GetPerformanceCounts(perfMap, &resp));
    ASSERT_EQ(StatusCode::OK, st) << resp.msg;

    auto layerInfo = perfMap["flatten_0"];
    EXPECT_EQ(InferenceEngineProfileInfo::OPTIMIZED_OUT, layerInfo.status);
}

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsReshapeBeforeFC_nightly,
        ::testing::Values(CONFIG_VALUE(YES), CONFIG_VALUE(NO))
);

class myriadLayersTestsReshapeFasterRCNN_nightly: public ConvolutionTest<>{
};

TEST_P(myriadLayersTestsReshapeFasterRCNN_nightly, Convolution) {
    std::map<std::string, std::string> permute_params = {
              {"order", "0,2,3,1"}
    };
    std::map<std::string, std::string> reshape_params = {
                {"axis", "0"}
              , {"dim", "0,-1,2"}
              , {"num_axes", "-1"}
    };
    InferenceEngine::SizeVector perm_out = {1, 14, 14, 24};
    AddLayer("Permute",
             &permute_params,
             {_output_tensor},
             {perm_out},
             ref_permute_wrap);

    AddLayer("Reshape",
             &reshape_params,
             {perm_out},
             {{1, 2352, 2}},
             ref_reshape_wrap);

    float maxerr = 0;
    maxerr = 0.00066 * (IC) * kernel.x * kernel.y;
    ASSERT_TRUE(GenerateNetAndInfer());
    Compare(_outputMap.begin()->second, GenReferenceOutput(), maxerr);
}

static const std::vector<InferenceEngine::SizeVector> s_convTensor = {
    {{1, 512, 14, 14}} 
};

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsReshapeFasterRCNN_nightly,
        ::testing::Combine(
            ::testing::ValuesIn(s_convTensor)
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<uint32_t>(24)
          , ::testing::Values<uint32_t>(1)
          )
);
