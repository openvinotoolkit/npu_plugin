// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"

using namespace InferenceEngine;

TEST_F(myriadLayersTests_nightly, graphTransformerNotThrowExceptionIfConvOutputIsInputForReLUAndGroupDeconv) {
    const std::string model = R"V0G0N(
    <net name="multi_hcp01" version="2" batch="1">
            <layers>
                <layer name="input" type="Input" precision="FP16" id="0">
                    <output>
                        <port id="0">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>23</dim>
                            <dim>40</dim>
                        </port>
                    </output>
                </layer>
               <layer name="conv1" type="Convolution" precision="FP16" id="1">
                    <convolution_data stride-x="1" stride-y="1" pad-x="0" pad-y="0" kernel-x="1" kernel-y="1" output="3" group="1"/>
                    <input>
                        <port id="1">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>23</dim>
                            <dim>40</dim>
                        </port>
                    </input>
                    <output>
                        <port id="2">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>23</dim>
                            <dim>40</dim>
                        </port>
                    </output>
                    <weights offset="0" size="18"/>
                    <biases offset="18" size="6"/>
                </layer>
                <layer name="conv1/relu" type="ReLU" precision="FP16" id="2">
                    <data negative_slope="0.000000" engine="caffe.ReLUParameter.DEFAULT"/>
                    <input>
                        <port id="3">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>23</dim>
                            <dim>40</dim>
                        </port>
                    </input>
                    <output>
                        <port id="4">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>23</dim>
                            <dim>40</dim>
                        </port>
                    </output>
                </layer>
            <layer name="deconv" type="Deconvolution" precision="FP16" id="3">
                <deconvolution_data stride-x="2" stride-y="2" pad-x="1" pad-y="1" kernel-x="4" kernel-y="4" output="3" group="3"/>
                <input>
                    <port id="5">
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>23</dim>
                        <dim>40</dim>
                    </port>
                </input>
                <output>
                    <port id="6">
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>46</dim>
                        <dim>80</dim>
                    </port>
                </output>
                <weights offset="24" size="96"/>
                <biases offset="120" size="0"/>
            </layer>
        </layers>
            <edges>
                <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
                <edge from-layer="1" from-port="2" to-layer="2" to-port="3"/>
                <edge from-layer="1" from-port="2" to-layer="3" to-port="5"/>
            </edges>
        </net>
        )V0G0N";

    TBlob<uint8_t>::Ptr weightsBlob(GenWeights(120));
    StatusCode st;

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());
    ASSERT_NO_THROW(_net_reader.SetWeights(weightsBlob));

    auto network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setInputPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["conv1/relu"]->setPrecision(Precision::FP16);
    _outputsInfo["deconv"]->setPrecision(Precision::FP16);

    ASSERT_NO_THROW(st = myriadPluginPtr->LoadNetwork(_exeNetwork, network, {}, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
}

TEST_F(myriadLayersTests_nightly, ReLU_PostOp_Conflict) {
    const std::string model = R"V0G0N(
        <Net name="ReLU_PostOp_Conflict" version="2" batch="1">
            <layers>
                <layer name="input" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>64</dim>
                            <dim>64</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv" type="Convolution" precision="FP16" id="2">
                    <convolution_data
                        stride-x="1"
                        stride-y="1"
                        pad-x="1"
                        pad-y="1"
                        kernel-x="3"
                        kernel-y="3"
                        output="16"
                        group="1"/>
                    <input>
                        <port id="2">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>64</dim>
                            <dim>64</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>64</dim>
                            <dim>64</dim>
                        </port>
                    </output>
                    <weights offset="0" size="864"/>
                    <biases offset="864" size="32"/>
                </layer>
                <layer name="relu" type="ReLU" precision="FP16" id="3">
                    <data negative_slope="0.0" engine="caffe.ReLUParameter.DEFAULT"/>
                    <input>
                        <port id="4">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>64</dim>
                            <dim>64</dim>
                        </port>
                    </input>
                    <output>
                        <port id="5">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>64</dim>
                            <dim>64</dim>
                        </port>
                    </output>
                </layer>
                <layer name="power" type="Power" precision="FP16" id="4">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="6">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>64</dim>
                            <dim>64</dim>
                        </port>
                    </input>
                    <output>
                        <port id="7">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>64</dim>
                            <dim>64</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="1" to-layer="2" to-port="2"/>
                <edge from-layer="2" from-port="3" to-layer="3" to-port="4"/>
                <edge from-layer="2" from-port="3" to-layer="4" to-port="6"/>
            </edges>
        </Net>
    )V0G0N";

    size_t num_weights = 432;
    size_t num_bias = 16;

    TBlob<uint8_t>::Ptr weights(GenWeights(num_weights + num_bias));

    StatusCode st;

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());
    ASSERT_NO_THROW(_net_reader.SetWeights(weights));

    auto network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setInputPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["relu"]->setPrecision(Precision::FP16);
    _outputsInfo["power"]->setPrecision(Precision::FP16);

    ASSERT_NO_THROW(st = myriadPluginPtr->LoadNetwork(_exeNetwork, network, {}, &_resp));
    ASSERT_EQ(st, StatusCode::OK);
}

class myriadLayersTestsReLUMergeWithBias_nightly : public myriadLayersTests_nightly {
public:
    void RunTest(const std::string& model, size_t num_weights, size_t num_bias) {
        StatusCode st;

        TBlob<uint8_t>::Ptr weights(GenWeights(num_weights + num_bias));

        ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
        ASSERT_TRUE(_net_reader.isParseSuccess());
        ASSERT_NO_THROW(_net_reader.SetWeights(weights));

        auto network = _net_reader.getNetwork();

        _inputsInfo = network.getInputsInfo();
        _inputsInfo["input"]->setInputPrecision(Precision::FP16);

        _outputsInfo = network.getOutputsInfo();
        _outputsInfo["relu"]->setPrecision(Precision::FP16);

        ASSERT_NO_THROW(st = myriadPluginPtr->LoadNetwork(_exeNetwork, network, { {CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES)},
                                                                                  {VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), CONFIG_VALUE(NO)} },
                                                          &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
        ASSERT_NE(_exeNetwork, nullptr) << _resp.msg;

        ASSERT_NO_THROW(st = _exeNetwork->CreateInferRequest(_inferRequest, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        ASSERT_NO_THROW(st = _inferRequest->Infer(&_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        std::map<std::string, InferenceEngineProfileInfo> perfMap;
        ASSERT_NO_THROW(st = _inferRequest->GetPerformanceCounts(perfMap, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        {
            auto reluAndBiasLayerIt = perfMap.find("relu+Bias");
            ASSERT_TRUE(reluAndBiasLayerIt != perfMap.end());
            EXPECT_EQ(InferenceEngineProfileInfo::EXECUTED, reluAndBiasLayerIt->second.status);
        }
    }
};

TEST_F(myriadLayersTestsReLUMergeWithBias_nightly, DISABLED_AfterConv) {
    const std::string model = R"V0G0N(
        <Net name="ReLU_MergeWithBias_AfterConv" version="2" batch="1">
            <layers>
                <layer name="input" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>64</dim>
                            <dim>64</dim>
                        </port>
                    </output>
                </layer>
                <layer name="main" type="Convolution" precision="FP16" id="2">
                    <convolution_data
                        stride-x="1"
                        stride-y="1"
                        pad-x="1"
                        pad-y="1"
                        kernel-x="3"
                        kernel-y="3"
                        output="16"
                        group="1"/>
                    <input>
                        <port id="2">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>64</dim>
                            <dim>64</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>64</dim>
                            <dim>64</dim>
                        </port>
                    </output>
                    <weights offset="0" size="864"/>
                    <biases offset="864" size="32"/>
                </layer>
                <layer name="relu" type="ReLU" precision="FP16" id="3">
                    <data negative_slope="0.0" engine="caffe.ReLUParameter.DEFAULT"/>
                    <input>
                        <port id="4">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>64</dim>
                            <dim>64</dim>
                        </port>
                    </input>
                    <output>
                        <port id="5">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>64</dim>
                            <dim>64</dim>
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

    size_t num_weights = 432;
    size_t num_bias = 16;

    RunTest(model, num_weights, num_bias);
}

TEST_F(myriadLayersTestsReLUMergeWithBias_nightly, DISABLED_AfterFullyConnected) {
    const std::string model = R"V0G0N(
        <Net name="ReLU_MergeWithBias_AfterFC" version="2" batch="1">
            <layers>
                <layer name="input" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>4</dim>
                            <dim>8</dim>
                            <dim>16</dim>
                        </port>
                    </output>
                </layer>
                <layer name="main" type="FullyConnected" precision="FP16" id="2">
                    <data out-size="8"/>
                    <input>
                        <port id="2">
                            <dim>1</dim>
                            <dim>4</dim>
                            <dim>8</dim>
                            <dim>16</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>8</dim>
                        </port>
                    </output>
                    <weights offset="0" size="8192"/>
                    <biases offset="8192" size="16"/>
                </layer>
                <layer name="relu" type="ReLU" precision="FP16" id="3">
                    <data negative_slope="0.0" engine="caffe.ReLUParameter.DEFAULT"/>
                    <input>
                        <port id="4">
                            <dim>1</dim>
                            <dim>8</dim>
                        </port>
                    </input>
                    <output>
                        <port id="5">
                            <dim>1</dim>
                            <dim>8</dim>
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

    size_t num_weights = 4096;
    size_t num_bias = 8;

    RunTest(model, num_weights, num_bias);
}

#define ERROR_BOUND (1.e-4f)

using namespace InferenceEngine;

struct  ReLULayerDef {
    ParamsStruct list;
}ReLULayer;

static std::vector<ReLULayerDef> s_reluLayerParams = {
    {{{"negative_slope", "0.0"}}},
    {{{"negative_slope", "0.1"}}},
};

typedef myriadLayerTestBaseWithParam<std::tuple<InferenceEngine::SizeVector, ReLULayerDef>> myriadLayerReLU_nightly;

TEST_P(myriadLayerReLU_nightly, ReLU) {
    auto input_dims = std::get<0>(GetParam());
    auto extraLayerParams = std::get<1>(GetParam());
    IN_OUT_desc input_tensor;
    input_tensor.push_back(input_dims);

    /* Copy is implemented to perform filling of the output buffer */
    AddLayer("Copy",
             nullptr,
             input_tensor,
             input_tensor,
             ref_copy_wrap);

    AddLayer("ReLU",
             &extraLayerParams.list,
             input_tensor,
             input_tensor,
             ref_ReLU_wrap);

    ASSERT_TRUE(GenerateNetAndInfer());
    Compare(_outputMap.begin()->second, GenReferenceOutput(), ERROR_BOUND);
}

static std::vector<InferenceEngine::SizeVector> s_copyTensors = {
    {{1, 8, 16, 32},
    {1, 12, 64, 32},
    {24, 32, 16},
    {16, 8, 16}},
};

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayerReLU_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(s_copyTensors),
        ::testing::ValuesIn(s_reluLayerParams)
        )
);

class myriadLayerFullyConnectedWithReLU_nightly: public FCTest<ReLULayerDef>{
};

TEST_P(myriadLayerFullyConnectedWithReLU_nightly, TestsFullyConnected)
{
    auto p = ::testing::WithParamInterface<std::tuple<fcon_test_params, int32_t, int32_t, ReLULayerDef>>::GetParam();
    auto extraLayerParams = std::get<3>(p);
    AddLayer("ReLU",
             &extraLayerParams.list,
             {_output_tensor},
             {_output_tensor},
             ref_ReLU_wrap);
    ASSERT_TRUE(GenerateNetAndInfer());
    Compare(_outputMap.begin()->second, GenReferenceOutput(), _par.error_bound);
}

INSTANTIATE_TEST_CASE_P(
    accuracy, myriadLayerFullyConnectedWithReLU_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(g_fcTestParamsSubset),
        ::testing::Values(g_dimensionsFC[0]),
        ::testing::ValuesIn(g_addBiasFC),
        ::testing::ValuesIn(s_reluLayerParams)
    )
);

#define ERROR_BOUND_WITH_RELU (4.e-3f)

class myriadLayersTestsMaxPoolingWithReLU_nightly: public PoolingTest<POOLING_MAX, ReLULayerDef>{
};

class myriadLayersTestsAvgPoolingWithReLU_nightly: public PoolingTest<POOLING_AVG, ReLULayerDef>{
};

TEST_P(myriadLayersTestsMaxPoolingWithReLU_nightly, TestsMaxPoolingWithReLU)
{
    auto p = ::testing::WithParamInterface<std::tuple<InferenceEngine::SizeVector, pooling_layer_params, const char*, ReLULayerDef>>::GetParam();
    auto extraLayerParams = std::get<3>(p);
    AddLayer("ReLU",
             &extraLayerParams.list,
             {_output_tensor},
             {_output_tensor},
             ref_ReLU_wrap);
    ASSERT_TRUE(GenerateNetAndInfer());
    Compare(_outputMap.begin()->second, GenReferenceOutput(), ERROR_BOUND_WITH_RELU);
}

TEST_P(myriadLayersTestsAvgPoolingWithReLU_nightly, TestsAvgPoolingWithReLU)
{
    auto p = ::testing::WithParamInterface<std::tuple<InferenceEngine::SizeVector, pooling_layer_params, const char*, ReLULayerDef>>::GetParam();
    auto extraLayerParams = std::get<3>(p);
    AddLayer("ReLU",
             &extraLayerParams.list,
             {_output_tensor},
             {_output_tensor},
             ref_ReLU_wrap);
    ASSERT_TRUE(GenerateNetAndInfer());
    Compare(_outputMap.begin()->second, GenReferenceOutput(), ERROR_BOUND_WITH_RELU);
}

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsMaxPoolingWithReLU_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(g_poolingInput),
        ::testing::ValuesIn(g_poolingLayerParamsLite),
        ::testing::ValuesIn(g_poolingLayout),
        ::testing::ValuesIn(s_reluLayerParams))
);

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsAvgPoolingWithReLU_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(g_poolingInput),
        ::testing::ValuesIn(g_poolingLayerParamsLite),
        ::testing::ValuesIn(g_poolingLayout),
        ::testing::ValuesIn(s_reluLayerParams))
);

INSTANTIATE_TEST_CASE_P(accuracy_postop, myriadLayersTestsMaxPoolingWithReLU_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(g_poolingInput_postOp),
        ::testing::Values<pooling_layer_params>(MAKE_STRUCT(pooling_layer_params, {3, 3}, {1, 1}, {1, 1})),
        ::testing::ValuesIn(g_poolingLayout),
        ::testing::Values<ReLULayerDef>(MAKE_STRUCT(ReLULayerDef, {{{"negative_slope", "0.0"}}})))
);

INSTANTIATE_TEST_CASE_P(accuracy_postop, myriadLayersTestsAvgPoolingWithReLU_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(g_poolingInput_postOp),
        ::testing::Values<pooling_layer_params>(MAKE_STRUCT(pooling_layer_params, {3, 3}, {1, 1}, {1, 1})),
        ::testing::ValuesIn(g_poolingLayout),
        ::testing::Values<ReLULayerDef>(MAKE_STRUCT(ReLULayerDef, {{{"negative_slope", "0.0"}}})))
);

class myriadLayerConvolutionWithReLU_nightly: public ConvolutionTest<ReLULayerDef>{
};

TEST_P(myriadLayerConvolutionWithReLU_nightly, Convolution) {
    auto p = ::testing::WithParamInterface<std::tuple<InferenceEngine::SizeVector, param_size, param_size, param_size, uint32_t, uint32_t, ReLULayerDef>>::GetParam();
    auto ReLUParam = std::get<6>(p);
    AddLayer("ReLU",
             &ReLUParam.list,
             {_output_tensor},
             {_output_tensor},
             ref_ReLU_wrap);

    float maxerr = 0;
    if (group == 1)
        maxerr = 0.00055 * IC * kernel.x * kernel.y;
    else // TODO: currently dephConv is slightly less accurate
        maxerr = 0.00066 * (IC / group) * kernel.x * kernel.y;
    ASSERT_TRUE(GenerateNetAndInfer());
    Compare(_outputMap.begin()->second, GenReferenceOutput(), maxerr);
}

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayerConvolutionWithReLU_nightly,
        ::testing::Combine(
            ::testing::ValuesIn(g_convolutionTensors)
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<uint32_t>(16)
          , ::testing::Values<uint32_t>(1)
          , ::testing::ValuesIn(s_reluLayerParams)
          )
);

INSTANTIATE_TEST_CASE_P(DISABLED_accuracy_postop, myriadLayerConvolutionWithReLU_nightly,
        ::testing::Combine(
            ::testing::ValuesIn(g_poolingInput_postOp)
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1)/*, MAKE_STRUCT(param_size, 2, 2)*/)
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<uint32_t>(32)
          , ::testing::Values<uint32_t>(32)
          , ::testing::Values<ReLULayerDef>(MAKE_STRUCT(ReLULayerDef, {{{"negative_slope", "0.0"}}}))
          )
);
