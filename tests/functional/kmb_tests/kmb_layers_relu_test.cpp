//
// Copyright 2019 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include <vpu/kmb_plugin_config.hpp>

#include "kmb_layers_tests.hpp"

#define ERROR_BOUND (.1f)

using namespace InferenceEngine;

typedef kmbLayerTestBaseWithParam<tensor_test_params> kmbLayersTestsReLUParams;

#ifdef ENABLE_MCM_COMPILER
TEST_F(kmbLayersTests_nightly, DISABLED_TestsReLUAfterConvolution) {
    // TODO: mcmCompiler compilation fails (Convolution with bias): Segmentation fault. Jira: VPUNND-1474
    const std::string model = R"V0G0N(
    <net batch="1" name="RELU_TEST" version="2">
        <layers>
            <layer id="0" name="input" precision="FP16" type="Input">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>224</dim>
                        <dim>224</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="scale_shift" precision="FP16" type="ScaleShift">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>224</dim>
                        <dim>224</dim>
                    </port>
                </input>
                <output>
                    <port id="3">
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>224</dim>
                        <dim>224</dim>
                    </port>
                </output>
                <blobs>
                    <weights offset="0" size="6"/>
                    <biases offset="6" size="6"/>
                </blobs>
            </layer>
            <layer id="2" name="conv" precision="FP16" type="Convolution">
                <data dilations="1,1" group="1" kernel="7,7" output="64" pads_begin="3,3" pads_end="3,3" strides="2,2"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>224</dim>
                        <dim>224</dim>
                    </port>
                </input>
                <output>
                    <port id="3">
                        <dim>1</dim>
                        <dim>64</dim>
                        <dim>112</dim>
                        <dim>112</dim>
                    </port>
                </output>
                <blobs>
                    <weights offset="12" size="18816"/>
                    <biases offset="18828" size="128"/>
                </blobs>
            </layer>
            <layer id="3" name="relu_test" precision="FP16" type="ReLU">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>64</dim>
                        <dim>112</dim>
                        <dim>112</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>64</dim>
                        <dim>112</dim>
                        <dim>112</dim>
                    </port>
                </output>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
            <edge from-layer="1" from-port="3" to-layer="2" to-port="0"/>
            <edge from-layer="2" from-port="3" to-layer="3" to-port="0"/>
        </edges>
    </net>
        )V0G0N";

    std::size_t weightSize = 6 + 18816;
    std::size_t biasSize = 6 + 128;
    TBlob<uint8_t>::Ptr weightsBlob(GenWeights<uint16_t>(weightSize + biasSize));

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());
    ASSERT_NO_THROW(_net_reader.SetWeights(weightsBlob));

    auto network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["relu_test"]->setPrecision(Precision::FP16);

    std::map<std::string, std::string> config;
    setCommonConfig(config);

    // Parsing only is enabled because mcmCompiler can't compile layers.
    // TODO: turn off parsing only when mcmCompiler will be able to compile this layers.
    config[VPU_COMPILER_CONFIG_KEY(PARSING_ONLY)] = CONFIG_VALUE(YES);
    config[VPU_COMPILER_CONFIG_KEY(GENERATE_BLOB)] = CONFIG_VALUE(YES);
    config[VPU_COMPILER_CONFIG_KEY(GENERATE_DOT)] = CONFIG_VALUE(YES);
    config[VPU_COMPILER_CONFIG_KEY(GENERATE_JSON)] = CONFIG_VALUE(YES);

    ASSERT_NO_THROW(_exeNetwork = ie.LoadNetwork(network, "kmb", config));
}

TEST_F(kmbLayersTests_nightly, DISABLED_TestsReLUOnly) {
    const std::string model = R"V0G0N(
    <net batch="1" name="RELU_TEST" version="2">
        <layers>
            <layer id="0" name="input" precision="FP16" type="Input">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>64</dim>
                        <dim>112</dim>
                        <dim>112</dim>
                    </port>
                </output>
            </layer>
            <layer id="3" name="relu_test" precision="FP16" type="ReLU">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>64</dim>
                        <dim>112</dim>
                        <dim>112</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>64</dim>
                        <dim>112</dim>
                        <dim>112</dim>
                    </port>
                </output>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        </edges>
    </net>
        )V0G0N";

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());

    auto network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["relu_test"]->setPrecision(Precision::FP16);

    std::map<std::string, std::string> config;
    // LoadNetwork returns (-1) when MCM_PARSING_ONLY is set to NO
    // TODO disable 'parse only' and find out why LoadNetwork fails
    setCommonConfig(config);
    config[VPU_COMPILER_CONFIG_KEY(PARSING_ONLY)] = CONFIG_VALUE(YES);

    ASSERT_NO_THROW(_exeNetwork = ie.LoadNetwork(network, "kmb", config));
}

TEST_P(kmbLayersTestsReLUParams, TestsReLUNetInit) {
    auto param = GetParam();
    tensor_test_params tensor = param;

    const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();

    std::cout << ::testing::UnitTest::GetInstance()->current_test_info()->name()
              << " test_info->name()=" << test_info->name() << " test_info->test_case_name() "
              << test_info->test_case_name() << std::endl;

    std::map<std::string, std::string> params;

    SetInputTensor(tensor);
    SetOutputTensor(tensor);

    // TODO replace the code below with NetworkInit when LoadNetwork stops failing
    std::string xml;
    genXML("ReLU", &params, 0, 0, xml);

    ASSERT_NO_THROW(_net_reader.ReadNetwork(xml.data(), xml.length()));
    ASSERT_EQ(_net_reader.isParseSuccess(), true);

    CNNNetwork network = _net_reader.getNetwork();
    _inputsInfo = network.getInputsInfo();
    for (const auto& in : _inputsInfo) {
        in.second->setPrecision(Precision::FP16);
    }
    _outputsInfo = network.getOutputsInfo();
    for (const auto& outputInfo : _outputsInfo) {
        outputInfo.second->setPrecision(Precision::FP16);
    }
    std::map<std::string, std::string> config;
    // LoadNetwork results in 'Null pointer dereference' response
    // TODO disable 'parse only' and find out why LoadNetwork fails
    config[VPU_COMPILER_CONFIG_KEY(PARSING_ONLY)] = CONFIG_VALUE(YES);

    ASSERT_NO_THROW(_exeNetwork = ie.LoadNetwork(network, "kmb", config));
}

static const tensor_test_params paramsTable[] = {
    {1, 64, 112, 112},  // input and output tensors
};

INSTANTIATE_TEST_CASE_P(loadNetworkNoThrow, kmbLayersTestsReLUParams, ::testing::ValuesIn(paramsTable));
#endif
