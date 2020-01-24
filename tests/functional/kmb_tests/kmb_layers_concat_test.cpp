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
#include <vpu/vpu_compiler_config.hpp>

#include "kmb_layers_tests.hpp"

#define ERROR_BOUND (.1f)

using namespace InferenceEngine;

#ifdef ENABLE_MCM_COMPILER
TEST_F(kmbLayersTests_nightly, DISABLED_TestsConcatenationAfterSoftmax) {
    // MCM compiler does not support multiple Input layers
    // SoftMax result is used as the second input to Concat
    // TODO find a way to specify several inputs to Concat without other layers
    const std::string model = R"V0G0N(
    <net batch="1" name="CONCAT_TEST" version="2">
        <layers>
            <layer id="0" name="input" precision="FP16" type="Input">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>64</dim>
                        <dim>35</dim>
                        <dim>35</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="softmax1" precision="FP16" type="SoftMax">
                <data axis="1"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>64</dim>
                        <dim>35</dim>
                        <dim>35</dim>
                    </port>
                </input>
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>64</dim>
                        <dim>35</dim>
                        <dim>35</dim>
                    </port>
                </output>
            </layer>
            <layer id="2" name="concat1" precision="FP16" type="Concat">
                <data axis="1"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>64</dim>
                        <dim>35</dim>
                        <dim>35</dim>
                    </port>
                    <port id="1">
                        <dim>1</dim>
                        <dim>64</dim>
                        <dim>35</dim>
                        <dim>35</dim>
                    </port>
                </input>
                <output>
                    <port id="2">
                        <dim>1</dim>
                        <dim>128</dim>
                        <dim>35</dim>
                        <dim>35</dim>
                    </port>
                </output>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
            <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
            <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
        </edges>
    </net>
        )V0G0N";

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());

    auto network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["concat1"]->setPrecision(Precision::FP16);

    std::map<std::string, std::string> config;
    setCommonConfig(config);

    // Parsing only is enabled because mcmCompiler can't compile layers.
    // TODO: turn off parsing only when mcmCompiler will be able to compile this layers.
    config[VPU_COMPILER_CONFIG_KEY(PARSING_ONLY)] = CONFIG_VALUE(YES);
    config[VPU_COMPILER_CONFIG_KEY(GENERATE_BLOB)] = CONFIG_VALUE(YES);
    config[VPU_COMPILER_CONFIG_KEY(GENERATE_DOT)] = CONFIG_VALUE(YES);
    config[VPU_COMPILER_CONFIG_KEY(GENERATE_JSON)] = CONFIG_VALUE(YES);

    ASSERT_NO_THROW(ie.LoadNetwork(network, "kmb", config));
}
#endif
