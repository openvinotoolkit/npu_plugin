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

#include <memory>
#include <vpu/kmb_plugin_config.hpp>
#include "kmb_layers_tests.hpp"

#define ERROR_BOUND (.1f)

using namespace InferenceEngine;

#ifdef ENABLE_MCM_COMPILER
TEST_F(kmbLayersTests_nightly, DISABLED_TestsEltwiseAfterScaleShift) {
    // MCM compiler does not support multiple Input layers
    // ScaleShift result is used as the second input to Eltwise
    // TODO find a way to specify several inputs to Eltwise without other layers
    const std::string model = R"V0G0N(
    <net batch="1" name="ELTWISE_TEST" version="2">
        <layers>
            <layer id="0" name="input" precision="FP16" type="Input">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>6</dim>
                        <dim>56</dim>
                        <dim>56</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="scale_shift1" precision="FP16" type="ScaleShift">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>6</dim>
                        <dim>56</dim>
                        <dim>56</dim>
                    </port>
                </input>
                <output>
                    <port id="3">
                        <dim>1</dim>
                        <dim>6</dim>
                        <dim>56</dim>
                        <dim>56</dim>
                    </port>
                </output>
                <blobs>
                    <weights offset="0" size="12"/>
                    <biases offset="12" size="0"/>
                </blobs>
            </layer>
            <layer id="2" name="eltwise1" precision="FP16" type="Eltwise">
                <data operation="sum"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>6</dim>
                        <dim>56</dim>
                        <dim>56</dim>
                    </port>
                    <port id="1">
                        <dim>1</dim>
                        <dim>6</dim>
                        <dim>56</dim>
                        <dim>56</dim>
                    </port>
                </input>
                <output>
                    <port id="2">
                        <dim>1</dim>
                        <dim>6</dim>
                        <dim>56</dim>
                        <dim>56</dim>
                    </port>
                </output>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
            <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
            <edge from-layer="1" from-port="3" to-layer="2" to-port="1"/>
        </edges>
    </net>
        )V0G0N";

    std::size_t weightSize = 12;
    std::size_t biasSize = 0;
    TBlob<uint8_t>::Ptr weightsBlob(GenWeights<uint16_t >(weightSize + biasSize));

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());
    ASSERT_NO_THROW(_net_reader.SetWeights(weightsBlob));

    CNNNetwork network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["eltwise1"]->setPrecision(Precision::FP16);

    std::map<std::string, std::string> config;
    setCommonConfig(config);
    config[VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY)] = CONFIG_VALUE(NO);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_BLOB)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_DOT)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_JSON)] = CONFIG_VALUE(YES);

    ASSERT_NO_THROW(ie.LoadNetwork(network, "kmb", config));
}

TEST_F(kmbLayersTests_nightly, DISABLED_TestsEltwiseAfterScaleShiftWithLargeWeight) {
    const std::string model = R"V0G0N(
    <net batch="1" name="ELTWISE_TEST" version="2">
        <layers>
            <layer id="0" name="input" precision="FP16" type="Input">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>256</dim>
                        <dim>56</dim>
                        <dim>56</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="scale_shift1" precision="FP16" type="ScaleShift">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>256</dim>
                        <dim>56</dim>
                        <dim>56</dim>
                    </port>
                </input>
                <output>
                    <port id="3">
                        <dim>1</dim>
                        <dim>256</dim>
                        <dim>56</dim>
                        <dim>56</dim>
                    </port>
                </output>
                <blobs>
                    <weights offset="0" size="512"/>
                    <biases offset="512" size="0"/>
                </blobs>
            </layer>
            <layer id="2" name="eltwise1" precision="FP16" type="Eltwise">
                <data operation="sum"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>256</dim>
                        <dim>56</dim>
                        <dim>56</dim>
                    </port>
                    <port id="1">
                        <dim>1</dim>
                        <dim>256</dim>
                        <dim>56</dim>
                        <dim>56</dim>
                    </port>
                </input>
                <output>
                    <port id="2">
                        <dim>1</dim>
                        <dim>256</dim>
                        <dim>56</dim>
                        <dim>56</dim>
                    </port>
                </output>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
            <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
            <edge from-layer="1" from-port="3" to-layer="2" to-port="1"/>
        </edges>
    </net>
        )V0G0N";

    std::size_t weightSize = 512;
    std::size_t biasSize = 0;
    TBlob<uint8_t>::Ptr weightsBlob(GenWeights<uint16_t >(weightSize + biasSize));

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());
    ASSERT_NO_THROW(_net_reader.SetWeights(weightsBlob));

    CNNNetwork network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["eltwise1"]->setPrecision(Precision::FP16);

    std::map<std::string, std::string> config;
    setCommonConfig(config);
    config[VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_BLOB)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_DOT)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_JSON)] = CONFIG_VALUE(YES);

    ASSERT_NO_THROW(ie.LoadNetwork(network, "kmb", config));
}
#endif
