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

#include "kmb_layers_tests.hpp"

using namespace InferenceEngine;

static const std::string model_regionYolo = R"V0G0N(
<?xml version="1.0"?>
<net name="yolov2_IR" version="3" batch="1">
<layers>
    <layer name="data" type="Input" precision="FP16" id="0">
        <output>
            <port id="0">
                <dim>1</dim>
                <dim>125</dim>
                <dim>13</dim>
                <dim>13</dim>
            </port>
        </output>
    </layer>
    <layer name="RegionYolo" type="RegionYolo" precision="FP16" id="1">
        <data axis="1" classes="20" coords="4" do_softmax="1" end_axis="3" num="5" />
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>125</dim>
                    <dim>13</dim>
                    <dim>13</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>21125</dim>
                </port>
            </output>
    </layer>
</layers>
<edges>
    <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
</edges>
</net>
)V0G0N";

// TODO: enable test when mcmCompiler will be able to compile it
#ifdef ENABLE_MCM_COMPILER
TEST_F(kmbLayersTests_nightly, DISABLED_TestRegionYolo) {
    std::string model = model_regionYolo;

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());

    CNNNetwork network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["data"]->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["RegionYolo"]->setPrecision(Precision::FP16);

    std::map<std::string, std::string> config;
    setCommonConfig(config);
    config[VPU_COMPILER_CONFIG_KEY(PARSING_ONLY)] = CONFIG_VALUE(YES);

    ASSERT_NO_THROW(ie.LoadNetwork(network, "kmb", config));
};
#endif
