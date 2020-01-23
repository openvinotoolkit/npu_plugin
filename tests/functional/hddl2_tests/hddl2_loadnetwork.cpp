//
// Copyright 2020 Intel Corporation.
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

#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>

#include <fstream>
#include <ie_core.hpp>

namespace IE = InferenceEngine;

class HDDL2_loadNetwork_Tests : public ::testing::Test {
public:
    std::string device_name = "HDDL2";
    InferenceEngine::Core ie;
    InferenceEngine::ExecutableNetwork executableNetwork;
    InferenceEngine::InferRequest inferRequest;

    const std::string model = R"V0G0N(
    <net batch="1" name="POOLING_TEST" version="2">
    <layers>
        <layer id="0" name="input" precision="U8" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>224</dim>
                    <dim>224</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="pooling_test" precision="U8" type="Pooling">
            <data kernel="3,3" strides="2,2" exclude-pad="true" pool-method="max" pads_begin="0,0" pads_end="0,0"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>224</dim>
                    <dim>224</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>125</dim>
                    <dim>125</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
    </net>
        )V0G0N";

    InferenceEngine::Blob::CPtr weights;
    InferenceEngine::CNNNetwork network = ie.ReadNetwork(model, weights);
};

TEST_F(HDDL2_loadNetwork_Tests, CanFindPlugin) { ASSERT_NO_THROW(ie.LoadNetwork(network, device_name)); }

TEST_F(HDDL2_loadNetwork_Tests, CanCreateExecutableNetworkLoadNetwork) {
    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(network, device_name));
}

TEST_F(HDDL2_loadNetwork_Tests, CanCreateInferRequestAfterLoadNetwork) {
    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(network, device_name));
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());
}

TEST_F(HDDL2_loadNetwork_Tests, CanCallInfer) {
    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(network, device_name));
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    ASSERT_NO_THROW(inferRequest.Infer());
}
