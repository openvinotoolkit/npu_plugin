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
#include "kmb_xml_tests.hpp"

#include <ie_icnn_network_stats.hpp>
#include <cnn_network_int8_normalizer.hpp>
#include <ie_util_internal.hpp>
#include <conv_ref.hpp>

#define ERROR_BOUND (.1f)

using namespace ::testing;
using namespace InferenceEngine;
using namespace details;

static const std::string layout_conversion_model = R"V0G0N(
    <net batch="1" name="LAYOUT_CONVERSION_TEST" version="6">
        <layers>
            <layer id="0" name="input" precision="U8" type="Input">
                <output>
                    <port id="1">
                        <dim>1</dim>
                        <dim>16</dim>
                        <dim>16</dim>
                        <dim>16</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="conv_test1/weights" precision="U8" type="Const">
                <output>
                    <port id="1">
                        <dim>128</dim>
                        <dim>16</dim>
                        <dim>3</dim>
                        <dim>3</dim>
                    </port>
                </output>
                <blobs>
                    <custom offset="0" size="18432"/>
                </blobs>
            </layer>
            <layer id="2" name="conv_test1/bias" precision="I32" type="Const">
                <output>
                    <port id="1">
                        <dim>128</dim>
                    </port>
                </output>
                <blobs>
                    <custom offset="18432" size="512"/>
                </blobs>
            </layer>
            <layer id="3" name="output" precision="U8" type="Convolution">
                <data kernel="3,3" output="128" strides="1,1" dilations="1,1" group="1"   pads_begin="0,0" pads_end="0,0" />
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>16</dim>
                        <dim>16</dim>
                        <dim>16</dim>
                    </port>
                    <port id="1">
                        <dim>128</dim>
                        <dim>16</dim>
                        <dim>3</dim>
                        <dim>3</dim>
                    </port>
                    <port id="2">
                        <dim>128</dim>
                    </port>
                </input>
                <output>
                    <port id="3">
                        <dim>1</dim>
                        <dim>128</dim>
                        <dim>14</dim>
                        <dim>14</dim>
                    </port>
                </output>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="1" to-layer="3" to-port="0"/>
            <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
            <edge from-layer="2" from-port="1" to-layer="3" to-port="2"/>
        </edges>
    </net>
    )V0G0N";

class LayoutConversionTest : public testing::WithParamInterface<InferenceEngine::Layout>, public kmbLayersTests_nightly {};

#ifdef ENABLE_MCM_COMPILER
template <class srcType, class dstType>
static void testOverflow(const Blob::Ptr& blob) {
    auto data = blob->buffer().as<srcType*>();
    auto maxValue = std::numeric_limits<dstType>::max();
    auto minValue = std::numeric_limits<dstType>::min();
    for (size_t i = 0; i < blob->size(); ++i) {
        if (data[i] < minValue || maxValue < data[i]) {
            THROW_IE_EXCEPTION << "Blob contains value " << data[i] <<  " that exceeds desired range [" << minValue << ", " << maxValue << "]";
        }
    }
}

TEST_P(LayoutConversionTest, layout_conversion_test_manual) {
    const std::vector<size_t> input_dims = {1, 16, 16, 16};
    const conv_common_params conv_params = {{1, 1}, {3, 3}, {0, 0}, {0, 0}, {1, 1}, "", 1, 128, true, true, ""};
    InferenceEngine::Layout output_layout = GetParam();
    SizeVector output_dims;
    getConvOutShape(input_dims, conv_params, output_dims);

    size_t weightsByteSize = getConvWeightsSize(input_dims, conv_params, "U8");
    size_t weightsSize = weightsByteSize / sizeof(uint8_t);

    size_t biasByteSize = output_dims[1] * sizeof(int32_t);
    size_t biasSize = biasByteSize / sizeof(int32_t);

    auto weightsBuffer = make_shared_blob<uint8_t>({Precision::U8, {weightsByteSize + biasByteSize}, Layout::C});
    weightsBuffer->allocate();
    auto weightsBufferData = weightsBuffer->buffer().as<uint8_t*>();
    std::fill_n(weightsBufferData, weightsSize, static_cast<uint8_t>(1));

    uint32_t* biasData = reinterpret_cast<uint32_t*>(weightsBuffer->buffer().as<uint8_t*>() + weightsSize);
    std::fill_n(biasData, biasSize, static_cast<uint32_t>(1));

    const std::string & model = layout_conversion_model;

    CNNNetReader reader;
    ASSERT_NO_THROW(reader.ReadNetwork(model.data(), model.length()));
    ASSERT_NO_THROW(reader.SetWeights(weightsBuffer));
    ASSERT_TRUE(reader.isParseSuccess());
    CNNNetwork network = reader.getNetwork();

    auto _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::U8);

    auto _outputsInfo = network.getOutputsInfo();
    _outputsInfo["output"]->setPrecision(Precision::U8);
    _outputsInfo["output"]->setLayout(output_layout);

    std::map<std::string, std::string> config;
    setCommonConfig(config);
    config[VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY)] = CONFIG_VALUE(NO);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_BLOB)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(LOAD_NETWORK_AFTER_COMPILATION)] = CONFIG_VALUE(YES);

    Core ie;
    InferenceEngine::ExecutableNetwork exeNetwork;
    ASSERT_NO_THROW(exeNetwork = ie.LoadNetwork(network, "KMB", config));
    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = exeNetwork.CreateInferRequest());
    Blob::Ptr inputBlob;
    ASSERT_NO_THROW(inputBlob = inferRequest.GetBlob(exeNetwork.GetInputsInfo().begin()->first));
    auto data = inputBlob->buffer().as<uint8_t*>();
    fillIntBuffer(data, inputBlob->byteSize(), static_cast<uint8_t>(1), static_cast<uint8_t>(1));

    auto weightsData = weightsBuffer->buffer().as<int8_t*>();
    auto bias_data = conv_params.with_bias ? reinterpret_cast<int32_t*>(weightsData + weightsSize) : nullptr;

    auto outputBlob = inferRequest.GetBlob(exeNetwork.GetOutputsInfo().begin()->first);
    auto outputDesc = outputBlob->getTensorDesc();
    auto refOutputBlob = make_shared_blob<float>({Precision::FP32, outputDesc.getDims(), outputDesc.getLayout()});
    refOutputBlob->allocate();
    data = refOutputBlob->buffer().as<uint8_t*>();
    std::fill(data, data + refOutputBlob->byteSize(), 0);

    ref_conv_common({inputBlob}, *refOutputBlob, weightsData, weightsSize, bias_data, biasSize, conv_params);
    testOverflow<float, uint8_t>(refOutputBlob);
    ASSERT_NO_THROW(inferRequest.Infer());
    Blob::Ptr outputBlobFP32 = ConvertU8ToFP32(outputBlob);

    Compare(refOutputBlob, outputBlobFP32, 1.1f);
}
#endif

const std::vector<InferenceEngine::Layout> test_layouts = {
    InferenceEngine::Layout::NHWC, InferenceEngine::Layout::NCHW
};

INSTANTIATE_TEST_CASE_P(accuracy, LayoutConversionTest, ::testing::ValuesIn(test_layouts));
