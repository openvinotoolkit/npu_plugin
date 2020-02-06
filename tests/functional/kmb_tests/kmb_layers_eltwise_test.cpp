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

#include <kmb_conv_utils.hpp>
#include <memory>
#include <vpu/kmb_plugin_config.hpp>

#include "common_test_utils/common_layers_params.hpp"
#include "kmb_layers_tests.hpp"
#include "kmb_xml_tests.hpp"

#define ERROR_BOUND (.1f)

using namespace InferenceEngine;

#ifdef ENABLE_MCM_COMPILER

// TODO find a way to specify several inputs to Eltwise without other layers

struct eltwise_test_params {
    SizeVector input_dim;
    CommonTestUtils::conv_common_params conv_params;
};

class EltwiseTest : public testing::WithParamInterface<eltwise_test_params>, public kmbLayersTests_nightly {
public:
    const SizeVector& input_dims() { return GetParam().input_dim; }
    const CommonTestUtils::conv_common_params& conv_params() { return GetParam().conv_params; }
};

// MCM compiler does not support multiple Input layers
// This test throws when single input is bound to both ports of eltwise
TEST_P(EltwiseTest, TestsEltwiseOnTheSameInputToBothPortsNegative_Test) {
    SizeVector output_dims;
    CommonTestUtils::getConvOutShape(input_dims(), conv_params(), output_dims);

    const size_t weightsByteSize = getConvWeightsSize(input_dims(), conv_params(), "U8");
    const size_t weightsSize = weightsByteSize / sizeof(uint8_t);

    const size_t biasByteSize = output_dims[1] * sizeof(int32_t);
    const size_t biasSize = biasByteSize / sizeof(int32_t);

    const auto weightsBuffer = make_shared_blob<uint8_t>({Precision::U8, {weightsByteSize + biasByteSize}, Layout::C});
    weightsBuffer->allocate();
    const auto weightsBufferData = weightsBuffer->buffer().as<uint8_t*>();
    std::fill_n(weightsBufferData, weightsSize, static_cast<uint8_t>(1));

    uint32_t* biasData = reinterpret_cast<uint32_t*>(weightsBuffer->buffer().as<uint8_t*>() + weightsSize);
    std::fill_n(biasData, biasSize, static_cast<uint32_t>(1));

    const convolution_test_desc allTestParams = {
        input_dims(), conv_params(), "U8", "U8", "U8", 0, "I32", "", elt_on_same_input_with_bias_template};
    const std::string model = instantiateConvTestIR(allTestParams);

    Core ie;
    CNNNetwork network = ie.ReadNetwork(model, weightsBuffer);

    auto _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::U8);

    auto _outputsInfo = network.getOutputsInfo();
    _outputsInfo["output"]->setPrecision(Precision::U8);

    std::map<std::string, std::string> config;
    setCommonConfig(config);
    config[VPU_COMPILER_CONFIG_KEY(PARSING_ONLY)] = CONFIG_VALUE(NO);
    config[VPU_COMPILER_CONFIG_KEY(GENERATE_BLOB)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(LOAD_NETWORK_AFTER_COMPILATION)] = CONFIG_VALUE(YES);

    InferenceEngine::ExecutableNetwork exeNetwork;
    ASSERT_ANY_THROW(exeNetwork = ie.LoadNetwork(network, "KMB", config));
}

// MCM compiler does not support multiple Input layers
// This test throws due to two inputs in the network
TEST_P(EltwiseTest, TestsEltwiseOnTwoDifferentInputsNegative_Test) {
    SizeVector output_dims;
    CommonTestUtils::getConvOutShape(input_dims(), conv_params(), output_dims);

    const size_t weightsByteSize = getConvWeightsSize(input_dims(), conv_params(), "U8");
    const size_t weightsSize = weightsByteSize / sizeof(uint8_t);

    const size_t biasByteSize = output_dims[1] * sizeof(int32_t);
    const size_t biasSize = biasByteSize / sizeof(int32_t);

    const auto weightsBuffer = make_shared_blob<uint8_t>({Precision::U8, {weightsByteSize + biasByteSize}, Layout::C});
    weightsBuffer->allocate();
    const auto weightsBufferData = weightsBuffer->buffer().as<uint8_t*>();
    std::fill_n(weightsBufferData, weightsSize, static_cast<uint8_t>(1));

    uint32_t* biasData = reinterpret_cast<uint32_t*>(weightsBuffer->buffer().as<uint8_t*>() + weightsSize);
    std::fill_n(biasData, biasSize, static_cast<uint32_t>(1));

    const convolution_test_desc allTestParams = {
        input_dims(), conv_params(), "U8", "U8", "U8", 0, "I32", "", elt_on_two_inputs_with_bias_template};
    const std::string model = instantiateConvTestIR(allTestParams);

    Core ie;
    CNNNetwork network = ie.ReadNetwork(model, weightsBuffer);

    auto _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::U8);
    _inputsInfo["input2"]->setPrecision(Precision::U8);

    auto _outputsInfo = network.getOutputsInfo();
    _outputsInfo["output"]->setPrecision(Precision::U8);

    std::map<std::string, std::string> config;
    setCommonConfig(config);
    config[VPU_COMPILER_CONFIG_KEY(PARSING_ONLY)] = CONFIG_VALUE(NO);
    config[VPU_COMPILER_CONFIG_KEY(GENERATE_BLOB)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(LOAD_NETWORK_AFTER_COMPILATION)] = CONFIG_VALUE(YES);

    InferenceEngine::ExecutableNetwork exeNetwork;
    ASSERT_ANY_THROW(exeNetwork = ie.LoadNetwork(network, "KMB", config));
}

static std::vector<eltwise_test_params> test_params = {
    {{1, 16, 16, 16}, {{1, 1}, {3, 3}, {0, 0}, {0, 0}, {1, 1}, "", 1, 128, true, true, ""}},
};

// MCM compiler does not support multiple Input layers
// ScaleShift result is used as the second input to Eltwise
TEST_P(EltwiseTest, DISABLED_TestsEltwiseAfterScaleShift) {
    SizeVector output_dims;
    CommonTestUtils::getConvOutShape(input_dims(), conv_params(), output_dims);

    const size_t weightsByteSize = getConvWeightsSize(input_dims(), conv_params(), "U8");
    const size_t weightsSize = weightsByteSize / sizeof(uint8_t);

    const size_t biasByteSize = output_dims[1] * sizeof(int32_t);
    const size_t biasSize = biasByteSize / sizeof(int32_t);

    const auto weightsBuffer = make_shared_blob<uint8_t>({Precision::U8, {weightsByteSize + biasByteSize}, Layout::C});
    weightsBuffer->allocate();
    const auto weightsBufferData = weightsBuffer->buffer().as<uint8_t*>();
    std::fill_n(weightsBufferData, weightsSize, static_cast<uint8_t>(1));

    uint32_t* biasData = reinterpret_cast<uint32_t*>(weightsBuffer->buffer().as<uint8_t*>() + weightsSize);
    std::fill_n(biasData, biasSize, static_cast<uint32_t>(1));

    const convolution_test_desc allTestParams = {
        input_dims(), conv_params(), "U8", "U8", "U8", 0, "I32", "", elt_on_input_and_scaleshift};

    const std::string model = instantiateConvTestIR(allTestParams);

    IE_SUPPRESS_DEPRECATED_START
    CNNNetReader reader;
    ASSERT_NO_THROW(reader.ReadNetwork(model.data(), model.length()));
    ASSERT_NO_THROW(reader.SetWeights(weightsBuffer));
    ASSERT_TRUE(reader.isParseSuccess());
    const CNNNetwork network = reader.getNetwork();
    IE_SUPPRESS_DEPRECATED_END

    auto _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::U8);

    auto _outputsInfo = network.getOutputsInfo();
    _outputsInfo["output"]->setPrecision(Precision::U8);

    // set up weights fot the scale shift layer
    const size_t ssWeightsByteSize = sizeof(uint8_t) * input_dims()[1];
    CNNLayerPtr cnl = network.getLayerByName("scale_shift1");
    std::shared_ptr<ScaleShiftLayer> ssl = std::static_pointer_cast<ScaleShiftLayer>(cnl);
    ssl->_weights.reset(
        new InferenceEngine::TBlob<uint8_t>(TensorDesc({Precision::U8, {ssWeightsByteSize}, Layout::C})));
    const auto ssWeightsBuffer = ssl->_weights;
    ssWeightsBuffer->allocate();
    const auto ssWeightsBufferData = ssWeightsBuffer->buffer().as<uint8_t*>();
    std::fill_n(ssWeightsBufferData, ssWeightsByteSize, static_cast<uint8_t>(1));

    std::map<std::string, std::string> config;
    setCommonConfig(config);
    config[VPU_COMPILER_CONFIG_KEY(PARSING_ONLY)] = CONFIG_VALUE(YES);
    config[VPU_COMPILER_CONFIG_KEY(GENERATE_BLOB)] = CONFIG_VALUE(YES);
    config[VPU_COMPILER_CONFIG_KEY(GENERATE_DOT)] = CONFIG_VALUE(YES);
    config[VPU_COMPILER_CONFIG_KEY(GENERATE_JSON)] = CONFIG_VALUE(YES);

    ASSERT_NO_THROW(ie.LoadNetwork(network, "kmb", config));
}

// corresponding ticket #-23769
TEST_F(kmbLayersTests_nightly, DISABLED_EltwiseWithFakeQuantize) {
    const std::string model = R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="EltwiseTest" version="6">
	<layers>
		<layer id="0" name="input.1" precision="U8" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>224</dim>
					<dim>224</dim>
				</port>
			</output>
		</layer>
        <layer id="183" name="PoolLayer" precision="U8" type="Pooling">
            <data kernel="1,1" pads_begin="0,0" pads_end="0,0" pool-method="max" rounding_type="floor" strides="1,1"/>
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
					<dim>224</dim>
					<dim>224</dim>
                </port>
            </output>
        </layer>
		<layer id="2" name="fq_1_1" precision="FP32" type="Const">
			<output>
				<port id="1"/>
			</output>
			<blobs>
				<custom offset="24" size="4"/>
			</blobs>
		</layer>
		<layer id="3" name="fq_1_2" precision="FP32" type="Const">
			<output>
				<port id="1"/>
			</output>
			<blobs>
				<custom offset="28" size="4"/>
			</blobs>
		</layer>
		<layer id="4" name="fq_1_3" precision="FP32" type="Const">
			<output>
				<port id="1"/>
			</output>
			<blobs>
				<custom offset="24" size="4"/>
			</blobs>
		</layer>
		<layer id="5" name="fq_1_4" precision="FP32" type="Const">
			<output>
				<port id="1"/>
			</output>
			<blobs>
				<custom offset="28" size="4"/>
			</blobs>
		</layer>
		<layer id="6" name="ELtwiseInput1" precision="FP32" type="FakeQuantize">
			<data levels="256.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>224</dim>
					<dim>224</dim>
				</port>
				<port id="1"/>
				<port id="2"/>
				<port id="3"/>
				<port id="4"/>
			</input>
			<output>
				<port id="5">
					<dim>1</dim>
					<dim>3</dim>
					<dim>224</dim>
					<dim>224</dim>
				</port>
			</output>
		</layer>
		<layer id="7" name="ReLU_2_Input" precision="U8" type="ReLU">
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
					<dim>224</dim>
					<dim>224</dim>
				</port>
			</output>
		</layer>
		<layer id="8" name="fq_2_1" precision="FP32" type="Const">
			<output>
				<port id="1"/>
			</output>
			<blobs>
				<custom offset="24" size="4"/>
			</blobs>
		</layer>
		<layer id="9" name="fq_2_2" precision="FP32" type="Const">
			<output>
				<port id="1"/>
			</output>
			<blobs>
				<custom offset="28" size="4"/>
			</blobs>
		</layer>
		<layer id="10" name="fq_2_3" precision="FP32" type="Const">
			<output>
				<port id="1"/>
			</output>
			<blobs>
				<custom offset="24" size="4"/>
			</blobs>
		</layer>
		<layer id="11" name="fq_2_4" precision="FP32" type="Const">
			<output>
				<port id="1"/>
			</output>
			<blobs>
				<custom offset="28" size="4"/>
			</blobs>
		</layer>
		<layer id="12" name="ELtwiseInput2" precision="FP32" type="FakeQuantize">
			<data levels="256.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>224</dim>
					<dim>224</dim>
				</port>
				<port id="1"/>
				<port id="2"/>
				<port id="3"/>
				<port id="4"/>
			</input>
			<output>
				<port id="5">
					<dim>1</dim>
					<dim>3</dim>
					<dim>224</dim>
					<dim>224</dim>
				</port>
			</output>
		</layer>
		<layer id="13" name="EltwiseSum" precision="FP32" type="Eltwise">
			<data operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>224</dim>
					<dim>224</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>3</dim>
					<dim>224</dim>
					<dim>224</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>3</dim>
					<dim>224</dim>
					<dim>224</dim>
				</port>
			</output>
		</layer>
		<layer id="14" name="fq_3_1" precision="FP32" type="Const">
			<output>
				<port id="1"/>
			</output>
			<blobs>
				<custom offset="24" size="4"/>
			</blobs>
		</layer>
		<layer id="15" name="fq_3_2" precision="FP32" type="Const">
			<output>
				<port id="1"/>
			</output>
			<blobs>
				<custom offset="28" size="4"/>
			</blobs>
		</layer>
		<layer id="16" name="fq_3_3" precision="FP32" type="Const">
			<output>
				<port id="1"/>
			</output>
			<blobs>
				<custom offset="24" size="4"/>
			</blobs>
		</layer>
		<layer id="17" name="fq_3_4" precision="FP32" type="Const">
			<output>
				<port id="1"/>
			</output>
			<blobs>
				<custom offset="28" size="4"/>
			</blobs>
		</layer>
		<layer id="18" name="ELtwiseOutputFQ" precision="FP32" type="FakeQuantize">
			<data levels="256.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>224</dim>
					<dim>224</dim>
				</port>
				<port id="1"/>
				<port id="2"/>
				<port id="3"/>
				<port id="4"/>
			</input>
			<output>
				<port id="5">
					<dim>1</dim>
					<dim>3</dim>
					<dim>224</dim>
					<dim>224</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="183" to-port="0"/>
		<edge from-layer="183" from-port="1" to-layer="6" to-port="0"/>
		<edge from-layer="2" from-port="1" to-layer="6" to-port="1"/>
		<edge from-layer="3" from-port="1" to-layer="6" to-port="2"/>
		<edge from-layer="4" from-port="1" to-layer="6" to-port="3"/>
		<edge from-layer="5" from-port="1" to-layer="6" to-port="4"/>
		<edge from-layer="0" from-port="0" to-layer="7" to-port="0"/>
		<edge from-layer="7" from-port="1" to-layer="12" to-port="0"/>
		<edge from-layer="8" from-port="1" to-layer="12" to-port="1"/>
		<edge from-layer="9" from-port="1" to-layer="12" to-port="2"/>
		<edge from-layer="10" from-port="1" to-layer="12" to-port="3"/>
		<edge from-layer="11" from-port="1" to-layer="12" to-port="4"/>
		<edge from-layer="6" from-port="5" to-layer="13" to-port="0"/>
		<edge from-layer="12" from-port="5" to-layer="13" to-port="1"/>
		<edge from-layer="13" from-port="2" to-layer="18" to-port="0"/>
		<edge from-layer="14" from-port="1" to-layer="18" to-port="1"/>
		<edge from-layer="15" from-port="1" to-layer="18" to-port="2"/>
		<edge from-layer="16" from-port="1" to-layer="18" to-port="3"/>
		<edge from-layer="17" from-port="1" to-layer="18" to-port="4"/>
	</edges>
	<meta_data>
		<MO_version value="unknown version"/>
		<cli_parameters>
			<unset unset_cli_parameters=""/>
		</cli_parameters>
	</meta_data>
</net>
        )V0G0N";

    std::size_t weightSize = 512;
    std::size_t biasSize = 0;
    TBlob<uint8_t>::Ptr weightsBlob(GenWeights<uint16_t>(weightSize + biasSize));

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());
    ASSERT_NO_THROW(_net_reader.SetWeights(weightsBlob));

    CNNNetwork network = _net_reader.getNetwork();

    std::map<std::string, std::string> config;
    setCommonConfig(config);
    config[VPU_COMPILER_CONFIG_KEY(PARSING_ONLY)] = CONFIG_VALUE(NO);

    ExecutableNetwork executableNetwork;
    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(network, "kmb", config));
}

INSTANTIATE_TEST_CASE_P(accuracy, EltwiseTest, ::testing::ValuesIn(test_params));

#endif
