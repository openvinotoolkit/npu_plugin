// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"
#include "weights_for_test.h"

using std::tuple;
using std::get;

using namespace InferenceEngine;

PRETTY_PARAM(kernel, param_size);
PRETTY_PARAM(stride, param_size);
PRETTY_PARAM(pad, param_size);
PRETTY_PARAM(out_channels, int);
PRETTY_PARAM(group, int);
PRETTY_PARAM(dilation_factor, param_size);
PRETTY_PARAM(layout, const char*);

typedef myriadLayerTestBaseWithParam<tuple<DimsInput, kernel, stride, pad
        , out_channels, group, dilation_factor, layout >> myriadLayerConvolution_nightly;

typedef myriadLayerTestBaseWithParam<tuple<DimsInput, DimsOutput, kernel, stride, pad
        , group, dilation_factor, layout >> myriadLayerConvolutionTensorFlow_nightly;

TEST_P(myriadLayerConvolution_nightly, Convolution) {
    tensor_test_params input_dims = get<0>(GetParam());
    param_size kernel = get<1>(GetParam());
    param_size stride = get<2>(GetParam());
    param_size pad = get<3>(GetParam());
    size_t out_channels = get<4>(GetParam());
    size_t group = get<5>(GetParam());
    param_size dilation_factor = get<6>(GetParam());
    const char* layout = get<7>(GetParam());

    _config[VPU_CONFIG_KEY(COMPUTE_LAYOUT)] = layout;

    size_t out_w = (input_dims.w + 2 * pad.x - dilation_factor.x * (kernel.x - 1) - 1 + stride.x) / stride.x;
    size_t out_h = (input_dims.h + 2 * pad.y - dilation_factor.y * (kernel.y - 1) - 1 + stride.y) / stride.y;

    tensor_test_params output_dims = {1, out_channels, out_h, out_w};

    SetInputTensor(input_dims);
    SetOutputTensor(output_dims);

    size_t num_weights = kernel.x * kernel.y * (input_dims.c / group) * output_dims.c;
    size_t num_bias = output_dims.c;

    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr =
            InferenceEngine::TBlob<uint8_t>::Ptr(GenWeights(num_weights + num_bias));
    ie_fp16* weights = weights_ptr->data().as<ie_fp16*>();
    ie_fp16* bias = weights + num_weights;

    std::map<std::string, std::string> layer_params = {
              {"kernel-x", std::to_string(kernel.x)}
            , {"kernel-y", std::to_string(kernel.y)}
            , {"stride-x", std::to_string(stride.x)}
            , {"stride-y", std::to_string(stride.y)}
            , {"pad-x", std::to_string(pad.x)}
            , {"pad-y", std::to_string(pad.y)}
            , {"output", std::to_string(out_channels)}
            , {"group", std::to_string(group)}
            , {"dilation-x", std::to_string(dilation_factor.x)}
            , {"dilation-y", std::to_string(dilation_factor.y)}
    };
    NetworkInit("Convolution",
                  &layer_params,
                  num_weights * sizeof(uint16_t),
                  num_bias * sizeof(uint16_t),
                  weights_ptr,
                  InferenceEngine::Precision::FP16);
    SetFirstInputToRange(-0.9f, 0.9f);

    ASSERT_TRUE(Infer());
    auto inputBlob = _inputMap.begin()->second;
    auto outputBlob = _outputMap.begin()->second;

    ref_convolution(inputBlob, _refBlob, weights, bias, kernel, stride, pad, group, dilation_factor);

    float maxerr = 0;

    if (group == 1)
        maxerr = 0.00055 * input_dims.c * kernel.x * kernel.y;
    else // TODO: currently dephConv is slightly less accurate
        maxerr = 0.00066 * (input_dims.c / group) * kernel.x * kernel.y;

    Compare(outputBlob, _refBlob, maxerr);
}

TEST_P(myriadLayerConvolutionTensorFlow_nightly, Convolution) {
    tensor_test_params input_dims = get<0>(GetParam());
    tensor_test_params output_dims = get<1>(GetParam());
    param_size kernel = get<2>(GetParam());
    param_size stride = get<3>(GetParam());
    param_size pad = get<4>(GetParam());
    size_t group = get<5>(GetParam());
    param_size dilation_factor = get<6>(GetParam());

    SetInputTensor(input_dims);
    SetOutputTensor(output_dims);

    size_t num_weights = kernel.x * kernel.y * (input_dims.c / group) * output_dims.c;
    size_t num_bias = output_dims.c;

    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr =
            InferenceEngine::TBlob<uint8_t>::Ptr(GenWeights(num_weights + num_bias));
    ie_fp16* weights = weights_ptr->data().as<ie_fp16*>();
    ie_fp16* bias = weights + num_weights;

    std::map<std::string, std::string> layer_params = {
              {"kernel-x", std::to_string(kernel.x)}
            , {"kernel-y", std::to_string(kernel.y)}
            , {"stride-x", std::to_string(stride.x)}
            , {"stride-y", std::to_string(stride.y)}
            , {"pad-x", std::to_string(pad.x)}
            , {"pad-y", std::to_string(pad.y)}
            , {"output", std::to_string(output_dims.c)}
            , {"group", std::to_string(group)}
            , {"dilation-x", std::to_string(dilation_factor.x)}
            , {"dilation-y", std::to_string(dilation_factor.y)}
    };
    NetworkInit("Convolution",
                  &layer_params,
                  num_weights * sizeof(uint16_t),
                  num_bias * sizeof(uint16_t),
                  weights_ptr,
                  InferenceEngine::Precision::FP16);
    SetFirstInputToRange(-0.9f, 0.9f);
    ASSERT_TRUE(Infer());
    auto inputBlob = _inputMap.begin()->second;
    auto outputBlob = _outputMap.begin()->second;

    ref_convolution(inputBlob, _refBlob, weights, bias, kernel, stride, pad, group, dilation_factor);

    float maxerr = 0;

    maxerr = 0.00055 * (input_dims.c / group) * kernel.x * kernel.y;

    Compare(outputBlob, _refBlob, maxerr);
}

INSTANTIATE_TEST_CASE_P(accuracy_chw_dilation, myriadLayerConvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 37, 43, 43)
                                       , MAKE_STRUCT(tensor_test_params, 1, 37, 19, 19))
          , ::testing::Values<kernel>(
                                      MAKE_STRUCT(param_size, 3, 3)
                                     )
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)
                                    , MAKE_STRUCT(param_size, 2, 2)
                                    , MAKE_STRUCT(param_size, 3, 3)
                                    )
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 3, 2))
          , ::testing::Values<out_channels>(24)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 6, 5))
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NCHW))
          )
);

void FillWeights(uint16_t* ptr, size_t weightsSize, size_t biasSize) {
    ASSERT_NE(ptr, nullptr);
    auto szW = sizeof(s_3X3X3YOLO_Weights)/sizeof(s_3X3X3YOLO_Weights[0]);
    auto szB = sizeof(s_3X3X3YOLO_Biases)/sizeof(s_3X3X3YOLO_Biases[0]);
    ASSERT_EQ(szW, weightsSize);
    ASSERT_EQ(szB, biasSize);
    auto sz = szW + szB;
    size_t indx = 0;
    for (; indx < szW; ++indx) {
        ptr[indx] = PrecisionUtils::f32tof16(s_3X3X3YOLO_Weights[indx]);
    }
    for (; indx < sz; ++indx) {
        ptr[indx] = PrecisionUtils::f32tof16(s_3X3X3YOLO_Biases[indx - szW]);
    }
}

void loadConstData(InferenceEngine::Blob::Ptr blob) {
    /* input blob has predefined size and CHW layout */
    ASSERT_NE(blob, nullptr);
    auto inDims = blob->dims();
    uint16_t *inputBlobRawDataFp16 = static_cast<uint16_t *>(blob->buffer());
    ASSERT_NE(inputBlobRawDataFp16, nullptr);

    for (int indx = 0; indx < blob->size(); indx++) {
        inputBlobRawDataFp16[indx] = PrecisionUtils::f32tof16(128.0);
    }
}

class myriadLayers_3X3X3_ConstInput_nightly: public ConvolutionTest<const char*>{
};

TEST_P(myriadLayers_3X3X3_ConstInput_nightly, Convolution) {
    auto p = ::testing::WithParamInterface<std::tuple<InferenceEngine::SizeVector, param_size, param_size, param_size, uint32_t, uint32_t, const char*>>::GetParam();
    _config[VPU_CONFIG_KEY(COMPUTE_LAYOUT)] = std::get<6>(p);
    _testNet[0].fillWeights = FillWeights;
    _genDataCallback = loadConstData;
    ASSERT_TRUE(GenerateNetAndInfer(false));
    auto outputBlob = _outputMap.begin()->second;
    const uint16_t *res_ptr = outputBlob->buffer().as<const uint16_t*>();
    size_t res_size = outputBlob->size();

    size_t N = outputBlob->getTensorDesc().getDims()[0];
    size_t C = outputBlob->getTensorDesc().getDims()[1];
    size_t H = outputBlob->getTensorDesc().getDims()[2];
    size_t W = outputBlob->getTensorDesc().getDims()[3];

    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            auto ref_offs = outputBlob->getTensorDesc().getLayout() == NCHW ?
                            1 + 1*W + c*W*H + n*W*H*C : c + 1*C + 1*C*W + n*W*H*C;
            float ref_val = PrecisionUtils::f16tof32(res_ptr[ref_offs]);
            for (size_t h = 1; h < H - 1; h++) {
                for (size_t w = 1; w < W - 1; w++) {
                    size_t actualIdx = outputBlob->getTensorDesc().getLayout() == NCHW ?
                                        w + h*W + c*W*H + n*W*H*C : c + w*C + h*C*W + n*W*H*C;
                    float cur_val = PrecisionUtils::f16tof32(res_ptr[actualIdx]);
                    ASSERT_FLOAT_EQ(cur_val, ref_val);
                }
            }
        }
    }
    /* to check max error */
    Compare(_outputMap.begin()->second, GenReferenceOutput(), 0.02);
}

/* IR version 3 tests, main difference is a changes in padding parameters definitions */
typedef std::tuple<InferenceEngine::SizeVector, param_size, param_size, param_size, param_size, uint32_t, uint32_t> IR3_params;

class myriadLayers_IR3_ConvTests_nightly: public myriadLayersTests_nightly, /*input tensor, kernel, stride, pads_begin, pads_end, out_channel, group */
                                  public testing::WithParamInterface<IR3_params> {
};

TEST_P(myriadLayers_IR3_ConvTests_nightly, Conv) {
    std::map<std::string, std::string> params;
    InferenceEngine::SizeVector output_tensor;
    int32_t IW = 0;
    int32_t IH = 0;
    int32_t IC = 0;
    int32_t I_N = 0;
    size_t  group = 0;


    auto p = ::testing::WithParamInterface<IR3_params>::GetParam();
    auto input_tensor = std::get<0>(p);
    param_size kernel = std::get<1>(p);
    param_size stride = std::get<2>(p);
    param_size pads_begin = std::get<3>(p);
    param_size pads_end = std::get<4>(p);
    size_t out_channels = std::get<5>(p);
    group = std::get<6>(p);
    get_dims(input_tensor, IW, IH, IC, I_N);
    if (I_N > 1)
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);
    else
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(YES);
    size_t out_w = (IW + pads_begin.x + pads_end.x - kernel.x + stride.x) / stride.x;
    size_t out_h = (IH + pads_begin.y + pads_end.y - kernel.y + stride.y) / stride.y;
    gen_dims(output_tensor, input_tensor.size(), out_w, out_h, out_channels, I_N);

    size_t num_weights = kernel.x * kernel.y * (IC / group) * out_channels;
    size_t num_bias    = out_channels;

    std::string padsB = std::to_string(pads_begin.x) + ",";
    padsB += std::to_string(pads_begin.y);
    std::string padsE = std::to_string(pads_end.x) + ",";
    padsE += std::to_string(pads_end.y);
    std::string strides = std::to_string(stride.x) + ",";
    strides += std::to_string(stride.y);
    std::string kern = std::to_string(kernel.x) + ",";
    kern += std::to_string(kernel.y);

    std::map<std::string, std::string> layer_params = {
              {"kernel",     kern}
            , {"strides",    strides}
            , {"pads_begin", padsB}
            , {"pads_end",   padsE}
            , {"output", std::to_string(out_channels)}
            , {"group", std::to_string(group)}
            , {"auto_pad", "same_upper"}
            , {"dilations", "1,1"}
    };
    AddLayer("Convolution",
             &layer_params,
             num_weights,
             num_bias,
             defaultWeightsRange,
             {input_tensor},
             {output_tensor},
             ref_convolution_wrap);
    ASSERT_TRUE(GenerateNetAndInfer(CheckMyriadX(), true, 3));
    float maxerr = 0.0009f * (IC / group) * kernel.x * kernel.y;
    Compare(_outputMap.begin()->second, GenReferenceOutput(), maxerr);
}

class myriadLayers_BatchTest_ConvTests_nightly: public myriadLayersTests_nightly, /*input tensor, kernel, stride, pads_begin, pads_end, out_channel, group */
                                        public testing::WithParamInterface<IR3_params> {
};

class myriadLayers_BatchTest2_ConvTests_nightly: public myriadLayersTests_nightly, /*input tensor, kernel, stride, pads_begin, pads_end, out_channel, group */
                                         public testing::WithParamInterface<IR3_params> {
};

void constWeightsRange(uint16_t* ptr, size_t weightsSize, size_t biasSize) {
    ASSERT_NE(ptr, nullptr);
    float shft = 0.0011f;
    float val = 0.125f;
    for (size_t count = 0 ; count < (weightsSize + biasSize); ++count) {
        
        ptr[count] = PrecisionUtils::f32tof16(val);
        val += shft;
        if (val > 1.)
            val = -1.0f;
    }
}

static void genTestData(InferenceEngine::Blob::Ptr blob) {
    ASSERT_NE(blob, nullptr);
    Layout layout = blob->layout();
    SizeVector dims = blob->getTensorDesc().getDims();

    ie_fp16* ptr = blob->buffer().as<ie_fp16*>();
    if (layout == NCHW || layout == NHWC) {
        size_t N = dims[0];
        size_t C = dims[1];
        size_t H = dims[2];
        size_t W = dims[3];

        float counter = 0.125f;
        for (size_t n = 0; n < N; n++) {
            for (size_t c = 0; c < C; c++) {
                for (size_t h = 0; h < H; h++) {
                    for (size_t w = 0; w < W; w++) {
                        size_t actualIdx = layout == NCHW ?
                                           w + h * W + c * W * H + n * W * H * C : c + w * C + h * C * W +
                                                                                   n * W * H * C;
                        ptr[actualIdx] = PrecisionUtils::f32tof16(counter);
                        counter += 0.025f;
                        if (counter > .90f) {
                            counter = -.90f;
                        }
                    }
                }
            }
        }
    } else {
        ASSERT_TRUE(false);
    }
}

TEST_P(myriadLayers_BatchTest_ConvTests_nightly, Conv) {
    std::map<std::string, std::string> params;
    InferenceEngine::SizeVector output_tensor;
    int32_t IW = 0;
    int32_t IH = 0;
    int32_t IC = 0;
    int32_t I_N = 0;
    size_t  group = 0;

    auto p = ::testing::WithParamInterface<IR3_params>::GetParam();
    auto input_tensor = std::get<0>(p);
    param_size kernel = std::get<1>(p);
    param_size stride = std::get<2>(p);
    param_size pads_begin = std::get<3>(p);
    param_size pads_end = std::get<4>(p);
    size_t out_channels = std::get<5>(p);
    group = std::get<6>(p);
    get_dims(input_tensor, IW, IH, IC, I_N);
    if (I_N > 1)
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);
    else
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(YES);
    size_t out_w = (IW + pads_begin.x + pads_end.x - kernel.x + stride.x) / stride.x;
    size_t out_h = (IH + pads_begin.y + pads_end.y - kernel.y + stride.y) / stride.y;
    gen_dims(output_tensor, input_tensor.size(), out_w, out_h, out_channels, I_N);

    size_t num_weights = kernel.x * kernel.y * (IC / group) * out_channels;
    size_t num_bias    = out_channels;

    std::string padsB   = gen_param(pads_begin);
    std::string padsE   = gen_param(pads_end);
    std::string strides = gen_param(stride);
    std::string kern    = gen_param(kernel);

    std::map<std::string, std::string> layer_params = {
              {"kernel",     kern}
            , {"strides",    strides}
            , {"pads_begin", padsB}
            , {"pads_end",   padsE}
            , {"output", std::to_string(out_channels)}
            , {"group", std::to_string(group)}
            , {"auto_pad", "same_upper"}
            , {"dilations", "1,1"}
    };
    _genDataCallback = genTestData;
    AddLayer("Convolution",
             &layer_params,
             num_weights,
             num_bias,
             constWeightsRange,
             {input_tensor},
             {output_tensor},
             ref_convolution_wrap);

    ASSERT_TRUE(GenerateNetAndInfer(CheckMyriadX(), true, 3));
    float maxerr = 0.0009f * (IC / group) * kernel.x * kernel.y;
    Compare(_outputMap.begin()->second, GenReferenceOutput(), maxerr);
}

static const std::string MODEL_RFCNN = R"V0G0N(
<net name="MODEL_TEST" version="3" batch="10">
    <layers>
        <layer id="0" name="input" precision="FP16" type="Input">
			<output>
				<port id="0">
					<dim>10</dim>
					<dim>576</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</output>
		</layer>
		<layer id="142" name="MaxPool2D/MaxPool" precision="FP16" type="Pooling">
			<data auto_pad="valid" exclude-pad="true" kernel="2,2" pads_begin="0,0" pads_end="0,0" pool-method="max" strides="2,2"/>
			<input>
				<port id="0">
					<dim>10</dim>
					<dim>576</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>10</dim>
					<dim>576</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</output>
		</layer>
		<layer id="143" name="SecondStageFeatureExtractor/InceptionV2/Mixed_5a/Branch_0/Conv2d_0a_1x1/Conv2D" precision="FP16" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>10</dim>
					<dim>576</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>10</dim>
					<dim>128</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="147456"/>
				<biases offset="147456" size="256"/>
			</blobs>
		</layer>
		<layer id="144" name="SecondStageFeatureExtractor/InceptionV2/Mixed_5a/Branch_0/Conv2d_0a_1x1/Relu" precision="FP16" type="ReLU">
			<input>
				<port id="0">
					<dim>10</dim>
					<dim>128</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>10</dim>
					<dim>128</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</output>
		</layer>
		<layer id="145" name="SecondStageFeatureExtractor/InceptionV2/Mixed_5a/Branch_0/Conv2d_1a_3x3/Conv2D" precision="FP16" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="3,3" output="192" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>10</dim>
					<dim>128</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>10</dim>
					<dim>192</dim>
					<dim>4</dim>
					<dim>4</dim>
				</port>
			</output>
			<blobs>
				<weights offset="147712" size="442368"/>
				<biases offset="590080" size="384"/>
			</blobs>
		</layer>
		<layer id="146" name="SecondStageFeatureExtractor/InceptionV2/Mixed_5a/Branch_0/Conv2d_1a_3x3/Relu" precision="FP16" type="ReLU">
			<input>
				<port id="0">
					<dim>10</dim>
					<dim>192</dim>
					<dim>4</dim>
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>10</dim>
					<dim>192</dim>
					<dim>4</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="147" name="SecondStageFeatureExtractor/InceptionV2/Mixed_5a/Branch_1/Conv2d_0a_1x1/Conv2D" precision="FP16" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="192" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>10</dim>
					<dim>576</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>10</dim>
					<dim>192</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</output>
			<blobs>
				<weights offset="590464" size="221184"/>
				<biases offset="811648" size="384"/>
			</blobs>
		</layer>
		<layer id="148" name="SecondStageFeatureExtractor/InceptionV2/Mixed_5a/Branch_1/Conv2d_0a_1x1/Relu" precision="FP16" type="ReLU">
			<input>
				<port id="0">
					<dim>10</dim>
					<dim>192</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>10</dim>
					<dim>192</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</output>
		</layer>
		<layer id="149" name="SecondStageFeatureExtractor/InceptionV2/Mixed_5a/Branch_1/Conv2d_0b_3x3/Conv2D" precision="FP16" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="3,3" output="256" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>10</dim>
					<dim>192</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>10</dim>
					<dim>256</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</output>
			<blobs>
				<weights offset="812032" size="884736"/>
				<biases offset="1696768" size="512"/>
			</blobs>
		</layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="142" to-port="0"/>
            <edge from-layer="142" from-port="1" to-layer="143" to-port="0"/>
            <edge from-layer="143" from-port="3" to-layer="144" to-port="0"/>
            <edge from-layer="144" from-port="1" to-layer="145" to-port="0"/>
            <edge from-layer="145" from-port="3" to-layer="146" to-port="0"/>
            <edge from-layer="142" from-port="1" to-layer="147" to-port="0"/>
            <edge from-layer="147" from-port="3" to-layer="148" to-port="0"/>
            <edge from-layer="148" from-port="1" to-layer="149" to-port="0"/>
        </edges>
    </net>
)V0G0N";

TEST_F(myriadLayersTests_nightly, tests125) {
    std::string outName1 = "SecondStageFeatureExtractor/InceptionV2/Mixed_5a/Branch_1/Conv2d_0b_3x3/Conv2D";
    std::string outName2 = "SecondStageFeatureExtractor/InceptionV2/Mixed_5a/Branch_0/Conv2d_1a_3x3/Relu";
    InferenceEngine::TBlob<uint8_t>::Ptr weights(GenWeights(1697280 / sizeof(ie_fp16)));
    constWeightsRange(weights->data().as<uint16_t *>(), 1697280 / sizeof(ie_fp16), 0);

    StatusCode st;
    InferenceEngine::CNNNetReader            net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(MODEL_RFCNN.data(), MODEL_RFCNN.length()));
    ASSERT_NO_THROW(net_reader.SetWeights(weights));
    ASSERT_TRUE(net_reader.isParseSuccess());

    auto network = net_reader.getNetwork();

    auto inputsInfo = network.getInputsInfo();
    inputsInfo["input"]->setInputPrecision(Precision::FP16);

    auto outputsInfo = network.getOutputsInfo();
    outputsInfo[outName1]->setPrecision(Precision::FP16);
    outputsInfo[outName2]->setPrecision(Precision::FP16);

    InferenceEngine::IExecutableNetwork::Ptr exeNetwork;
    ASSERT_NO_THROW(st = myriadPluginPtr->LoadNetwork(exeNetwork, network, {}, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    InferenceEngine::IInferRequest::Ptr     inferRequest;
    ASSERT_NO_THROW(st = exeNetwork->CreateInferRequest(inferRequest, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    Blob::Ptr input;
    ASSERT_NO_THROW(st = inferRequest->GetBlob("input", input, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    genTestData(input);
    
    ASSERT_NO_THROW(st = inferRequest->Infer(&_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    Blob::Ptr out1;
    Blob::Ptr out2;
    ASSERT_NO_THROW(st = inferRequest->GetBlob(outName1.c_str(), out1, &_resp));
    ASSERT_NO_THROW(st = inferRequest->GetBlob(outName2.c_str(), out2, &_resp));
};

TEST_P(myriadLayers_BatchTest2_ConvTests_nightly, Conv) {
    std::map<std::string, std::string> params;
    InferenceEngine::SizeVector output_tensor;
    int32_t IW = 0;
    int32_t IH = 0;
    int32_t IC = 0;
    int32_t I_N = 0;
    size_t  group = 0;

    auto p = ::testing::WithParamInterface<IR3_params>::GetParam();
    auto input_tensor = std::get<0>(p);
    param_size kernel = std::get<1>(p);
    param_size stride = std::get<2>(p);
    param_size pads_begin = std::get<3>(p);
    param_size pads_end = std::get<4>(p);
    size_t out_channels = std::get<5>(p);
    group = std::get<6>(p);
    get_dims(input_tensor, IW, IH, IC, I_N);
    if (I_N > 1)
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);
    else
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(YES);
    size_t out_w = (IW + pads_begin.x + pads_end.x - kernel.x + stride.x) / stride.x;
    size_t out_h = (IH + pads_begin.y + pads_end.y - kernel.y + stride.y) / stride.y;
    gen_dims(output_tensor, input_tensor.size(), out_w, out_h, out_channels, I_N);

    size_t num_weights = kernel.x * kernel.y * (IC / group) * out_channels;
    size_t num_bias    = out_channels;

    std::string padsB   = gen_param(pads_begin);
    std::string padsE   = gen_param(pads_end);
    std::string strides = gen_param(stride);
    std::string kern    = gen_param(kernel);

    std::map<std::string, std::string> layer_params = {
              {"kernel",     kern}
            , {"strides",    strides}
            , {"pads_begin", padsB}
            , {"pads_end",   padsE}
            , {"output", std::to_string(out_channels)}
            , {"group", std::to_string(group)}
            , {"auto_pad", "same_upper"}
            , {"dilations", "1,1"}
    };
    _genDataCallback = genTestData;
    AddLayer("Convolution",
             &layer_params,
             num_weights,
             num_bias,
             constWeightsRange,
             {input_tensor},
             {output_tensor},
             ref_convolution_wrap);
    AddLayer("ReLU", nullptr,
             {output_tensor},
             {output_tensor},
             ref_ReLU_wrap);

    std::map<std::string, std::string> conv2_params = {
              {"kernel",     "3,3"}
            , {"strides",    "1,1"}
            , {"pads_begin", "1,1"}
            , {"pads_end",   "1,1"}
            , {"output", "256"}
            , {"group", "1"}
            , {"auto_pad", "same_upper"}
            , {"dilations", "1,1"}
    };
    AddLayer("Convolution",
             &conv2_params,
             442368,
             256,
             constWeightsRange,
             {output_tensor},
             {{10, 256, 7, 7}},
             ref_convolution_wrap);
    AddLayer("ReLU", nullptr,
             {{10, 256, 7, 7}},
             {{10, 256, 7, 7}},
             ref_ReLU_wrap);

    ASSERT_TRUE(GenerateNetAndInfer(CheckMyriadX(), true, 3));
    // Error is calculated for sum of 2 convolutions
    float maxerr = 0.001f * (IC + 256) * kernel.x * kernel.y * 9;
    Compare(_outputMap.begin()->second, GenReferenceOutput(), maxerr);
}

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayers_IR3_ConvTests_nightly,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({1, 3, 32, 24})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<uint32_t>(12)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_Batch_0, myriadLayers_BatchTest_ConvTests_nightly,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({10, 5, 1, 1})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<uint32_t>(5)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_Batch_1, myriadLayers_BatchTest_ConvTests_nightly,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({10, 576, 7, 7})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<uint32_t>(128)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_Batch_2, myriadLayers_BatchTest_ConvTests_nightly,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({10, 128, 7, 7})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<uint32_t>(192)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_Batch_3, myriadLayers_BatchTest_ConvTests_nightly,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({10, 4, 7, 7})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<uint32_t>(4)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_Batch_4, myriadLayers_BatchTest_ConvTests_nightly,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({10, 256, 7, 7})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<uint32_t>(256)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_Batch_5, myriadLayers_BatchTest_ConvTests_nightly,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({10, 1024, 4, 4})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<uint32_t>(352)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_Batch_6, myriadLayers_BatchTest_ConvTests_nightly,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({10, 192, 4, 4})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<uint32_t>(320)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_Batch_7, myriadLayers_BatchTest_ConvTests_nightly,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({10, 160, 4, 4})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<uint32_t>(224)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_Batch_8, myriadLayers_BatchTest_ConvTests_nightly,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({10, 224, 4, 4})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<uint32_t>(224)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_Batch_9, myriadLayers_BatchTest_ConvTests_nightly,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({10, 1024, 4, 4})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<uint32_t>(128)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_Batch_10, myriadLayers_BatchTest_ConvTests_nightly,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({1, 64, 56, 56})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<uint32_t>(192)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_Batch_11, myriadLayers_BatchTest_ConvTests_nightly,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({10, 192, 7, 7})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<uint32_t>(256)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_Batch_12, myriadLayers_BatchTest_ConvTests_nightly,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({10, 576, 7, 7})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<uint32_t>(192)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_Batch_1, myriadLayers_BatchTest2_ConvTests_nightly,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({10, 576, 7, 7})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<uint32_t>(192)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_3X3, myriadLayers_IR3_ConvTests_nightly,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({1, 3, 32, 24})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<uint32_t>(12)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_3X1, myriadLayers_IR3_ConvTests_nightly,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({1, 3, 32, 24})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 1))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 0))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 0))
          , ::testing::Values<uint32_t>(16)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_1X3, myriadLayers_IR3_ConvTests_nightly,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({1, 4, 16, 16})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 1))
          , ::testing::Values<uint32_t>(8)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayers_3X3X3_ConstInput_nightly,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({1, 3, 10, 10})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<uint32_t>(32)
          , ::testing::Values<uint32_t>(1)
          , ::testing::ValuesIn(g_poolingLayout) // this array keeps possible layouts
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_crossroad_spatialConv, myriadLayerConvolutionTensorFlow_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 3, 1024, 1024))
          , ::testing::Values<DimsOutput>(MAKE_STRUCT(tensor_test_params, 1, 3, 512, 512))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 7, 7))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC))
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_inception_v2, myriadLayerConvolutionTensorFlow_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 16, 28, 28))
          , ::testing::Values<DimsOutput>(MAKE_STRUCT(tensor_test_params, 1, 64, 14, 14))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC))
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_inception_v1, myriadLayerConvolutionTensorFlow_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 3, 224, 224),
                                         MAKE_STRUCT(tensor_test_params, 1, 32, 224, 224)
            )
          , ::testing::Values<DimsOutput>(MAKE_STRUCT(tensor_test_params, 1, 64, 112, 112))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 7, 7))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC))
          )
);

INSTANTIATE_TEST_CASE_P(test_3x3_SSD_dilation, myriadLayerConvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 512, 19, 19))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 6, 6))
          , ::testing::Values<out_channels>(1024)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 6, 5))
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC))
          )
);

INSTANTIATE_TEST_CASE_P(test_TF_Resnet_50, myriadLayers_IR3_ConvTests_nightly,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({1, 512, 38, 38})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<out_channels>(128)
          , ::testing::Values<group>(1)
          )
);

INSTANTIATE_TEST_CASE_P(test_3x3_icvnet_dilation, myriadLayerConvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 24, 20, 20))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)
                                    , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1)
                                 , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<out_channels>(48)
          , ::testing::Values<group>(6, 8)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 2, 3))
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC))
          )
);

INSTANTIATE_TEST_CASE_P(test_5x5_with_dilation, myriadLayerConvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 32, 64,  77))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 5, 5))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)
                                    , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1)
                                 , MAKE_STRUCT(param_size, 0, 0)
                                 , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<out_channels>(32)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 2, 3), MAKE_STRUCT(param_size, 3, 4))
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC))
          )
);

INSTANTIATE_TEST_CASE_P(test_7x7_with_dilation, myriadLayerConvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 32, 64,  77))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 7, 7))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)
                                    , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1)
                                 , MAKE_STRUCT(param_size, 0, 0)
                                 , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<out_channels>(32)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 2, 3), MAKE_STRUCT(param_size, 3, 4))
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC))
          )
);


INSTANTIATE_TEST_CASE_P(test_conv1x1, myriadLayerConvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 10, 13, 13))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)
                                    , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<out_channels>(20)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC))
           )
);

INSTANTIATE_TEST_CASE_P(test_yolo_tiny_2_512x13x13_use_3x3_convolution, myriadLayerConvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 512, 13, 13))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<out_channels>(1024)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC))
           )
);

INSTANTIATE_TEST_CASE_P(test_yolo_tiny_2_512x13x13_use_1x1_convolution, myriadLayerConvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 4608, 13, 13))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<out_channels>(1024)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC))
           )
);

INSTANTIATE_TEST_CASE_P(DISABLED_accuracy_group_s0_postop, myriadLayerConvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 32, 112, 96))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<out_channels>(32)
          , ::testing::Values<group>(32, 1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC))
          )
);

INSTANTIATE_TEST_CASE_P(DISABLED_accuracy_group_s1_postop, myriadLayerConvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 32, 56, 48))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<out_channels>(32)
          , ::testing::Values<group>(32, 2)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC))
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_group, myriadLayerConvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 32, 64,  77)
                                       , MAKE_STRUCT(tensor_test_params, 1, 32, 112, 96))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3)
                                  , MAKE_STRUCT(param_size, 5, 5)
                                  , MAKE_STRUCT(param_size, 7, 7))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)
                                    , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1)
                                 , MAKE_STRUCT(param_size, 0, 0)
                                 , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<out_channels>(32)
          , ::testing::Values<group>(32)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC))
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_group_large_input, myriadLayerConvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 32, 192, 336))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1)
                                 , MAKE_STRUCT(param_size, 0, 0)
                                 , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<out_channels>(32)
          , ::testing::Values<group>(32)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC))
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_any_group, myriadLayerConvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 32, 64,  77))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)
                                    , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1)
                                 , MAKE_STRUCT(param_size, 0, 0)
                                 , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<out_channels>(32)
          , ::testing::Values<group>(2, 4, 16)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2), MAKE_STRUCT(param_size, 2, 3), MAKE_STRUCT(param_size, 3, 4))
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC), VPU_CONFIG_VALUE(NCHW))
          )
);

INSTANTIATE_TEST_CASE_P(set_optimization_for_3x3_with_group, myriadLayerConvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 24, 80, 80)
                                       , MAKE_STRUCT(tensor_test_params, 1, 36, 80, 80))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1),
                                      MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<out_channels>(24)
          , ::testing::Values<group>(6)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2), MAKE_STRUCT(param_size, 2, 3))
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC))
          )
);

INSTANTIATE_TEST_CASE_P(set_optimization_for_3x3s1, myriadLayerConvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 24, 80, 80))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<out_channels>(1,2,3,4,5,6,7,8)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC))
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_1x1, myriadLayerConvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 16, 64, 64)
                                       , MAKE_STRUCT(tensor_test_params, 1, 32, 1, 1))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)
                                    , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0),
                                   MAKE_STRUCT(param_size, 1, 1)
                                  )
          , ::testing::Values<out_channels>(16, 24)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC))
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_3x3, myriadLayerConvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 8, 16, 16)
                                       , MAKE_STRUCT(tensor_test_params, 1, 8, 59, 73))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)
                                    , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0)
                                 , MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<out_channels>(8, 15/*, 20 failed for 3x3s2*/, 32)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC))
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_1x3, myriadLayerConvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 8, 59, 73))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 3), MAKE_STRUCT(param_size, 3, 1))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0)
                                 , MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<out_channels>(7, 32)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC))
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_5x5, myriadLayerConvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 16, 32, 32)
                                     /*, MAKE_STRUCT(tensor_test_params, 1, 8, 511, 399) failed*/)
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 5, 5))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)
                                    , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0)
                               /*, MAKE_STRUCT(param_size, 1, 1) failed*/
                                 , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<out_channels>(16, 32)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC), VPU_CONFIG_VALUE(NCHW))
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_7x7, myriadLayerConvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 8, 32, 32)
                                     /*, MAKE_STRUCT(tensor_test_params, 1, 8, 511, 399) failed*/)
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 7, 7))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)
                                    , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0)
                               /*, MAKE_STRUCT(param_size, 1, 1) failed*/
                               /*, MAKE_STRUCT(param_size, 3, 3) failed*/)
          , ::testing::Values<out_channels>(16, 32)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC))
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_3x3_large_input_1, myriadLayerConvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 3, 720, 1280))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<out_channels>(8)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC))
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_3x3_large_input_2, myriadLayerConvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 24, 357, 637))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<out_channels>(8)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC))
          )
);


INSTANTIATE_TEST_CASE_P(accuracy_3x3_large_input_3, myriadLayerConvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 16, 359, 639))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<out_channels>(12)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC))
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_1x1_large_input, myriadLayerConvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 24, 355, 635))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<out_channels>(2, 3)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC))
          )
);

INSTANTIATE_TEST_CASE_P(accuracy_small_input_0, myriadLayerConvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 128, 38, 38))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<out_channels>(6)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC))
          )
);
INSTANTIATE_TEST_CASE_P(accuracy_small_input_1, myriadLayerConvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 256, 2, 3))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<out_channels>(8)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC))
          )
);
INSTANTIATE_TEST_CASE_P(accuracy_small_input_2, myriadLayerConvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 256, 2, 2))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<out_channels>(8)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC))
          )
);
INSTANTIATE_TEST_CASE_P(accuracy_small_input_3, myriadLayerConvolution_nightly,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 256, 1, 1))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<out_channels>(84)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<layout>(VPU_CONFIG_VALUE(NHWC))
           )
 );

// This tests checks that conv3x3s1 case doesn't corrupt its input.
// To check this we run two convolutions on the same input
// and check that they return same results.
TEST_F(myriadLayersTests_nightly, SmallConv_CorruptInputBug) {
    const std::string model = R"V0G0N(
        <Net name="SmallConv_CorruptInputBug" version="2" batch="1">
            <layers>
                <layer name="input" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </output>
                </layer>
                <layer name="input_copy" type="Power" precision="FP16" id="2">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="2">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv1" type="Convolution" precision="FP16" id="3">
                    <convolution_data stride-x="1" stride-y="1" pad-x="1" pad-y="1" kernel-x="3" kernel-y="3" output="84" group="1"/>
                    <input>
                        <port id="4">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </input>
                    <output>
                        <port id="5">
                            <dim>1</dim>
                            <dim>84</dim>
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </output>
                    <weights offset="0" size="387072"/>
                    <biases offset="387072" size="168"/>
                </layer>
                <layer name="conv2" type="Convolution" precision="FP16" id="4">
                    <convolution_data stride-x="1" stride-y="1" pad-x="1" pad-y="1" kernel-x="3" kernel-y="3" output="84" group="1"/>
                    <input>
                        <port id="6">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </input>
                    <output>
                        <port id="7">
                            <dim>1</dim>
                            <dim>84</dim>
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </output>
                    <weights offset="0" size="387072"/>
                    <biases offset="387072" size="168"/>
                </layer>
                <layer name="conv1_out" type="Power" precision="FP16" id="5">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="8">
                            <dim>1</dim>
                            <dim>84</dim>
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </input>
                    <output>
                        <port id="9">
                            <dim>1</dim>
                            <dim>84</dim>
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv2_out" type="Power" precision="FP16" id="6">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="10">
                            <dim>1</dim>
                            <dim>84</dim>
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </input>
                    <output>
                        <port id="11">
                            <dim>1</dim>
                            <dim>84</dim>
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="1" to-layer="2" to-port="2"/>
                <edge from-layer="2" from-port="3" to-layer="3" to-port="4"/>
                <edge from-layer="2" from-port="3" to-layer="4" to-port="6"/>
                <edge from-layer="3" from-port="5" to-layer="5" to-port="8"/>
                <edge from-layer="4" from-port="7" to-layer="6" to-port="10"/>
            </edges>
        </Net>
    )V0G0N";

    SetSeed(DEFAULT_SEED_VALUE);

    size_t num_weights = 193536;
    size_t num_bias = 84;

    TBlob<uint8_t>::Ptr weightsBlob(GenWeights(num_weights + num_bias));
    const ie_fp16 *weights = weightsBlob->readOnly().as<const ie_fp16 *>();
    const ie_fp16 *bias = weights + num_weights;

    StatusCode st;

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());
    ASSERT_NO_THROW(_net_reader.SetWeights(weightsBlob));

    auto network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setInputPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["conv1_out"]->setPrecision(Precision::FP16);
    _outputsInfo["conv2_out"]->setPrecision(Precision::FP16);

    ASSERT_NO_THROW(st = myriadPluginPtr->LoadNetwork(_exeNetwork, network, {}, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = _exeNetwork->CreateInferRequest(_inferRequest, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    Blob::Ptr input;
    ASSERT_NO_THROW(st = _inferRequest->GetBlob("input", input, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    {
        ie_fp16 *dst = input->buffer().as<ie_fp16 *>();
        for (int i = 0; i < input->size(); ++i) {
            float val = static_cast<float>(std::rand()) / RAND_MAX;
            dst[i] = PrecisionUtils::f32tof16(val);
        }
    }

    ASSERT_NO_THROW(st = _inferRequest->Infer(&_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    Blob::Ptr conv1;
    ASSERT_NO_THROW(_inferRequest->GetBlob("conv1_out", conv1, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    Blob::Ptr conv2;
    ASSERT_NO_THROW(_inferRequest->GetBlob("conv2_out", conv2, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    {
        SCOPED_TRACE("Compare with itself");
        Compare(conv1, conv2, 0.0);
    }

    {
        SCOPED_TRACE("Compare with reference");

        _refBlob = make_shared_blob<ie_fp16, const SizeVector>(Precision::FP16, Layout::NHWC, conv1->dims());
        _refBlob->allocate();
        ref_convolution(input, _refBlob, weights, bias, {3, 3}, {1, 1}, {1, 1}, 1);

        Compare(conv1, _refBlob, 0.1);
        Compare(conv2, _refBlob, 0.1);
    }
}
