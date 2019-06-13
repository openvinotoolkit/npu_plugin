// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "myriad_layers_tests.hpp"
#include <math.h>

using namespace InferenceEngine;

struct region_test_params {
    tensor_test_params in;
    int coords;
    int classes;
    int num;
    std::string customLayers;
    friend std::ostream& operator<<(std::ostream& os, region_test_params const& tst)
    {
        return os << "tensor (" << tst.in
                  << "),coords=" << tst.coords
                  << ", classes=" << tst.classes
                  << ", num=" << tst.num
                  << ", by using custom layer=" << (tst.customLayers.empty() ? "no" : "yes");
    };
};

class myriadLayerRegionYolo_nightly: public myriadLayersTests_nightly,
                             public testing::WithParamInterface<region_test_params> {
};

TEST_P(myriadLayerRegionYolo_nightly, BaseTestsRegion) {
    region_test_params p = ::testing::WithParamInterface<region_test_params>::GetParam();

    // TODO: M2 mode is not working for OpenCL compiler
    if(!p.customLayers.empty() && !CheckMyriadX()) {
        GTEST_SKIP()<<"Custom layers for MYRIAD2 not supported";
    }

    std::map<std::string, std::string> params;

    params["coords"] = std::to_string(p.coords);
    params["classes"] = std::to_string(p.classes);
    params["num"] = std::to_string(p.num);
    InferenceEngine::SizeVector tensor;
    tensor.resize(4);
    tensor[3] = p.in.w;
    tensor[2] = p.in.h;
    tensor[1] = p.in.c;
    tensor[0] = 1;
    _config[VPU_CONFIG_KEY(CUSTOM_LAYERS)] = p.customLayers;
    AddLayer("RegionYolo",
             &params,
             {tensor},
             {{1, tensor[0] * tensor[1] * tensor[2]* tensor[3]}},
             ref_RegionYolo_wrap);
    ASSERT_TRUE(GenerateNetAndInfer(false));
    Compare(_outputMap.begin()->second, GenReferenceOutput(), 0.0025);
}

static std::vector<region_test_params> s_regionData = {
    region_test_params{{1, 125, 13, 13}, 4, 20, 5, ""},
    region_test_params{{1, 425, 13, 13}, 4, 80, 5, ""},
    region_test_params{{1, 125, 13, 13}, 4, 20, 5, TestsCommon::get_data_path() + "/vpu/mvcl/customLayers_m7.xml"},
    // not supported by OpenCL implementation itself
    region_test_params{{1, 425, 13, 13}, 4, 80, 5, TestsCommon::get_data_path() + "/vpu/mvcl/customLayers_m7.xml"}
};

INSTANTIATE_TEST_CASE_P(
        accuracy, myriadLayerRegionYolo_nightly,
        ::testing::ValuesIn(s_regionData)
);

/* HW network needs to be created to test strides influence to RegionYolo input */
/* so convolution layer added as the first layer to this test                   */
class myriadLayersTestsRegion_CHW_HW_nightly: public ConvolutionTest<>{
};

/*80 input classes */
class myriadLayersTestsRegion_CHW_HW_80cl_nightly: public ConvolutionTest<>{
};

/* to passthrough "original" data */
void constWeightsRange_125(uint16_t* ptr, size_t weightsSize, size_t biasSize) {
    ASSERT_NE(ptr, nullptr);
    ASSERT_EQ(weightsSize, 125 * 125);
    std::memset(ptr, 0, sizeof(uint16_t) * (weightsSize + biasSize));
    for (int i = 0; i < weightsSize/125; ++i) {
        ptr[i * 125 + i] = PrecisionUtils::f32tof16(1.0f);
    }
}

/* to passthrough "original" data */
void constWeightsRange_425(uint16_t* ptr, size_t weightsSize, size_t biasSize) {
#define CONST_425 (425)
    ASSERT_NE(ptr, nullptr);

    ASSERT_EQ(weightsSize, CONST_425 * CONST_425);
    std::memset(ptr, 0, sizeof(uint16_t) * (weightsSize + biasSize));
    for (int i = 0; i < weightsSize/CONST_425; ++i) {
        ptr[i * CONST_425 + i] = PrecisionUtils::f32tof16(1.0f);
    }
}

void loadData(InferenceEngine::Blob::Ptr blob) {
    /* input blob has predefined size and CHW layout */
    ASSERT_NE(blob, nullptr);
    auto inDims = blob->dims();
    InferenceEngine::Blob::Ptr inputBlobRef =
            InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, InferenceEngine::NCHW, inDims);
    inputBlobRef->allocate();
    const float* ref_values = inputBlobRef->buffer();

    std::string inputTensorBinary = TestsCommon::get_data_path();
    inputTensorBinary += "/vpu/InputYoLoV2Tiny.bin";
    ASSERT_TRUE(fromBinaryFile(inputTensorBinary, inputBlobRef));
    uint16_t *inputBlobRawDataFp16 = static_cast<uint16_t *>(blob->buffer());
    ASSERT_NE(inputBlobRawDataFp16, nullptr);

    switch(blob->layout()) {
    case InferenceEngine::NCHW:
        for (int indx = 0; indx < blob->size(); indx++) {
            inputBlobRawDataFp16[indx] = PrecisionUtils::f32tof16(ref_values[indx]);
        }
        break;
    case InferenceEngine::NHWC:
        for (int h = 0 ; h < inDims[1]; ++h) {
            for (int w = 0 ; w < inDims[0]; ++w) {
                for (int c = 0 ; c < inDims[2]; ++c) {
                    int src_i = w + inDims[0] * h + inDims[0] * inDims[1] * c;
                    int dst_i = c + inDims[2] * w + inDims[0] * inDims[2] * h;
                    inputBlobRawDataFp16[dst_i] = PrecisionUtils::f32tof16(ref_values[src_i]);
                }
            }
        }
        break;
    }
}

void loadData_80cl(InferenceEngine::Blob::Ptr blob) {
    /* input blob has predefined size and CHW layout */
    ASSERT_NE(blob, nullptr);
    auto inDims = blob->dims();
    InferenceEngine::Blob::Ptr inputBlobRef =
            InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, InferenceEngine::NCHW, inDims);
    inputBlobRef->allocate();
    const float* ref_values = inputBlobRef->buffer();

    std::string inputTensorBinary = TestsCommon::get_data_path();
    inputTensorBinary += "/vpu/InputYoLoV2_80cl.bin";
    ASSERT_TRUE(fromBinaryFile(inputTensorBinary, inputBlobRef));
    uint16_t *inputBlobRawDataFp16 = static_cast<uint16_t *>(blob->buffer());
    ASSERT_NE(inputBlobRawDataFp16, nullptr);

    switch(blob->layout()) {
    case InferenceEngine::NCHW:
        for (int indx = 0; indx < blob->size(); indx++) {
            inputBlobRawDataFp16[indx] = PrecisionUtils::f32tof16(ref_values[indx]);
        }
        break;
    case InferenceEngine::NHWC:
        for (int h = 0 ; h < inDims[1]; ++h) {
            for (int w = 0 ; w < inDims[0]; ++w) {
                for (int c = 0 ; c < inDims[2]; ++c) {
                    int src_i = w + inDims[0] * h + inDims[0] * inDims[1] * c;
                    int dst_i = c + inDims[2] * w + inDims[0] * inDims[2] * h;
                    inputBlobRawDataFp16[dst_i] = PrecisionUtils::f32tof16(ref_values[src_i]);
                }
            }
        }
        break;
    }
}

TEST_P(myriadLayersTestsRegion_CHW_HW_nightly, RegionYolo) {
    std::map<std::string, std::string> params;
    params["coords"] = "4";
    params["classes"] = "20";
    params["num"] = "5";
    AddLayer("RegionYolo",
             &params,
             {_output_tensor},
             {{1, _output_tensor[0] * _output_tensor[1] * _output_tensor[2] * _output_tensor[3]}},
             ref_RegionYolo_wrap);
    _testNet[0].fillWeights = constWeightsRange_125;
    _genDataCallback = loadData;
    ASSERT_TRUE(GenerateNetAndInfer(true));
    Compare(_outputMap.begin()->second, GenReferenceOutput(), 0.0035);
}

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsRegion_CHW_HW_nightly,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({1, 125, 13, 13})
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<uint32_t>(125)
          , ::testing::Values<uint32_t>(1)
          )
);



TEST_P(myriadLayersTestsRegion_CHW_HW_80cl_nightly, RegionYolol) {
    std::map<std::string, std::string> params;
    params["coords"] = "4";
    params["classes"] = "80";
    params["num"] = "5";
    AddLayer("RegionYolo",
             &params,
             {_output_tensor},
             {{1, _output_tensor[0] * _output_tensor[1] * _output_tensor[2] * _output_tensor[3]}},
             ref_RegionYolo_wrap);
    _testNet[0].fillWeights = constWeightsRange_425;
    _genDataCallback = loadData_80cl;
    ASSERT_TRUE(GenerateNetAndInfer(true));
    Compare(_outputMap.begin()->second, GenReferenceOutput(), 0.0060);
}

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsRegion_CHW_HW_80cl_nightly,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({1, 425, 13, 13})
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<uint32_t>(425)
          , ::testing::Values<uint32_t>(1)
          )
);

class myriadLayerRegionYolo_CHW_nightly: public myriadLayersTests_nightly,
                             public testing::WithParamInterface<int> {
};

TEST_P(myriadLayerRegionYolo_CHW_nightly, TestsRegion) {
    auto classes = GetParam();
    InferenceEngine::SizeVector input_dims = {1, 125, 13, 13};
    if (classes == 80) {
        input_dims[1] = 425;
    }
    IN_OUT_desc input_tensor;
    input_tensor.push_back(input_dims);

    std::map<std::string, std::string> params;
    params["coords"] = "4";
    params["classes"] = std::to_string(classes);
    params["num"] = "5";
    AddLayer("RegionYolo",
             &params,
             input_tensor,
             {{1, input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3]}},
             ref_RegionYolo_wrap);
    _genDataCallback = loadData;
    if (classes == 80) {
        _genDataCallback = loadData_80cl;
    }
    _config[VPU_CONFIG_KEY(COMPUTE_LAYOUT)] = VPU_CONFIG_VALUE(NCHW);
    ASSERT_TRUE(GenerateNetAndInfer());
    /* bound is too high , set for M2 tests */
    Compare(_outputMap.begin()->second, GenReferenceOutput(), 0.006);
}

TEST_P(myriadLayerRegionYolo_CHW_nightly, Test_CHW_HWC_Compare) {
    // CVS-14246
    if (pluginName.find("HDDLPlugin") != std::string::npos) {
        SKIP() << "Disabled for HDDLPlugin." << std::endl;
    }
    auto classes = GetParam();
    IN_OUT_desc input_tensor;
    InferenceEngine::SizeVector input_dims = {1, 125, 13, 13};
    if (classes == 80) {
        input_dims[1] = 425;
    }

    input_tensor.push_back(input_dims);

    std::map<std::string, std::string> params;
    params["coords"] = "4";
    params["classes"] = std::to_string(classes);
    params["num"] = "5";
    AddLayer("RegionYolo",
             &params,
             input_tensor,
             {{1, input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3]}},
             ref_RegionYolo_wrap);
    if (classes == 80) {
        _genDataCallback = loadData_80cl;
    }
    _config[VPU_CONFIG_KEY(COMPUTE_LAYOUT)] = VPU_CONFIG_VALUE(NCHW);
    _config[VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION)] = CONFIG_VALUE(NO);
    ASSERT_TRUE(GenerateNetAndInfer(false, false));
    /* create  NHWC version                                */
    /* we cannot use the same GenerateNetAndInfer call due */
    /* to IE bug.                                          */
    InferenceEngine::InputsDataMap           inputsInfo;
    InferenceEngine::BlobMap                 outputMap;
    InferenceEngine::OutputsDataMap          outputsInfo;
    InferenceEngine::IExecutableNetwork::Ptr exeNetwork;
    InferenceEngine::IInferRequest::Ptr      inferRequest;
    InferenceEngine::ICNNNetwork &network = _net_reader.getNetwork();

    _config[VPU_CONFIG_KEY(COMPUTE_LAYOUT)] = VPU_CONFIG_VALUE(NHWC);
    InferenceEngine::StatusCode st = InferenceEngine::StatusCode::GENERAL_ERROR;
    ASSERT_NO_THROW(st = myriadPluginPtr->LoadNetwork(exeNetwork, network, _config, &_resp));
    ASSERT_NE(exeNetwork, nullptr) << _resp.msg;
    ASSERT_NO_THROW(exeNetwork->CreateInferRequest(inferRequest, &_resp)) << _resp.msg;
    ASSERT_NE(inferRequest, nullptr) << _resp.msg;
    ASSERT_NO_THROW(network.getInputsInfo(inputsInfo));
    auto inIt = _inputsInfo.begin();
    for (auto in = _inputsInfo.begin(); in != _inputsInfo.end(); in++) {
        Blob::Ptr inpt;
        ASSERT_NO_THROW(_inferRequest->GetBlob(inIt->first.c_str(), inpt, &_resp));
        ASSERT_NO_THROW(inferRequest->SetBlob(inIt->first.c_str(), inpt, &_resp));
        ++inIt;
    }
    ASSERT_NO_THROW(network.getOutputsInfo(outputsInfo));
    auto outIt = _outputsInfo.begin();
    for (auto outputInfo : outputsInfo) {
        outputInfo.second->setPrecision(outIt->second->getPrecision());
        InferenceEngine::SizeVector outputDims = outputInfo.second->dims;
        Blob::Ptr outputBlob = nullptr;
        Layout netLayout = outIt->second->getTensorDesc().getLayout();
        // work only with NHWC layout if size of the input dimensions == NHWC
        Layout layout = netLayout == NHWC || netLayout == NCHW? NHWC : netLayout;
        switch (outputInfo.second->getPrecision()) {
        case Precision::FP16:
            outputBlob = InferenceEngine::make_shared_blob<ie_fp16, const InferenceEngine::SizeVector>
                                                          (Precision::FP16, layout, outputDims);
            break;
        case Precision::FP32:
            outputBlob = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>
                                                          (Precision::FP32, layout, outputDims);
            break;
        default:
            THROW_IE_EXCEPTION << "Unsupported precision for output. Supported FP16, FP32";
        }
        outputBlob->allocate();
        InferenceEngine::StatusCode st = inferRequest->SetBlob(outputInfo.first.c_str(), outputBlob, &_resp);
        outputMap[outputInfo.first] = outputBlob;
        ASSERT_EQ((int) InferenceEngine::StatusCode::OK, st) << _resp.msg;
        ++outIt;
    }
    ASSERT_EQ(inferRequest->Infer(&_resp), InferenceEngine::OK);
    /* bound is too high !!!! investigation TBD */
    Compare(_outputMap.begin()->second, outputMap.begin()->second, 0.001);
}

const std::vector<int> s_classes = {20, 80};

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayerRegionYolo_CHW_nightly,
        ::testing::ValuesIn(s_classes)
);
