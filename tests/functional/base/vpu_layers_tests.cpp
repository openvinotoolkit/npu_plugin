// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu_layers_tests.hpp"

#include <chrono>
#include <iostream>
#include <thread>
#include <vpu/kmb_plugin_config.hpp>

#include "blob_factory.hpp"
#include "ie_memcpy.h"
#include "plugin_cache.hpp"

using namespace InferenceEngine;

void setCommonConfig(std::map<std::string, std::string>& config) {
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_JSON)] = CONFIG_VALUE(NO);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_DOT)] = CONFIG_VALUE(NO);

    const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();

    config[VPU_KMB_CONFIG_KEY(MCM_COMPILATION_RESULTS_PATH)] = test_info->test_case_name();
    config[VPU_KMB_CONFIG_KEY(MCM_COMPILATION_RESULTS)] = test_info->name();
}

size_t precisionToBytesize(const std::string& precision) {
    size_t type_size = 1lu;
    if (precision == "FP32") {
        type_size = sizeof(float);
    } else if (precision == "FP16") {
        type_size = sizeof(ie_fp16);
    } else if (precision == "U8" || precision == "I8") {
        type_size = sizeof(uint8_t);
    } else if (precision == "I32" || precision == "U32") {
        type_size = sizeof(uint32_t);
    }
    return type_size;
}

void PrintTo(const tensor_test_params& sz, std::ostream* os) {
    *os << "{" << std::setw(2) << sz.n << ", " << std::setw(3) << sz.c << ", " << std::setw(3) << sz.h << ", "
        << std::setw(3) << sz.w << "}";
}

void print_buffer_HWC_fp16(ie_fp16* src_data, int32_t IW, int32_t IH, int32_t IC, const char* tname, int32_t iw0,
    int32_t iw1, int32_t ih0, int32_t ih1, int32_t ic0, int32_t ic1) {
    iw1 = (iw1 == -1) ? IW - 1 : iw1;
    ih1 = (ih1 == -1) ? IH - 1 : ih1;
    ic1 = (ic1 == -1) ? IC - 1 : ic1;

    printf("%s: H=%i, W=%i, C=%i\n", tname, IH, IW, IC);
    for (int ih = ih0; ih <= ih1; ih++) {
        printf("h %i: ", ih);
        for (int iw = iw0; iw <= iw1; iw++) {
            printf("(");
            for (int ic = ic0; ic <= ic1; ic++) {
                printf("%8.4f ", PrecisionUtils::f16tof32(src_data[ic + iw * IC + ih * IC * IW]));
            }
            printf("), ");
        }
        printf("\n");
    }
}

void vpuLayersTests::SetUp() {
    pluginName = "myriadPlugin";

    // TODO: we need another way to enable per-layer tests on HDDL
    // but for now it is ok because it is the fastest way
#ifdef USE_HDDL
    if (auto envVar = std::getenv("IE_VPU_ENABLE_PER_LAYER_TESTS_HDDL")) {
        if (std::stoi(envVar) != 0) pluginName = "HDDLPlugin";
    }
#endif
#ifdef USE_KMB
    pluginName = "kmbPlugin";
#endif

    _netInitialized = false;
    _genDataCallback = GenRandomData;
    TestsCommon::SetUp();
    SetSeed(DEFAULT_SEED_VALUE);
}

void vpuLayersTests::TearDown() {
    if (auto test_info = testing::UnitTest::GetInstance()->current_test_info()) {
        if (auto type_param = test_info->type_param()) {
            std::cout << "[ TYPE     ] \t" << type_param << std::endl;
        }
        if (auto value_param = test_info->value_param()) {
            std::cout << "[ VALUE    ] \t" << value_param << std::endl;
        }

        if (auto dumpModelsPath = std::getenv("IE_VPU_DUMP_LAYER_TESTS_MODELS_DIRECTORY")) {
            std::string testName = test_info->name();
            std::replace(testName.begin(), testName.end(), '/', '_');

            auto filename = dumpModelsPath + std::string("/") + testName;

            std::string xmlName = filename + ".xml";
            std::string weightsName = filename + ".bin";
            IE_SUPPRESS_DEPRECATED_START
            _net_reader.getNetwork().serialize(xmlName, weightsName);
            IE_SUPPRESS_DEPRECATED_END

            std::string blobName = filename + ".blob";
            _exeNetwork.Export(blobName);
        }
    }
}

bool vpuLayersTests::CheckMyriadX() {
    if (auto envVar = std::getenv("IE_VPU_MYRIADX")) {
        return std::stoi(envVar) != 0;
    }
    return false;
}

void vpuLayersTests::SetSeed(uint32_t seed) {
    /*just to be able to repeat results */
    std::srand(seed);
}

void vpuLayersTests::dumpPerformance() {
    std::map<std::string, InferenceEngineProfileInfo> perfMap;
    _inferRequest->GetPerformanceCounts(perfMap, nullptr);
    std::vector<std::pair<std::string, InferenceEngineProfileInfo>> perfVec(perfMap.begin(), perfMap.end());
    std::sort(perfVec.begin(), perfVec.end(),
        [=](const std::pair<std::string, InferenceEngineProfileInfo>& pair1,
            const std::pair<std::string, InferenceEngineProfileInfo>& pair2) -> bool {
            return pair1.second.execution_index < pair2.second.execution_index;
        });

    for (auto it = perfVec.begin(); it != perfVec.end(); ++it) {
        std::string layerName = it->first;
        InferenceEngineProfileInfo info = it->second;
        if (info.status == InferenceEngineProfileInfo::EXECUTED) {
            printf("\x1B[32m[----------]\x1B[0m Myriad time = '%s' layer with '%s' type is %f ms.\n", layerName.c_str(),
                info.exec_type, info.realTime_uSec / 1000.f);
        }
    }
}

std::string vpuLayersTests::genPorts(const IN_OUT_desc& tensors, size_t* inoutIndex, const std::string& bracketName) {
    static const std::string portStr = R"V0G0N(                 <port id="_IDX_">
)V0G0N";
    std::string outLine;
    outLine = "            <BRACKET>\n";
    outLine.replace(outLine.begin() + 13, outLine.begin() + 20, bracketName);
    for (auto iter : tensors) {
        std::string temp = portStr;
        REPLACE_WITH_NUM(temp, "_IDX_", *inoutIndex);
        (*inoutIndex)++;
        outLine += temp;
        const size_t* ptr = iter.data();
        for (size_t indx = 0; indx < iter.size(); ++indx) {
            temp = "                     <dim>_IN_</dim>\n";
            REPLACE_WITH_NUM(temp, "_IN_", ptr[indx]);
            outLine += temp;
        }
        outLine += "                 </port>\n";
    }
    std::string tempLine = "            </BRACKET>\n";
    tempLine.replace(tempLine.begin() + 14, tempLine.begin() + 21, bracketName);
    outLine += tempLine;
    return outLine;
}

void vpuLayersTests::genLayer(std::string layer_type, std::map<std::string, std::string>* params, size_t* inoutIndex,
    std::string& out, IN_OUT_desc& inpTensors, IN_OUT_desc& outTensors, std::string* newName) {
    std::string layerHead = R"V0G0N(
        <layer name="_LAYER_NAME_" type="_LAYER_TYPE_" precision="FP16" id="_ID_">
)V0G0N";
    if (newName == nullptr) {
        _layerName = layer_type + "_TEST";
    } else {
        _layerName = (*newName);
    }
    REPLACE_WITH_NUM(layerHead, "_ID_", *inoutIndex);
    REPLACE_WITH_STR(layerHead, "_LAYER_NAME_", _layerName);
    REPLACE_WITH_STR(layerHead, "_LAYER_TYPE_", layer_type);
    out += layerHead;

    if (params != nullptr) {
        std::string layerParams = "";
        if (params->size() == 1 && params->begin()->first == "<crop-data>") {
            /* Crop layer parser does not allow to use common way for XML generation */
            layerParams = "            <crop-data>" + params->begin()->second;
            layerParams += "           </crop-data>\n";
        } else {
            layerParams = "            <data ";
            for (auto& kv : *params) {
                layerParams += kv.first + "=" + "\"" + kv.second + "\" ";
            }
            layerParams += "/> \n";
        }
        out += layerParams;
    }

    out += genPorts(inpTensors, inoutIndex, "input");
    out += genPorts(outTensors, inoutIndex, "output");
}

void vpuLayersTests::genWeights(int weights_size, int biases_size, size_t* inoutIndex, std::string& out) {
    std::string result;
    std::string weights_str = std::to_string(weights_size);
    if (weights_size != 0) {
        result = "            <weights offset=";
        result += "\"0\" size=\"" + weights_str + "\"/>\n";
    }
    if (biases_size != 0) {
        std::string biases_str = std::to_string(biases_size);
        result += "            <biases offset=\"" + weights_str + "\" size=\"" + biases_str + "\"/>\n";
    }
    out += result;
    out += "        </layer>";
}

void vpuLayersTests::genWeights(
    int weights_size, int biases_size, int weights_offset, int biases_offset, size_t* inoutIndex, std::string& out) {
    std::string result;
    if (weights_size != 0) {
        std::string weights_str = std::to_string(weights_size);
        std::string weights_off = std::to_string(weights_offset);
        result = "            <weights offset=\"" + weights_off;
        result += "\" size=\"" + weights_str + "\"/>\n";
    }
    if (biases_size != 0) {
        std::string biases_str = std::to_string(biases_size);
        std::string biases_off = std::to_string(biases_offset);
        result += "            <biases offset=\"";
        result += biases_off + "\" size=\"" + biases_str + "\"/>\n";
    }
    out += result;
    out += "        </layer>";
}

void vpuLayersTests::genXML(const std::string& layer_type, std::map<std::string, std::string>* params, int weights_size,
    int biases_size, std::string& model) {
    model = R"V0G0N(
<net name="_CURRENT_TEST_" version="2" batch="1">
    <layers>
)V0G0N";
    ASSERT_EQ(_inputTensors.empty(), false) << "Inputs are not defined.";
    ASSERT_EQ(_outputTensors.empty(), false) << "Outputs are not defined.";
    size_t inoutIndex = 0;
    /* input layer generation */
    std::string testName = layer_type + "_test";
    std::transform(testName.begin(), testName.end(), testName.begin(), [](unsigned char c) -> unsigned char {
        return std::toupper(c);
    });
    REPLACE_WITH_STR(model, "_CURRENT_TEST_", testName);
    IN_OUT_desc inputToReshape = _inputTensors;
    auto doReshape = _doReshape && _inputTensors.size() == 1 && _inputTensors[0].size() < 4;
    if (doReshape) {
        for (auto j = _inputTensors[0].size(); j != 4; j++) {
            inputToReshape[0].insert(inputToReshape[0].begin(), 1);
        }
    }
    std::swap(inputToReshape, _inputTensors);

    if (_inputTensors.size() == 1) {
        model += R"V0G0N(
        <layer name="input" type="Input" precision="FP16" id="0">
        )V0G0N";
        model += genPorts(_inputTensors, &inoutIndex, "output");
        model += "        </layer>";
    } else {
        IN_OUT_desc tensor(1);
        for (size_t indx = 0; indx < _inputTensors.size(); ++indx) {
            std::string temp = R"V0G0N(
        <layer name="input_INDX_" type="Input" precision="FP16" id="_INDX_">
        )V0G0N";
            REPLACE_WITH_NUM(temp, "_INDX_", indx);
            model += temp;
            tensor[0] = _inputTensors[indx];
            model += genPorts(tensor, &inoutIndex, "output");
            model += "        </layer>";
        }
    }
    std::swap(inputToReshape, _inputTensors);

    if (doReshape) {
        genLayer("reshape", params, &inoutIndex, model, inputToReshape, _inputTensors);
        model += "        </layer>";
    }
    genLayer(layer_type, params, &inoutIndex, model, _inputTensors, _outputTensors);
    genWeights(weights_size, biases_size, &inoutIndex, model);
    model += R"V0G0N(
    </layers>
    <edges>
)V0G0N";
    size_t indx = 0;
    std::swap(inputToReshape, _inputTensors);
    for (; indx < _inputTensors.size(); ++indx) {
        std::string val =
            R"V0G0N(       <edge from-layer="_INDX_OUT_" from-port="_INDX_OUT_" to-layer="_LAYER_" to-port="_INDX_IN_"/>
)V0G0N";
        REPLACE_WITH_NUM(val, "_INDX_OUT_", indx);
        REPLACE_WITH_NUM(val, "_LAYER_", _inputTensors.size());
        REPLACE_WITH_NUM(val, "_INDX_IN_", indx + _inputTensors.size());
        model += val;
    }

    std::swap(inputToReshape, _inputTensors);
    if (doReshape) {
        for (; indx < inputToReshape.size() + _inputTensors.size(); ++indx) {
            std::string val =
                R"V0G0N(       <edge from-layer="_INDX_OUT_" from-port="_INDX_PORT_OUT_" to-layer="_LAYER_" to-port="_INDX_IN_"/>
)V0G0N";

            REPLACE_WITH_NUM(val, "_INDX_OUT_", indx);
            REPLACE_WITH_NUM(val, "_INDX_PORT_OUT_", indx + _inputTensors.size());
            REPLACE_WITH_NUM(val, "_LAYER_", indx + inputToReshape.size() + _inputTensors.size());
            REPLACE_WITH_NUM(val, "_INDX_IN_", indx + inputToReshape.size() + _inputTensors.size());
            model += val;
        }
    }
    model += "    </edges>\n</net>\n";
}

void vpuLayersTests::genInputBlobs(Precision precision) {
    auto genDataCallback = (_genDataCallback0 != nullptr) ? _genDataCallback0 : _genDataCallback;
    for (auto inpt : _inputsInfo) {
        SizeVector inputDims = inpt.second->getTensorDesc().getDims();
        Blob::Ptr inputBlob = nullptr;
        Layout netLayout = inpt.second->getTensorDesc().getLayout();
        // work only with NHWC layout if size of the input dimensions == NHWC
        Layout layout = netLayout == NHWC || netLayout == NCHW ? NHWC : netLayout;
        switch (precision) {
        case Precision::U8:
            inputBlob = make_shared_blob<uint8_t>({Precision::U8, inputDims, layout});
            break;
        case Precision::FP16:
            inputBlob = make_shared_blob<ie_fp16>({Precision::FP16, inputDims, layout});
            break;
        case Precision::FP32:
            inputBlob = make_shared_blob<float>({Precision::FP32, inputDims, layout});
            break;
        default:
            THROW_IE_EXCEPTION << "Unsupported precision for input. Supported U8, FP16, FP32";
        }
        inputBlob->allocate();
        ASSERT_NE(genDataCallback, nullptr);
        genDataCallback(inputBlob);
        StatusCode st = _inferRequest->SetBlob(inpt.first.c_str(), inputBlob, &_resp);
        ASSERT_EQ((int)StatusCode::OK, st) << _resp.msg;
        _inputMap[inpt.first] = inputBlob;
        genDataCallback = _genDataCallback;
    }
}

void GenRandomData(Blob::Ptr blob) { GenRandomDataCommon(blob); }

void vpuLayersTests::genOutputBlobs(Precision precision) {
    for (auto outpt : _outputsInfo) {
        SizeVector outputDims = outpt.second->getTensorDesc().getDims();
        Blob::Ptr outputBlob = nullptr;
        Layout netLayout = outpt.second->getTensorDesc().getLayout();
        // work only with NHWC layout if size of the input dimensions == NHWC
        Layout layout = netLayout == NHWC || netLayout == NCHW ? NHWC : netLayout;
        switch (precision) {
        case Precision::FP16:
            outputBlob = make_shared_blob<ie_fp16>({Precision::FP16, outputDims, layout});
            break;
        case Precision::FP32:
            outputBlob = make_shared_blob<float>({Precision::FP32, outputDims, layout});
            break;
        default:
            THROW_IE_EXCEPTION << "Unsupported precision for output. Supported FP16, FP32";
        }
        outputBlob->allocate();
        StatusCode st = _inferRequest->SetBlob(outpt.first.c_str(), outputBlob, &_resp);
        _outputMap[outpt.first] = outputBlob;
        ASSERT_EQ((int)StatusCode::OK, st) << _resp.msg;
    }
}

void vpuLayersTests::setup(Precision outputPrecision, Precision inputPrecision, bool useHWOpt) {
    IE_SUPPRESS_DEPRECATED_START
    CNNNetwork network = _net_reader.getNetwork();
    IE_SUPPRESS_DEPRECATED_END
    _inputsInfo = network.getInputsInfo();
    for (const auto& in : _inputsInfo) {
        in.second->setPrecision(inputPrecision);
    }
    _outputsInfo = network.getOutputsInfo();
    for (const auto& outputInfo : _outputsInfo) {
        outputInfo.second->setPrecision(outputPrecision);
    }
    std::map<std::string, std::string> config(_config);
    if (useHWOpt) {
        config[VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION)] = CONFIG_VALUE(YES);
    } else {
        config[VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION)] = CONFIG_VALUE(NO);
    }
#if 0
    config[CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_INFO);
#endif
    config[CONFIG_KEY(PERF_COUNT)] = CONFIG_VALUE(YES);
    config[VPU_CONFIG_KEY(PERF_REPORT_MODE)] = VPU_CONFIG_VALUE(PER_STAGE);

    _exeNetwork = ie.LoadNetwork(network, "kmb", config);
    _inputsInfo = network.getInputsInfo();
    _outputsInfo = network.getOutputsInfo();
    genInputBlobs(inputPrecision);
    genOutputBlobs(outputPrecision);
    // FIXME: why we create and allocate refBlob here for myriadLayerTests if outputPrecision == FP16?
    if (outputPrecision == Precision::FP16) {
        Layout netLayout = _outputsInfo.begin()->second->getTensorDesc().getLayout();
        // work only with NHWC layout if size of the input dimensions == NHWC
        Layout layout = netLayout == NHWC || netLayout == NCHW ? NHWC : netLayout;
        _refBlob =
            make_shared_blob<ie_fp16>({Precision::FP16, _outputMap.begin()->second->getTensorDesc().getDims(), layout});
        _refBlob->allocate();
    }
}

void vpuLayersTests::doNetworkInit(const std::string& layer_type, std::map<std::string, std::string>* params,
    int weights_size, int biases_size, TBlob<uint8_t>::Ptr weights, Precision outputPrecision, Precision inputPrecision,
    bool useHWOpt) {
    std::string xml;
    genXML(layer_type, params, weights_size, biases_size, xml);
    IE_SUPPRESS_DEPRECATED_START
    ASSERT_NO_THROW(_net_reader.ReadNetwork(xml.data(), xml.length()));
    ASSERT_EQ(_net_reader.isParseSuccess(), true);
    if (weights != nullptr) {
        ASSERT_NO_THROW(_net_reader.SetWeights(weights));
    }
    IE_SUPPRESS_DEPRECATED_END
    setup(outputPrecision, inputPrecision, useHWOpt);
}

bool vpuLayersTests::Infer() {
    bool res = false;

    if (_inferRequest == nullptr || _inputMap.size() == 0 || _outputMap.size() == 0) return res;
    StatusCode st = GENERAL_ERROR;
    st = _inferRequest->Infer(&_resp);
    if (OK == st)
        res = true;
    else {
        EXPECT_EQ((int)StatusCode::OK, st) << _resp.msg;
    }
    // dumpPerformance();
    return res;
}

void vpuLayersTests::SetInputTensor(tensor_test_params const& tensor) {
    /* setup one input */
    _inputTensors.resize(1);
    _inputTensors[0].resize(4);
    _inputTensors[0][0] = tensor.n;
    _inputTensors[0][1] = tensor.c;
    _inputTensors[0][2] = tensor.h;
    _inputTensors[0][3] = tensor.w;
}

void vpuLayersTests::SetInputTensors(IN_OUT_desc in_tensors) { _inputTensors = in_tensors; }

void vpuLayersTests::SetOutputTensor(tensor_test_params const& tensor) {
    /* setup  one output*/
    _outputTensors.resize(1);
    _outputTensors[0].resize(4);
    _outputTensors[0][0] = tensor.n;
    _outputTensors[0][1] = tensor.c;
    _outputTensors[0][2] = tensor.h;
    _outputTensors[0][3] = tensor.w;
}

void vpuLayersTests::SetOutputTensors(IN_OUT_desc out_tensors) { _outputTensors = out_tensors; }

void vpuLayersTests::SetFirstInputToRange(float start, float finish) {
    ASSERT_NE(_inputMap.size(), 0u);
    ASSERT_LT(start, finish);
    float range = finish - start;
    /* input data preparation */
    auto inputBlob = _inputMap[_inputsInfo.begin()->first];
    uint16_t* inputBlobRawDataFp16 = inputBlob->buffer().as<uint16_t*>();
    ASSERT_NE(inputBlobRawDataFp16, nullptr);
    /* values generation in the range (start, finish) to check difference with float output */
    size_t count = inputBlob->size();
    float shift = range / count;
    float i = start;
    for (size_t indx = 0; indx < count; i += shift, indx++) {
        inputBlobRawDataFp16[indx] = PrecisionUtils::f32tof16(i);
    }
}

void vpuLayersTests::SetInputInOrder() {
    ASSERT_NE(_inputsInfo.size(), 0u);
    SizeVector inputDims = _inputsInfo.begin()->second->getTensorDesc().getDims();
    auto inputBlob = _inputMap[_inputsInfo.begin()->first];
    ASSERT_NE(inputBlob, nullptr);
    uint16_t* inputBlobRawDataFp16 = inputBlob->buffer().as<uint16_t*>();
    ASSERT_NE(inputBlobRawDataFp16, nullptr);
    /* values generation in the range (-BOUND, BOUND) to check difference with float output */
    int count = inputBlob->size();

    for (int indx = 0; indx < count; indx++) {
        inputBlobRawDataFp16[indx] = PrecisionUtils::f32tof16((float)indx);
    }
}

void vpuLayersTests::SetInputInOrderReverse() {
    ASSERT_NE(_inputsInfo.size(), 0u);
    auto inputBlob = _inputMap[_inputsInfo.begin()->first];
    ASSERT_NE(inputBlob, nullptr);
    uint16_t* dstPtr = inputBlob->buffer().as<uint16_t*>();
    ASSERT_NE(dstPtr, nullptr);
    size_t count = inputBlob->size();
    for (size_t indx = 0; indx < count; indx++) {
        dstPtr[indx] = PrecisionUtils::f32tof16((float)(count - 1 - indx));
    }
}

void vpuLayersTests::checkBlobs(Blob::Ptr actual, Blob::Ptr expected) {
    const auto& actual_dims = actual->getTensorDesc().getDims();
    const auto& expected_dims = expected->getTensorDesc().getDims();

    ASSERT_NE(actual_dims.size(), 0u);
    ASSERT_EQ(actual_dims.size(), expected_dims.size());
    for (size_t i = 0; i < actual_dims.size(); i++) {
        ASSERT_EQ(actual_dims[i], expected_dims[i]);
    }
    float* actualData = dynamic_cast<TBlob<float>*>(&(*actual))->data();
    ASSERT_NE(actualData, nullptr);
    float* expectedData = dynamic_cast<TBlob<float>*>(&(*expected))->data();
    ASSERT_NE(expectedData, nullptr);
    size_t countElems = actual_dims[0];
    for (size_t i = 1; i < actual_dims.size(); i++) {
        countElems *= actual_dims[i];
    }
    for (size_t i = 0; i < countElems; i++) {
        ASSERT_FLOAT_EQ(actualData[i], expectedData[i]);
    }
}

void Compare(Blob::Ptr actual, Blob::Ptr expected, float tolerance) {
    // do not compare anything if have fake device
    if (hasFakeXLinkDevice()) return;

    CompareCommonAbsolute(actual, expected, tolerance);
}

void vpuLayersTests::CompareWithNorm(Blob::Ptr actual, Blob::Ptr expected, float max_diff) {
    ASSERT_NE(actual, nullptr);
    ASSERT_NE(expected, nullptr);
    const uint16_t* res_ptr = actual->buffer().as<const uint16_t*>();
    size_t res_size = actual->size();

    const uint16_t* ref_ptr = expected->buffer().as<const uint16_t*>();
    size_t ref_size = expected->size();

    ASSERT_EQ(res_size, ref_size);

    for (size_t i = 0; i < ref_size; i++) {
        float val_res = PrecisionUtils::f16tof32(res_ptr[i]);
        float val_ref = PrecisionUtils::f16tof32(ref_ptr[i]);
        float norm = std::max(fabs(val_res), fabs(val_ref));
        if (norm < 1.0f) norm = 1.0f;
        ASSERT_NEAR(val_res, val_ref, (max_diff * norm));
    }
}

void vpuLayersTests::genNetwork(bool useHWOpt, int version) {
    ASSERT_TRUE(!_testNet.empty());
    int32_t counter = 0;
    size_t real_offset = 0;
    /* weights & biases calculation */
    for (auto& elem : _testNet) {
        ASSERT_EQ(elem.inDim.size(), 1u);
        if (elem.fillWeights) {
            elem.weights_offset = real_offset;
            real_offset += elem.weights_size;
            elem.biases_offset = real_offset;
            real_offset += elem.biases_size;
        }
    }
    TBlob<uint8_t>::Ptr weights_ptr;
    if (real_offset) {
        TBlob<uint8_t>::Ptr weights = make_shared_blob<uint8_t>({Precision::U8, {(real_offset) * sizeof(uint16_t)}, C});
        ASSERT_NE(weights, nullptr);
        weights->allocate();
        weights_ptr = weights;
        /* fillup data */
        uint16_t* inputBlobRawDataFp16 = weights_ptr->buffer().as<uint16_t*>();
        ASSERT_NE(inputBlobRawDataFp16, nullptr);
        uint16_t* ptr = inputBlobRawDataFp16;
        for (auto& elem : _testNet) {
            if (elem.fillWeights) {
                elem.fillWeights(ptr, elem.weights_size, elem.biases_size);
                if (elem.biases_size) {
                    ptr += elem.biases_offset + elem.biases_size;
                } else {
                    if (elem.weights_size) {
                        ptr += elem.weights_offset + elem.weights_size;
                    }
                }
            }
        }
    }
    counter = 0;
    std::string input_dim_prefix =
        R"V0G0N(
        <layer id="0" name="input" precision="FP16" type="input">
            <output>
                <port id="1">
)V0G0N";
    std::string input_dim_postfix =
        R"V0G0N(
                </port>
            </output>
        </layer>)V0G0N";

    auto bgn = _testNet.begin();
    std::string& strt = input_dim_prefix;
    for (size_t iter = 0; iter < bgn->inDim[0].size(); iter++) {
        strt += std::string("<dim>_DIM_") + std::to_string(iter) + std::string("_</dim>\n");
    }

    strt += input_dim_postfix;

    for (auto val : bgn->inDim[0]) {
        std::string ind = "_DIM_" + std::to_string(counter) + "_";
        REPLACE_WITH_NUM(strt, ind.c_str(), val);
        ++counter;
    }
    counter = 1;
    size_t inoutIndex = 2; /* to align out port index with input layer index */
    std::string layers;
    std::string edges = R"V0G0N(
    <edges>
)V0G0N";
    size_t outIdx = 0;
    for (auto& elem : _testNet) {
        std::string layer;
        elem.layer_name = "layer_" + std::to_string(counter);
        if (elem.params.empty()) {
            genLayer(elem.layer_type, nullptr, &inoutIndex, layers, elem.inDim, elem.outDim, &elem.layer_name);
        } else {
            genLayer(elem.layer_type, &elem.params, &inoutIndex, layers, elem.inDim, elem.outDim, &elem.layer_name);
        }
        genWeights(elem.weights_size * sizeof(uint16_t), elem.biases_size * sizeof(uint16_t),
            elem.weights_offset * sizeof(uint16_t), elem.biases_offset * sizeof(uint16_t), &inoutIndex, layers);

        std::string val =
            R"V0G0N(       <edge from-layer="_INDX_OUT_" from-port="_PORT_OUT_" to-layer="_LAYER_" to-port="_INDX_IN_"/>
 )V0G0N";
        REPLACE_WITH_NUM(val, "_INDX_OUT_", outIdx);
        REPLACE_WITH_NUM(val, "_PORT_OUT_", outIdx + 1);
        REPLACE_WITH_NUM(val, "_LAYER_", inoutIndex - 2);
        REPLACE_WITH_NUM(val, "_INDX_IN_", inoutIndex - 2);
        outIdx += 2;
        edges += val;
        ++counter;
    }
    edges += "    </edges>\n</net>\n";
    std::string model;
    if (version == 2)
        model = R"V0G0N(
<net name="_CURRENT_TEST_" version="2" batch="1">
    <layers>)V0G0N";
    else
        model = R"V0G0N(
<net name="_CURRENT_TEST_" version="3" batch="1">
    <layers>)V0G0N";
    model += strt;
    model += layers;
    model += R"V0G0N(
    </layers>)V0G0N";
    model += edges;
    IE_SUPPRESS_DEPRECATED_START
    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());
    if (weights_ptr != nullptr) {
        _net_reader.SetWeights(weights_ptr);
    }
    IE_SUPPRESS_DEPRECATED_END
    setup(Precision::FP16, Precision::FP16, useHWOpt);
    _netInitialized = true;
}

void vpuLayersTests::ReferenceGraph() {
    /* data preparation */
    ASSERT_TRUE(!_testNet.empty());
    ASSERT_TRUE(!_referenceGraph.callbacks.empty());
    ASSERT_EQ(_inputsInfo.size(), 1u);
    ASSERT_EQ(_testNet.size(), _referenceGraph.callbacks.size());
    auto referenceInput = _referenceGraph.callbacks.begin()->input;
    auto realInput = _inputMap[_inputsInfo.begin()->first];
    ASSERT_NE(referenceInput, nullptr);
    ASSERT_NE(realInput, nullptr);
    size_t count = referenceInput->size();
    ASSERT_EQ(count, realInput->size());
    uint16_t* inputBlobRawDataFp16 = realInput->buffer();
    uint16_t* refBlobRawDataFp16 = referenceInput->buffer();
    ASSERT_NE(inputBlobRawDataFp16, nullptr);
    ie_memcpy(refBlobRawDataFp16, realInput->byteSize(), inputBlobRawDataFp16, count * sizeof(uint16_t));
    IE_SUPPRESS_DEPRECATED_START
    ICNNNetwork& network = _net_reader.getNetwork();
    IE_SUPPRESS_DEPRECATED_END
    for (size_t ind = 0; ind < _testNet.size(); ++ind) {
        if (_testNet[ind].fillWeights != nullptr) {
            auto& refLayer = _referenceGraph.callbacks[ind];
            CNNLayerPtr layer;
            auto status = network.getLayerByName(_testNet[ind].layer_name.c_str(), layer, &_resp);
            ASSERT_EQ(status, (int)StatusCode::OK);
            if (layer->type == "PReLU") {
                /* PReLU is a custom layer */
                auto it = layer->blobs.find("weights");
                if (it == layer->blobs.end()) {
                    FAIL() << "PReLU doesn't have weights";
                }
                auto weightsBlob = it->second;
                auto weigths_sz = weightsBlob->size();
                uint16_t* inputWeightsDataFp16 = weightsBlob->buffer();
                ie_memcpy(refLayer.weights, refLayer.weights_size * sizeof(uint16_t), inputWeightsDataFp16,
                    weigths_sz * sizeof(uint16_t));
            } else {
                auto layerWithWeights = std::dynamic_pointer_cast<WeightableLayer>(layer);
                size_t weigths_sz = 0;
                if (layerWithWeights->_weights) {
                    weigths_sz = layerWithWeights->_weights->size();
                    ASSERT_EQ(weigths_sz, refLayer.weights_size);
                    uint16_t* inputWeightsDataFp16 = layerWithWeights->_weights->buffer();
                    ie_memcpy(refLayer.weights, refLayer.weights_size * sizeof(uint16_t), inputWeightsDataFp16,
                        weigths_sz * sizeof(uint16_t));
                }
                size_t bias_sz = 0;
                if (layerWithWeights->_biases) {
                    bias_sz = layerWithWeights->_biases->size();
                    ASSERT_EQ(bias_sz, refLayer.bias_size);
                    uint16_t* refBiasDataFp16 = layerWithWeights->_biases->buffer();
                    ie_memcpy(&refLayer.weights[weigths_sz], refLayer.bias_size * sizeof(uint16_t), refBiasDataFp16,
                        bias_sz * sizeof(uint16_t));
                }
            }
        }
    }
    _referenceGraph();
}

template <typename T>
T vpuLayersTests::generate_val(float min_val, float max_val) {
    float scale = (max_val - min_val) / RAND_MAX;
    float val = rand() * scale + min_val;
    return static_cast<T>(val);
}

template <>
uint16_t vpuLayersTests::generate_val<uint16_t>(float min_val, float max_val) {
    float scale = (max_val - min_val) / RAND_MAX;
    float val = rand() * scale + min_val;
    return InferenceEngine::PrecisionUtils::f32tof16(val);
}

template <>
void fillRealBuffer<InferenceEngine::ie_fp16>(
    InferenceEngine::ie_fp16* data, size_t size, InferenceEngine::ie_fp16 min, InferenceEngine::ie_fp16 max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(
        InferenceEngine::PrecisionUtils::f16tof32(min), InferenceEngine::PrecisionUtils::f16tof32(max));

    fillCommon(data, size, [&]() {
        return InferenceEngine::PrecisionUtils::f32tof16(dis(gen));
    });
}

Blob::Ptr ConvertU8ToFP32(const Blob::Ptr& inBlob) {
    TensorDesc inTensor = inBlob->getTensorDesc();
    if (inTensor.getPrecision() != Precision::U8) {
        throw std::runtime_error("ConvertU8ToFP16: only U8 input is supported");
    }
    SizeVector outDims = inTensor.getDims();
    Layout outLayout = inTensor.getLayout();
    Precision outPrecision = Precision::FP32;
    TensorDesc outTensor(outPrecision, outDims, outLayout);
    Blob::Ptr outBlob = make_blob_with_precision(outTensor);
    outBlob->allocate();
    uint8_t* blobRawDataU8 = inBlob->buffer().as<uint8_t*>();
    float* blobRawDataFP32 = outBlob->buffer().as<float*>();
    for (size_t pos = 0; pos < inBlob->size(); pos++) {
        blobRawDataFP32[pos] = blobRawDataU8[pos];
    }

    return outBlob;
}

void compareTopClasses(
    const InferenceEngine::Blob::Ptr& resultBlob, const InferenceEngine::Blob::Ptr& refBlob, size_t maxClasses) {
    // do not compare anything if have fake device
    if (hasFakeXLinkDevice()) return;

    std::vector<size_t> outTopClasses, refTopClasses;
    switch (resultBlob->getTensorDesc().getPrecision()) {
    case Precision::U8:
        outTopClasses = yieldTopClasses<uint8_t>(resultBlob, maxClasses);
        refTopClasses = yieldTopClasses<uint8_t>(refBlob, maxClasses);
        break;
    case Precision::FP32:
        outTopClasses = yieldTopClasses<float>(resultBlob, maxClasses);
        refTopClasses = yieldTopClasses<float>(refBlob, maxClasses);
        break;
    default:
        throw std::runtime_error("compareTopClasses: only U8 and FP32 are supported");
    }
    std::ostringstream logStream;
    logStream << "out: ";
    for (size_t classId = 0; classId < maxClasses; classId++) {
        logStream << outTopClasses[classId] << " ";
    }

    logStream << std::endl << "ref: ";
    for (size_t classId = 0; classId < maxClasses; classId++) {
        logStream << refTopClasses[classId] << " ";
    }
    EXPECT_TRUE(std::equal(outTopClasses.begin(), outTopClasses.end(), refTopClasses.begin())) << logStream.str();
}
