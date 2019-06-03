// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"

using namespace InferenceEngine;
using namespace ::testing;

typedef myriadLayerTestBaseWithParam<std::string> myriadBlobTests_nightly;

std::vector<char> readBinFile(std::string filename)
{
    std::ifstream file(filename, std::ios::binary);
    file.unsetf(std::ios::skipws);

    std::streampos fileSize;

    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> vec;
    vec.reserve(fileSize);

    vec.insert(vec.begin(),
        std::istream_iterator<char>(file),
        std::istream_iterator<char>());

    return vec;
}

TEST_P(myriadBlobTests_nightly, CanGetSameBlobsOnSameIR) {
    std::string HWConfigValue = GetParam();

    std::string modelName = "googlenet/bvlc_googlenet_fp16";
    std::ostringstream modelFile;
    modelFile << "/" << modelName << ".xml";

    std::ostringstream weightsFile;
    weightsFile << "/" << modelName << ".bin";

    std::string modelFilePath = ModelsPath() + modelFile.str();
    std::string weightsFilePath = ModelsPath() + weightsFile.str();

    ASSERT_NO_THROW(_net_reader.ReadNetwork(modelFilePath));
    ASSERT_TRUE(_net_reader.isParseSuccess());
    ASSERT_NO_THROW(_net_reader.ReadWeights(weightsFilePath));

    const size_t countBlobsToDump = 3;
    std::vector<std::string> filenames(countBlobsToDump);
    for (int i = 0; i < countBlobsToDump; i++) {

        StatusCode st;
        ASSERT_NO_THROW(st = myriadPluginPtr->LoadNetwork(_exeNetwork, _net_reader.getNetwork(),
        { {VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), HWConfigValue } }, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
        std::stringstream modelFilenameStream;
        modelFilenameStream << "googlenet_" << i << ".blob";
        filenames[i] = modelFilenameStream.str();
        ASSERT_NO_THROW(_exeNetwork->Export(modelFilenameStream.str(), nullptr));
    }

    for (int i = 0; i < filenames.size() - 1; i++) {
        std::vector<char>  blob1 = readBinFile(filenames[i]);
        std::vector<char>  blob2 = readBinFile(filenames[i + 1]);
        ASSERT_TRUE(blob1 == blob2);
    }

    for (int i = 0; i < filenames.size(); i++) {
        std::remove(filenames[i].c_str());
    }
}

INSTANTIATE_TEST_CASE_P(accuracy, myriadBlobTests_nightly,
    ::testing::Values(CONFIG_VALUE(YES), CONFIG_VALUE(NO))
);

using myriadBlobExportTests_nightly = myriadLayersTests_nightly;


TEST_F(myriadBlobExportTests_nightly, CanNotDoImportOnNonExistFile)
{
    InferenceEngine::IExecutableNetwork::Ptr importedNetworkPtr;
    ASSERT_EQ(StatusCode::NETWORK_NOT_READ, myriadPluginPtr->ImportNetwork(importedNetworkPtr, "I_dont_exist.blob", {}, nullptr));
}

TEST_F(myriadBlobExportTests_nightly, CanInferImportedNetworkOnExportedBlob)
{
    std::string modelName = "googlenet/bvlc_googlenet_fp16";
    std::ostringstream modelFile;
    modelFile << "/" << modelName << ".xml";

    std::ostringstream weightsFile;
    weightsFile << "/" << modelName << ".bin";

    std::string modelFilePath = ModelsPath() + modelFile.str();
    std::string weightsFilePath = ModelsPath() + weightsFile.str();

    ASSERT_NO_THROW(_net_reader.ReadNetwork(modelFilePath));
    ASSERT_TRUE(_net_reader.isParseSuccess());
    ASSERT_NO_THROW(_net_reader.ReadWeights(weightsFilePath));

    ASSERT_EQ(StatusCode::OK, myriadPluginPtr->LoadNetwork(_exeNetwork, _net_reader.getNetwork(), { }, &_resp)) << _resp.msg;
    std::stringstream modelFilenameStream;
    modelFilenameStream << "googlenet" << ".blob";
    ASSERT_EQ(StatusCode::OK, _exeNetwork->Export(modelFilenameStream.str(), &_resp)) << _resp.msg;

    InferenceEngine::IExecutableNetwork::Ptr importedNetworkPtr;
    ASSERT_EQ(StatusCode::OK, myriadPluginPtr->ImportNetwork(importedNetworkPtr, modelFilenameStream.str(), {}, &_resp)) << _resp.msg;
    InferenceEngine::IInferRequest::Ptr inferRequest;
    ASSERT_EQ(StatusCode::OK, importedNetworkPtr->CreateInferRequest(inferRequest, &_resp)) << _resp.msg;

    ASSERT_EQ(StatusCode::OK, inferRequest->Infer(&_resp)) << _resp.msg;
}

TEST_F(myriadBlobExportTests_nightly, CanGetPerfCountsImportedNetwork)
{
    std::string modelName = "googlenet/bvlc_googlenet_fp16";
    std::ostringstream modelFile;
    modelFile << "/" << modelName << ".xml";

    std::ostringstream weightsFile;
    weightsFile << "/" << modelName << ".bin";

    std::string modelFilePath = ModelsPath() + modelFile.str();
    std::string weightsFilePath = ModelsPath() + weightsFile.str();

    ASSERT_NO_THROW(_net_reader.ReadNetwork(modelFilePath));
    ASSERT_TRUE(_net_reader.isParseSuccess());
    ASSERT_NO_THROW(_net_reader.ReadWeights(weightsFilePath));

    ASSERT_EQ(StatusCode::OK, myriadPluginPtr->LoadNetwork(_exeNetwork, _net_reader.getNetwork(), {}, &_resp)) << _resp.msg;
    std::stringstream modelFilenameStream;
    modelFilenameStream << "googlenet" << ".blob";
    ASSERT_EQ(StatusCode::OK, _exeNetwork->Export(modelFilenameStream.str(), &_resp)) << _resp.msg;

    std::map<std::string, std::string> config = { {CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES)} };
    InferenceEngine::IExecutableNetwork::Ptr importedNetworkPtr;
    ASSERT_EQ(StatusCode::OK, myriadPluginPtr->ImportNetwork(importedNetworkPtr, modelFilenameStream.str(), config, &_resp)) << _resp.msg;
    InferenceEngine::IInferRequest::Ptr inferRequest;
    ASSERT_EQ(StatusCode::OK, importedNetworkPtr->CreateInferRequest(inferRequest, &_resp)) << _resp.msg;

    ASSERT_EQ(StatusCode::OK, inferRequest->Infer(&_resp)) << _resp.msg;
    std::map<std::string, InferenceEngineProfileInfo> perfCounts;
    ASSERT_EQ(StatusCode::OK, inferRequest->GetPerformanceCounts(perfCounts, &_resp)) << _resp.msg;

    ASSERT_NE(0, perfCounts.size());
    for (const auto &perfInfoElem : perfCounts) {
        InferenceEngineProfileInfo perfInfo = perfInfoElem.second;
        ASSERT_EQ(perfInfo.status, InferenceEngineProfileInfo::LayerStatus::EXECUTED);
        ASSERT_STREQ(perfInfo.exec_type, "UNKNOWN");
        ASSERT_STREQ(perfInfo.layer_type, "UNKNOWN");
        ASSERT_NE(perfInfo.realTime_uSec, 0);
    }
}

class myriadConfigsWithBlobImportTests_nightly: public myriadLayersTests_nightly {
protected:
    // use this stream to redirect cout to it,
    // needs to be able check output on warnings
    std::stringstream redirectCoutStream;

    void SetUp() override {
        myriadLayersTests_nightly::SetUp();
        backup = std::cout.rdbuf();
        std::cout.rdbuf(redirectCoutStream.rdbuf());
    }

    void TearDown() override {
        myriadLayersTests_nightly::TearDown();
        std::cout.rdbuf(backup);
        std::cout << redirectCoutStream.str();
    }

private:
    std::streambuf *backup;
};


TEST_F(myriadConfigsWithBlobImportTests_nightly, TryingToSetCompileOptionPrintsWarning)
{
    std::string modelName = "googlenet/bvlc_googlenet_fp16";
    std::ostringstream modelFile;
    modelFile << "/" << modelName << ".xml";
    std::cout << "atata" << std::endl;

    std::ostringstream weightsFile;
    weightsFile << "/" << modelName << ".bin";

    std::string modelFilePath = ModelsPath() + modelFile.str();
    std::string weightsFilePath = ModelsPath() + weightsFile.str();

    ASSERT_NO_THROW(_net_reader.ReadNetwork(modelFilePath));
    ASSERT_TRUE(_net_reader.isParseSuccess());
    ASSERT_NO_THROW(_net_reader.ReadWeights(weightsFilePath));

    ASSERT_EQ(StatusCode::OK, myriadPluginPtr->LoadNetwork(_exeNetwork, _net_reader.getNetwork(), {}, &_resp)) << _resp.msg;
    std::stringstream modelFilenameStream;
    modelFilenameStream << "googlenet" << ".blob";
    ASSERT_EQ(StatusCode::OK, _exeNetwork->Export(modelFilenameStream.str(), &_resp)) << _resp.msg;

    std::map<std::string, std::string> config = { {VPU_CONFIG_KEY(INPUT_NORM), std::to_string(1.f)},
                                                  {VPU_CONFIG_KEY(COMPUTE_LAYOUT), VPU_CONFIG_VALUE(AUTO)},
                                                  {VPU_CONFIG_KEY(HW_ADAPTIVE_MODE), CONFIG_VALUE(YES)},
                                                  {VPU_CONFIG_KEY(COPY_OPTIMIZATION), CONFIG_VALUE(YES)},
                                                  {VPU_CONFIG_KEY(INPUT_BIAS), std::to_string(1.f)},
                                                  {VPU_CONFIG_KEY(IGNORE_UNKNOWN_LAYERS), CONFIG_VALUE(YES)},
                                                  {VPU_CONFIG_KEY(NONE_LAYERS), CONFIG_VALUE(YES)},
                                                  {VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), CONFIG_VALUE(YES)},
                                                  {VPU_CONFIG_KEY(NUMBER_OF_SHAVES), std::to_string(10)},
                                                  {VPU_CONFIG_KEY(NUMBER_OF_CMX_SLICES), std::to_string(10)} };

    InferenceEngine::IExecutableNetwork::Ptr importedNetworkPtr;
    ASSERT_EQ(StatusCode::OK, myriadPluginPtr->ImportNetwork(importedNetworkPtr, modelFilenameStream.str(), config, &_resp)) << _resp.msg;

    std::string content = redirectCoutStream.str();
    for (auto &&elem : config) {
        std::stringstream expectedMsgStream;
        expectedMsgStream << "[Warning][VPU][Config] " << elem.first;
        std::string msg = expectedMsgStream.str();
        ASSERT_TRUE(content.find(msg) != std::string::npos) << msg;
    }
}

TEST_F(myriadConfigsWithBlobImportTests_nightly, TryingToSetRuntimeOptionDoesNotPrintWarning)
{
    std::string modelName = "googlenet/bvlc_googlenet_fp16";
    std::ostringstream modelFile;
    modelFile << "/" << modelName << ".xml";

    std::ostringstream weightsFile;
    weightsFile << "/" << modelName << ".bin";

    std::string modelFilePath = ModelsPath() + modelFile.str();
    std::string weightsFilePath = ModelsPath() + weightsFile.str();

    ASSERT_NO_THROW(_net_reader.ReadNetwork(modelFilePath));
    ASSERT_TRUE(_net_reader.isParseSuccess());
    ASSERT_NO_THROW(_net_reader.ReadWeights(weightsFilePath));

    ASSERT_EQ(StatusCode::OK, myriadPluginPtr->LoadNetwork(_exeNetwork, _net_reader.getNetwork(), {}, &_resp)) << _resp.msg;
    std::stringstream modelFilenameStream;
    modelFilenameStream << "googlenet" << ".blob";
    ASSERT_EQ(StatusCode::OK, _exeNetwork->Export(modelFilenameStream.str(), &_resp)) << _resp.msg;

    std::map<std::string, std::string> config = { {CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), CONFIG_VALUE(YES)},
                                                  {CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_INFO)},
                                                  {VPU_CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_INFO)},
                                                  {CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES)},
                                                  {VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME), CONFIG_VALUE(YES)},
                                                  {VPU_MYRIAD_CONFIG_KEY(FORCE_RESET), CONFIG_VALUE(NO)} };
    if (pluginName.find("HDDLPlugin") != std::string::npos) {
        // Disabled. See CVS-14245
        // config.insert({VPU_HDDL_CONFIG_KEY(GRAPH_TAG), "A"});
    } else if (pluginName.find("myriadPlugin") != std::string::npos) {
        config.insert({VPU_CONFIG_KEY(PLATFORM), VPU_CONFIG_VALUE(2480)});
    } else if (pluginName.find("kmbPlugin") != std::string::npos) {
        config.insert({VPU_CONFIG_KEY(PLATFORM), VPU_CONFIG_VALUE(2480)});
    }


    InferenceEngine::IExecutableNetwork::Ptr importedNetworkPtr;
    ASSERT_EQ(StatusCode::OK, myriadPluginPtr->ImportNetwork(importedNetworkPtr, modelFilenameStream.str(), config, &_resp)) << _resp.msg;

    std::string content = redirectCoutStream.str();
    for (auto &&elem : config) {
        std::stringstream expectedMsgStream;
        expectedMsgStream << "Warning:" << elem.first;
        std::string msg = expectedMsgStream.str();
        ASSERT_EQ(content.find(msg), std::string::npos);
    }
}


using myriadBlobExportAccuracyDifferentCountInAndOutTests_nightly = myriadLayerTestBaseWithParam<std::string>;

TEST_P(myriadBlobExportAccuracyDifferentCountInAndOutTests_nightly, IsResultOfImportedAndGeneratedModelSame)
{
    SetSeed(DEFAULT_SEED_VALUE);
    std::string modelName = GetParam();
    std::ostringstream modelFile;
    modelFile << "/" << modelName << ".xml";

    std::ostringstream weightsFile;
    weightsFile << "/" << modelName << ".bin";

    std::string modelFilePath = ModelsPath() + modelFile.str();
    std::string weightsFilePath = ModelsPath() + weightsFile.str();

    ASSERT_NO_THROW(_net_reader.ReadNetwork(modelFilePath));
    ASSERT_TRUE(_net_reader.isParseSuccess());
    ASSERT_NO_THROW(_net_reader.ReadWeights(weightsFilePath));

    InferenceEngine::IExecutableNetwork::Ptr originalExeNetworkPtr;
    ASSERT_EQ(StatusCode::OK, myriadPluginPtr->LoadNetwork(originalExeNetworkPtr, _net_reader.getNetwork(), { }, &_resp)) << _resp.msg;

    ConstInputsDataMap originalInputsInfo;
    ASSERT_EQ(StatusCode::OK, originalExeNetworkPtr->GetInputsInfo(originalInputsInfo, &_resp)) << _resp.msg;

    InferenceEngine::IInferRequest::Ptr orignalInferRequest;
    ASSERT_EQ(StatusCode::OK, originalExeNetworkPtr->CreateInferRequest(orignalInferRequest, &_resp)) << _resp.msg;

    std::vector<Blob::Ptr> inputBlobs(originalInputsInfo.size());
    auto inputBlobsIt = inputBlobs.begin();
    for (const auto &inputInfo : originalInputsInfo) {
        ASSERT_EQ(StatusCode::OK, orignalInferRequest->GetBlob(inputInfo.first.c_str(), *inputBlobsIt, &_resp)) << _resp.msg;
        GenRandomData(*inputBlobsIt);
        inputBlobsIt++;
    }

    ASSERT_EQ(StatusCode::OK, orignalInferRequest->Infer(&_resp)) << _resp.msg;

    ConstOutputsDataMap orignalOutputsInfo;
    ASSERT_EQ(StatusCode::OK, originalExeNetworkPtr->GetOutputsInfo(orignalOutputsInfo, &_resp)) << _resp.msg;

    std::vector<Blob::Ptr> originalOutputBlobs(orignalOutputsInfo.size());
    auto outputBlobsIt = originalOutputBlobs.begin();
    for (const auto &outputInfo: orignalOutputsInfo) {
        ASSERT_EQ(StatusCode::OK, orignalInferRequest->GetBlob(outputInfo.first.c_str(), *outputBlobsIt, &_resp)) << _resp.msg;
        outputBlobsIt++;
    }

    std::stringstream modelFilenameStream;
    modelFilenameStream << "exportedModel" << ".blob";
    ASSERT_EQ(StatusCode::OK, originalExeNetworkPtr->Export(modelFilenameStream.str(), &_resp)) << _resp.msg;

    InferenceEngine::IExecutableNetwork::Ptr importedNetworkPtr;
    ASSERT_EQ(StatusCode::OK, myriadPluginPtr->ImportNetwork(importedNetworkPtr, modelFilenameStream.str(), {}, &_resp)) << _resp.msg;
    InferenceEngine::IInferRequest::Ptr importedInferRequest;
    ASSERT_EQ(StatusCode::OK, importedNetworkPtr->CreateInferRequest(importedInferRequest, &_resp)) << _resp.msg;

    ConstInputsDataMap importedInputsInfo;
    ASSERT_EQ(StatusCode::OK, importedNetworkPtr->GetInputsInfo(importedInputsInfo, &_resp)) << _resp.msg;

    inputBlobsIt = inputBlobs.begin();
    for (const auto &inputInfo : importedInputsInfo) {
        ASSERT_EQ(StatusCode::OK, importedInferRequest->SetBlob(inputInfo.first.c_str(), *inputBlobsIt, &_resp)) << &_resp.msg;
        inputBlobsIt++;
    }

    ASSERT_EQ(StatusCode::OK, importedInferRequest->Infer(&_resp)) << _resp.msg;

    ConstOutputsDataMap importedOutputsInfo;
    ASSERT_EQ(StatusCode::OK, importedNetworkPtr->GetOutputsInfo(importedOutputsInfo, &_resp)) << _resp.msg;

    outputBlobsIt = originalOutputBlobs.begin();
    for (const auto &outputInfo : importedOutputsInfo) {
        Blob::Ptr importedOutputBlobPtr;
        ASSERT_EQ(StatusCode::OK, importedInferRequest->GetBlob(outputInfo.first.c_str(), importedOutputBlobPtr, &_resp)) << _resp.msg;

        Compare(importedOutputBlobPtr, *outputBlobsIt, 0.f);
        outputBlobsIt++;
    }
}

static const std::vector<std::string> modelNames = {"googlenet/bvlc_googlenet_fp16", "vpu_validation/age_gender_fp16",
                                                    "vpu_validation/reid_distance_fp16", "vpu_validation/LPRNet_48_fp16"};


INSTANTIATE_TEST_CASE_P(accuracy, myriadBlobExportAccuracyDifferentCountInAndOutTests_nightly, ::testing::ValuesIn(modelNames));

using myriadBlobExportAccuracyDifferentPrecisionOfInAndOutTests_nightly = myriadLayerTestBaseWithParam<std::tuple<InferenceEngine::Precision, InferenceEngine::Precision>>;

TEST_P(myriadBlobExportAccuracyDifferentPrecisionOfInAndOutTests_nightly, IsResultOfImportedAndGeneratedModelSame)
{
    SetSeed(DEFAULT_SEED_VALUE);
    InferenceEngine::Precision inputPrecision = std::get<0>(GetParam());
    InferenceEngine::Precision outputPrecision = std::get<1>(GetParam());
    std::string modelName = "googlenet/bvlc_googlenet_fp16";
    std::ostringstream modelFile;
    modelFile << "/" << modelName << ".xml";

    std::ostringstream weightsFile;
    weightsFile << "/" << modelName << ".bin";

    std::string modelFilePath = ModelsPath() + modelFile.str();
    std::string weightsFilePath = ModelsPath() + weightsFile.str();

    ASSERT_NO_THROW(_net_reader.ReadNetwork(modelFilePath));
    ASSERT_TRUE(_net_reader.isParseSuccess());
    ASSERT_NO_THROW(_net_reader.ReadWeights(weightsFilePath));

    auto network = _net_reader.getNetwork();
    InputsDataMap inputsInfo = network.getInputsInfo();
    ASSERT_EQ(inputsInfo.size(), 1);
    auto inputInfo = inputsInfo.begin();
    ASSERT_NO_THROW(inputInfo->second->setInputPrecision(inputPrecision));

    OutputsDataMap outputsInfo = network.getOutputsInfo();
    ASSERT_EQ(outputsInfo.size(), 1);
    auto outputInfo = outputsInfo.begin();
    ASSERT_NO_THROW(outputInfo->second->setPrecision(outputPrecision));

    InferenceEngine::IExecutableNetwork::Ptr originalExeNetworkPtr;
    ASSERT_EQ(StatusCode::OK, myriadPluginPtr->LoadNetwork(originalExeNetworkPtr, network, { }, &_resp)) << _resp.msg;

    InferenceEngine::IInferRequest::Ptr orignalInferRequest;
    ASSERT_EQ(StatusCode::OK, originalExeNetworkPtr->CreateInferRequest(orignalInferRequest, &_resp)) << _resp.msg;

    Blob::Ptr inputBlobPtr;
    ASSERT_EQ(StatusCode::OK, orignalInferRequest->GetBlob(inputInfo->first.c_str(), inputBlobPtr, &_resp)) << _resp.msg;
    GenRandomData(inputBlobPtr);

    ASSERT_EQ(StatusCode::OK, orignalInferRequest->Infer(&_resp)) << _resp.msg;

    Blob::Ptr outputBlobPtr;
    ASSERT_EQ(StatusCode::OK, orignalInferRequest->GetBlob(outputInfo->first.c_str(), outputBlobPtr, &_resp)) << _resp.msg;

    std::stringstream modelFilenameStream;
    modelFilenameStream << "exportedModel" << ".blob";
    ASSERT_EQ(StatusCode::OK, originalExeNetworkPtr->Export(modelFilenameStream.str(), &_resp)) << _resp.msg;

    InferenceEngine::IExecutableNetwork::Ptr importedNetworkPtr;
    ASSERT_EQ(StatusCode::OK, myriadPluginPtr->ImportNetwork(importedNetworkPtr, modelFilenameStream.str(), {}, &_resp)) << _resp.msg;
    InferenceEngine::IInferRequest::Ptr importedInferRequest;
    ASSERT_EQ(StatusCode::OK, importedNetworkPtr->CreateInferRequest(importedInferRequest, &_resp)) << _resp.msg;

    ConstInputsDataMap importedInputsInfo;
    ASSERT_EQ(StatusCode::OK, importedNetworkPtr->GetInputsInfo(importedInputsInfo, &_resp)) << _resp.msg;
    ASSERT_EQ(importedInputsInfo.size(), 1);
    auto importedInputInfo = importedInputsInfo.begin();

    ASSERT_EQ(StatusCode::OK, importedInferRequest->SetBlob(importedInputInfo->first.c_str(), inputBlobPtr, &_resp)) << &_resp.msg;

    ASSERT_EQ(StatusCode::OK, importedInferRequest->Infer(&_resp)) << _resp.msg;

    ConstOutputsDataMap importedOutputsInfo;
    ASSERT_EQ(StatusCode::OK, importedNetworkPtr->GetOutputsInfo(importedOutputsInfo, &_resp)) << _resp.msg;
    ASSERT_EQ(importedOutputsInfo.size(), 1);
    auto importedOutputInfo = importedOutputsInfo.begin();

    Blob::Ptr importedOutputBlobPtr;
    ASSERT_EQ(StatusCode::OK, importedInferRequest->GetBlob(importedOutputInfo->first.c_str(), importedOutputBlobPtr, &_resp)) << _resp.msg;

    Compare(importedOutputBlobPtr, outputBlobPtr, 0.f);
}

static const std::vector<InferenceEngine::Precision> inputPrecisions = {InferenceEngine::Precision::U8, InferenceEngine::Precision::FP16,
                                                                       InferenceEngine::Precision::FP32};

static const std::vector<InferenceEngine::Precision> outputPrecisions = {InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP32};


INSTANTIATE_TEST_CASE_P(accuracy, myriadBlobExportAccuracyDifferentPrecisionOfInAndOutTests_nightly,
                        ::testing::Combine(::testing::ValuesIn(inputPrecisions), ::testing::ValuesIn(outputPrecisions)));

using myriadExtraTests_nightly = myriadLayersTests_nightly;

TEST_F(myriadExtraTests_nightly, ThereIsNoSegfaultOnZeroConvolutionWeights) {
    if (!CheckMyriadX()) {
        SKIP() << "Non-MyriadX device";
    }

    tensor_test_params input_dims = { 1, 3, 25, 25 };
    param_size kernel = { 3, 3 };
    param_size stride = { 1, 1 };
    param_size pad = { 1, 1 };
    size_t out_channels = 3;
    size_t group = 1;
    param_size dilation_factor = { 1, 1 };

    size_t out_w = (input_dims.w + 2 * pad.x - dilation_factor.x * (kernel.x - 1) - 1 + stride.x) / stride.x;
    size_t out_h = (input_dims.h + 2 * pad.y - dilation_factor.y * (kernel.y - 1) - 1 + stride.y) / stride.y;

    tensor_test_params output_dims = { 1, out_channels, out_h, out_w };

    SetInputTensor(input_dims);
    SetOutputTensor(output_dims);

    size_t num_weights = kernel.x * kernel.y * (input_dims.c / group) * output_dims.c;
    size_t num_bias = output_dims.c;

    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr =
        InferenceEngine::TBlob<uint8_t>::Ptr(GenWeights(num_weights + num_bias));
    ie_fp16* weights = weights_ptr->data().as<ie_fp16*>();
    // set a small number in FP16
    for (size_t i = 0; i < num_weights + num_bias; i++) {
        weights[i] = 0;
    }

    std::map<std::string, std::string> layer_params = {
        { "kernel-x", std::to_string(kernel.x) }
        ,{ "kernel-y", std::to_string(kernel.y) }
        ,{ "stride-x", std::to_string(stride.x) }
        ,{ "stride-y", std::to_string(stride.y) }
        ,{ "pad-x", std::to_string(pad.x) }
        ,{ "pad-y", std::to_string(pad.y) }
        ,{ "output", std::to_string(out_channels) }
        ,{ "group", std::to_string(group) }
        ,{ "dilation-x", std::to_string(dilation_factor.x) }
        ,{ "dilation-y", std::to_string(dilation_factor.y) }
    };
    NetworkInit("Convolution",
        &layer_params,
        num_weights * sizeof(uint16_t),
        num_bias * sizeof(uint16_t),
        weights_ptr,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::FP16,
        true);
}
