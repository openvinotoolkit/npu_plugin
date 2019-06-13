// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include "myriad_xml_tests.hpp"

using namespace InferenceEngine;
using GetOutputTestsParams = std::tuple<std::tuple<std::string*, std::string*>, std::string>;

class myriadGetOutput_nightly :
        public myriadLayersTests_nightly,
        public testing::WithParamInterface<GetOutputTestsParams> {
public:
    std::string name_model_full;
    std::string name_model_crop;
    std::string name_output;
};

TEST_P(myriadGetOutput_nightly, AddOutput) {
    StatusCode st;

    name_model_full = (*(std::get<0>(std::get<0>(GetParam()))));
    name_model_crop = (*(std::get<1>(std::get<0>(GetParam()))));
    name_output = std::get<1>(GetParam());

    TBlob<uint8_t>::Ptr weights(GenWeights( ( 32786944 + 2000) / sizeof(ie_fp16),  0, 1));

    CNNNetReader crop_reader;
    ASSERT_NO_THROW(crop_reader.ReadNetwork(name_model_crop.c_str(), name_model_crop.length()));
    ASSERT_TRUE(crop_reader.isParseSuccess());
    ASSERT_NO_THROW(crop_reader.SetWeights(weights));

    auto crop_network = crop_reader.getNetwork();

    InferenceEngine::InputsDataMap networkInputs;
    ASSERT_NO_THROW(networkInputs = crop_network.getInputsInfo());
    InferenceEngine::OutputsDataMap networkOutputs;
    ASSERT_NO_THROW(networkOutputs = crop_network.getOutputsInfo());

    networkInputs.begin()->second->setInputPrecision(InferenceEngine::Precision::FP16);
    networkOutputs.begin()->second->setPrecision(InferenceEngine::Precision::FP16);

    InferenceEngine::Blob::Ptr inputBlob;

    InferenceEngine::IExecutableNetwork::Ptr exeNetwork;
    std::map<std::string, std::string> networkConfig;
    ASSERT_NO_THROW(st = myriadPluginPtr->LoadNetwork(exeNetwork, crop_network, networkConfig, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    ASSERT_NE(exeNetwork, nullptr) << _resp.msg;

    InferenceEngine::IInferRequest::Ptr inferRequest;
    ASSERT_NO_THROW(st = exeNetwork->CreateInferRequest(inferRequest, &_resp));

    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = inferRequest->GetBlob(networkInputs.begin()->first.c_str(), inputBlob, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    GenRandomData(inputBlob);

    InferenceEngine::Blob::Ptr output_crop;
    ASSERT_NO_THROW(st = inferRequest->Infer(&_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    ASSERT_NO_THROW(st = inferRequest->GetBlob(networkOutputs.begin()->first.c_str(), output_crop, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    /*Full Network Infer */

    CNNNetReader full_reader;
    ASSERT_NO_THROW(full_reader.ReadNetwork(name_model_full.c_str(), name_model_full.length()));
    ASSERT_TRUE(full_reader.isParseSuccess());
    ASSERT_NO_THROW(full_reader.SetWeights(weights));

    auto full_network = full_reader.getNetwork();

    full_network.addOutput(name_output, 0);

    InferenceEngine::InputsDataMap networkInputsFull;
    networkInputsFull = full_network.getInputsInfo();
    InferenceEngine::OutputsDataMap networkOutputsFull;
    networkOutputsFull = full_network.getOutputsInfo();

    networkInputsFull.begin()->second->setInputPrecision(InferenceEngine::Precision::FP16);
    networkOutputsFull.begin()->second->setPrecision(InferenceEngine::Precision::FP16);
    (++networkOutputsFull.begin())->second->setPrecision(InferenceEngine::Precision::FP16);

    InferenceEngine::IExecutableNetwork::Ptr exeNetworkFull;
    ASSERT_NO_THROW(st = myriadPluginPtr->LoadNetwork(exeNetworkFull, full_network, networkConfig, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    InferenceEngine::IInferRequest::Ptr inferRequestFull;
    ASSERT_NO_THROW(st = exeNetworkFull->CreateInferRequest(inferRequestFull, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = inferRequestFull->SetBlob("data", inputBlob, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    InferenceEngine::Blob::Ptr output_full;
    ASSERT_NO_THROW(st = inferRequestFull->Infer(&_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    ASSERT_NO_THROW(st = inferRequestFull->GetBlob(name_output.c_str(), output_full, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    Compare(output_full, output_crop, 0.0f);
}

std::string getTestCaseName(const testing::TestParamInfo<GetOutputTestsParams>& param) {
    return  "addOutput_" + std::get<1>(param.param);
}

INSTANTIATE_TEST_CASE_P(Test_params_pool, myriadGetOutput_nightly,
        testing::Values(
                std::make_tuple(std::make_tuple(&full_model, &poolModel), "pool1_3x3_s2"),
                std::make_tuple(std::make_tuple(&full_model, &convModel), "conv1_7x7_s2"),
                std::make_tuple(std::make_tuple(&full_model, &reluConvModel), "conv1_relu_7x7"),
                std::make_tuple(std::make_tuple(&full_model, &fcModel), "loss3_classifier"),
                std::make_tuple(std::make_tuple(&full_model, &reluFcModel), "ReluFC"),
                std::make_tuple(std::make_tuple(&concatModel, &concatModelConv), "conv1_2")
        ),
    getTestCaseName
);

class myriadCheckOutput_nightly :
        public myriadLayersTests_nightly {
};

TEST_F(myriadCheckOutput_nightly, ICVPDOutput) {
    StatusCode st;

    std::string bbox_name = "bbox_pred_reshape";
    std::string proposal_name = "cls_prob_reshape";
    std::string prob_name = "proposal";

    std::string model = "/icv-pd/icv-pd-hypernet-rfcn-wo-detOutput_fp16";
    _net_reader = CNNNetReader();

    std::ostringstream modelFile;
    modelFile << "/" << model << ".xml";

    std::ostringstream weightsFile;
    weightsFile << "/" << model << ".bin";

    std::string modelFilePath = ModelsPath() + modelFile.str();
    std::string weightsFilePath = ModelsPath() + weightsFile.str();

    ASSERT_NO_THROW(_net_reader.ReadNetwork(modelFilePath));
    ASSERT_TRUE(_net_reader.isParseSuccess());
    ASSERT_NO_THROW(_net_reader.ReadWeights(weightsFilePath));

    auto network = _net_reader.getNetwork();
    auto inputsInfo = network.getInputsInfo();

    network.addOutput(bbox_name, 0);
    network.addOutput(prob_name, 0);
    network.addOutput(proposal_name, 0);

    auto outputsInfo = network.getOutputsInfo();

    for (auto output = outputsInfo.begin(); output != outputsInfo.end(); ++output) {
      output->second->setPrecision(InferenceEngine::Precision::FP32);
    }

    InferenceEngine::IExecutableNetwork::Ptr exeNetwork;
    std::map<std::string, std::string> networkConfig;
    ASSERT_NO_THROW(st = myriadPluginPtr->LoadNetwork(exeNetwork, network, networkConfig, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    ASSERT_NE(exeNetwork, nullptr) << _resp.msg;

    InferenceEngine::IInferRequest::Ptr inferRequest;
    ASSERT_NO_THROW(st = exeNetwork->CreateInferRequest(inferRequest, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    InferenceEngine::Blob::Ptr inputBlob;
    ASSERT_NO_THROW(st = inferRequest->GetBlob(inputsInfo.begin()->first.c_str(), inputBlob, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    GenRandomData(inputBlob);

    ASSERT_NO_THROW(st = inferRequest->SetBlob("data", inputBlob, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = inferRequest->Infer(&_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
}
