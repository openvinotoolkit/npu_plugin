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

#include <hddl2_executable_network.h>
#include <hddl2_helpers.h>
#include <hddl2_infer_request.h>
#include <net_pass.h>

#include <algorithm>
#include <fstream>
#include <mcm_adapter.hpp>
#include <memory>
#include <string>
#include <vector>

// TODO: Get input/output info with custom parameters. HDDl cannot extract them.
InferenceEngine::InputsDataMap getCustomInputInfo() {
    InferenceEngine::InputsDataMap m_networkInputs;
    InferenceEngine::SizeVector inputDims({1, 3, 224, 224});
    InferenceEngine::Layout inputLayout = InferenceEngine::Layout::NCHW;
    InferenceEngine::Precision inputPrecision = InferenceEngine::Precision::U8;
    InferenceEngine::TensorDesc inputDesc(inputPrecision, inputDims, inputLayout);
    InferenceEngine::Data inputData("input", inputDesc);

    InferenceEngine::InputInfo inputInfo;
    inputInfo.setInputData(std::make_shared<InferenceEngine::Data>(inputData));
    m_networkInputs[inputInfo.name()] = std::make_shared<InferenceEngine::InputInfo>(inputInfo);
    return m_networkInputs;
}

InferenceEngine::OutputsDataMap getCustomOutputInfo() {
    InferenceEngine::OutputsDataMap m_networkOutputs;
    InferenceEngine::SizeVector outputDims({1, 1024, 1, 1});
    InferenceEngine::Layout outputLayout = InferenceEngine::Layout::NCHW;
    InferenceEngine::Precision outputPrecision = InferenceEngine::Precision::U8;
    InferenceEngine::TensorDesc outputDesc(outputPrecision, outputDims, outputLayout);

    InferenceEngine::Data outputData("output", outputDesc);
    m_networkOutputs[outputData.getName()] = std::make_shared<InferenceEngine::Data>(outputData);
    return m_networkOutputs;
}

InferenceEngine::InferRequestInternal::Ptr vpu::HDDL2Plugin::ExecutableNetwork::CreateInferRequestImpl(
    InferenceEngine::InputsDataMap networkInputs, InferenceEngine::OutputsDataMap networkOutputs) {
    return std::make_shared<HDDL2InferRequest>(networkInputs, networkOutputs, _graph);
}

vpu::HDDL2Plugin::ExecutableNetwork::ExecutableNetwork(const std::string& blobFilename, const HDDL2Config& config)
    : _config(config) {
    HddlStatusCode code = getAvailableDevices(_devices);
    if (code != HddlStatusCode::HDDL_OK) {
        THROW_IE_EXCEPTION << "getAvailableDevices != StatusCode::OK; " << code;
    }

    HddlStatusCode status = loadGraph(_graph, "resnet", blobFilename, _devices);
    if (status != HddlStatusCode::HDDL_OK) THROW_IE_EXCEPTION << "[ERROR] -load graph error: " << status;

    std::ifstream blobFile(blobFilename, std::ios::binary);
    if (!blobFile.is_open()) {
        THROW_IE_EXCEPTION << "[ERROR] *Could not open file: " << blobFilename;
    }

    std::ostringstream blobContentStream;
    blobContentStream << blobFile.rdbuf();
    const std::string& blobContentString = blobContentStream.str();
    std::copy(blobContentString.begin(), blobContentString.end(), std::back_inserter(_graphBlob));

    this->_networkInputs = getCustomInputInfo();
    this->_networkOutputs = getCustomOutputInfo();
}

vpu::HDDL2Plugin::ExecutableNetwork::ExecutableNetwork(InferenceEngine::ICNNNetwork& network, const HDDL2Config& config)
    : _config(config) {
#ifdef ENABLE_MCM_COMPILER
    vpu::MCMAdapter::compileNetwork(network, _config, _graphBlob);

    HddlStatusCode status = loadGraph(_graph, "resnet", _graphBlob.data(), _graphBlob.size(), _devices);
    if (status != HddlStatusCode::HDDL_OK) THROW_IE_EXCEPTION << "[ERROR] -load graph error: " << status;

    network.getInputsInfo(_networkInputs);
    network.getOutputsInfo(_networkOutputs);
#else
    UNUSED(network);
#endif
}

vpu::HDDL2Plugin::ExecutableNetwork::~ExecutableNetwork() {
    if (_graph != nullptr) unloadGraph(_graph, _devices);
}
