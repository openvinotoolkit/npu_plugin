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

#include "hddl2_infer_request.h"

#include <InferBlob.h>

#include <algorithm>
#include <functional>
#include <map>
#include <string>
#include <vector>

/**
 * @brief multiply vector's values
 * @param vec - vector with values
 * @return result of multiplication
 */
template <typename T, typename A>
static T product(std::vector<T, A> const& vec) {
    if (vec.empty()) return 0;
    T ret = vec[0];
    for (size_t i = 1; i < vec.size(); ++i) ret *= vec[i];
    return ret;
}

vpu::HDDL2Plugin::HDDL2InferRequest::HDDL2InferRequest(const InferenceEngine::InputsDataMap& networkInputs,
    const InferenceEngine::OutputsDataMap& networkOutputs, HddlUnite::Inference::Graph::Ptr graph)
    : InferRequestInternal(networkInputs, networkOutputs), _graph(graph) {
    // _inputSize = _graph->getInputSize();
    // _outputSize = _graph->getOutputSize();
    for (auto& networkInput : _networkInputs) {
        _inputSize = std::accumulate(networkInput.second->getInputData()->getDims().begin(),
            networkInput.second->getInputData()->getDims().end(), 1, std::multiplies<size_t>());
    }
    for (auto& networkOutput : _networkOutputs) {
        _outputSize = std::accumulate(networkOutput.second->getTensorDesc().getDims().begin(),
            networkOutput.second->getTensorDesc().getDims().end(), 1, std::multiplies<size_t>());
    }

    _inferData = makeInferData(types);
    if (_inferData.get() == nullptr) THROW_IE_EXCEPTION << "inferData == nullptr";

    _inferData->createBlob(
        "input", HddlUnite::Inference::BlobDesc(HddlUnite::Inference::Precision::U8, false, true, _inputSize), true);

    _inferData->createBlob(
        "output", HddlUnite::Inference::BlobDesc(HddlUnite::Inference::Precision::U8, false, true, _outputSize), false);

    IE_ASSERT(_networkInputs.size() == 1) << "Do not support more than 1 input";
    for (auto& networkInput : _networkInputs) {
        InferenceEngine::SizeVector dims = networkInput.second->getTensorDesc().getDims();
        InferenceEngine::Precision precision = networkInput.second->getTensorDesc().getPrecision();
        InferenceEngine::Layout layout = networkInput.second->getTensorDesc().getLayout();

        if (precision != InferenceEngine::Precision::U8) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported input precision: " << precision
                               << "! Supported precisions only U8";
        }

        _inputs[networkInput.first] = make_blob_with_precision(InferenceEngine::TensorDesc(precision, dims, layout));
        if (_inputs[networkInput.first] == nullptr) THROW_IE_EXCEPTION << "InputBlob is nullptr.";
        _inputs[networkInput.first]->allocate();
    }

    // allocate outputs
    IE_ASSERT(_networkOutputs.size() == 1) << "Do not support more than 1 output";
    for (auto& networkOutput : _networkOutputs) {
        InferenceEngine::SizeVector dims = networkOutput.second->getTensorDesc().getDims();
        InferenceEngine::Precision precision = networkOutput.second->getTensorDesc().getPrecision();
        InferenceEngine::Layout layout = networkOutput.second->getTensorDesc().getLayout();

        if (precision != InferenceEngine::Precision::FP32 && precision != InferenceEngine::Precision::FP16 &&
            precision != InferenceEngine::Precision::U8 && precision != InferenceEngine::Precision::I8) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported output precision: " << precision
                               << "! Supported precisions: FP32, FP16, U8, I8";
        }

        _outputs[networkOutput.first] = make_blob_with_precision(InferenceEngine::TensorDesc(precision, dims, layout));
        if (_outputs[networkOutput.first] == nullptr) THROW_IE_EXCEPTION << "InputBlob is nullptr.";
        _outputs[networkOutput.first]->allocate();
    }

    if (_networkOutputs.empty() || _networkInputs.empty())
        THROW_IE_EXCEPTION << "Internal error: no information about network's output/input";
}

void vpu::HDDL2Plugin::HDDL2InferRequest::InferImpl() {
    InferSync();
    GetResult();
}

void vpu::HDDL2Plugin::HDDL2InferRequest::Infer() { InferImpl(); }

void vpu::HDDL2Plugin::HDDL2InferRequest::GetPerformanceCounts(
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& perfMap) const {
    UNUSED(perfMap);
    std::cout << "GetPerformanceCounts call" << std::endl;
}

void vpu::HDDL2Plugin::HDDL2InferRequest::GetResult() {
    auto dataName = _networkOutputs.begin()->first;
    auto foundOutputBlob = _outputs.find(dataName);
    if (foundOutputBlob == _outputs.end()) THROW_IE_EXCEPTION << "Error: output [" << dataName << "] is not provided.";

    if (_outputSize != foundOutputBlob->second->byteSize()) THROW_IE_EXCEPTION << "_outputSize != data->byteSize()";

    auto outputBlob = _inferData->getOutputBlob("output");
    auto outputData = outputBlob->getData();
}

void vpu::HDDL2Plugin::HDDL2InferRequest::InferSync() {
    auto dataName = _networkInputs.begin()->first;
    auto foundInputBlob = _inputs.find(dataName);
    if (foundInputBlob == _inputs.end()) THROW_IE_EXCEPTION << "Error: input [" << dataName << "] is not provided.";

    auto inputBlob = _inferData->getInputBlob("input");
    auto outputBlob = _inferData->getOutputBlob("output");

    HddlUnite::Inference::BlobDesc inputDesc(HddlUnite::Inference::Precision::U8, false, true, _inputSize);
    // TODO: parameters rectangle (image size): x, y, width, height
    inputDesc.m_rect.push_back({0, 0, 224, 224});
    for (auto& networkInput : _networkInputs) {
        inputDesc.m_srcPtr = _inputs[networkInput.first]->buffer().as<void*>();
        inputDesc.m_dataSize = _inputs[networkInput.first]->byteSize();
    }
    inputBlob->updateBlob(inputDesc);

    HddlStatusCode syncCode = inferSync(*(_graph.get()), _inferData);
    if (syncCode != HddlStatusCode::HDDL_OK) THROW_IE_EXCEPTION << "InferSync FAILED! return code:" << syncCode;
}
