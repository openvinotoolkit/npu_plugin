//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "sync_infer_request.hpp"
#include "vpux/utils/IE/prefix.hpp"

#include <cpp_interfaces/plugin_itt.hpp>
#include <openvino/runtime/make_tensor.hpp>
#include <transformations/utils/utils.hpp>

namespace vpux {

SyncInferRequest::SyncInferRequest(const std::shared_ptr<const ov::ICompiledModel> compiledModel,
                                   const std::shared_ptr<const NetworkDescription> networkDescription)
        : _compiledModel(compiledModel),
          _inputNames(networkDescription->getInputNames()),
          _outputNames(networkDescription->getOutputNames()),
          _stateNames(networkDescription->getStateNames()),
          _parameterDescriptors(networkDescription->getParameterDescriptors()),
          _resultDescriptors(networkDescription->getResultDescriptors()),
          _stateDescriptors(networkDescription->getStateDescriptors()) {
    OPENVINO_ASSERT(_compiledModel);

    const std::vector<ov::Output<const ov::Node>>& inputs = get_inputs();
    const std::vector<ov::Output<const ov::Node>>& outputs = get_outputs();

    if (inputs.empty()) {
        OPENVINO_THROW("Inference request creation: no input found for network " + networkDescription->getName());
    }
    if (outputs.empty()) {
        OPENVINO_THROW("Inference request creation: no output found for network " + networkDescription->getName());
    }

    // Map the node names to the legacy ones used by the I/O tensors in order to allow an easier access to the tensors'
    // contents
    for (const auto& [legacyName, parameterDescriptor] : _parameterDescriptors) {
        _nodeNameToLegacyName[parameterDescriptor.currentNodeName] = legacyName;
    }
    for (const auto& [legacyName, resultDescriptor] : _resultDescriptors) {
        _nodeNameToLegacyName[resultDescriptor.currentNodeName] = legacyName;
    }

    _inputAndStateInputNames = _inputNames;
    _outputAndStateOutputNames = _outputNames;

    for (const std::string& stateName : _stateNames) {
        // State variables shall be identified by an "assign" prefix in order to avoid a potential tensor name collision
        _inputAndStateInputNames.push_back(ASSIGN_PREFIX + stateName);
        _outputAndStateOutputNames.push_back(ASSIGN_PREFIX + stateName);
    }
}

const std::vector<ov::Output<const ov::Node>>& SyncInferRequest::get_inputs() const {
    return _compiledModel->inputs();
}

const std::vector<ov::Output<const ov::Node>>& SyncInferRequest::get_outputs() const {
    return _compiledModel->outputs();
}

const std::shared_ptr<const ov::ICompiledModel>& SyncInferRequest::get_compiled_model() const {
    return _compiledModel;
}

void SyncInferRequest::initialize_states() {
    for (const std::string& stateName : _stateNames) {
        _variableStates.at(stateName)->reset();
    }
}

std::vector<ov::SoPtr<ov::IVariableState>> SyncInferRequest::query_state() const {
    std::vector<ov::SoPtr<ov::IVariableState>> queryResult;

    for (const std::string& stateName : _stateNames) {
        queryResult.push_back(_variableStates.at(stateName));
    }

    return queryResult;
}

ov::SoPtr<ov::ITensor> SyncInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    const auto& nodeNameMatch = _nodeNameToLegacyName.find(port.get_node()->get_friendly_name());
    OPENVINO_ASSERT(nodeNameMatch != _nodeNameToLegacyName.end(), "Cannot find tensor for port ", port);

    return _allTensors.at(nodeNameMatch->second);
}

void SyncInferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
    OV_ITT_SCOPED_TASK(InferenceEngine::itt::domains::Plugin, "set_tensor");
    try {
        check_tensor(port, tensor);
    } catch (const ov::Exception& ex) {
        OPENVINO_THROW("Failed to set tensor. ", ex.what());
    }

    const std::string& legacyName = _nodeNameToLegacyName.at(port.get_node()->get_friendly_name());
    _allTensors[legacyName] = tensor._ptr;
}

std::vector<ov::SoPtr<ov::ITensor>> SyncInferRequest::get_tensors(const ov::Output<const ov::Node>& /*port*/) const {
    OV_ITT_SCOPED_TASK(InferenceEngine::itt::domains::Plugin, "get_tensors");
    // Using batches of tensors is currently not supported by the NPU plugin. In this scenario, the OpenVINO API demands
    // returning an empty vector.
    return {};
}

void SyncInferRequest::set_tensors(const ov::Output<const ov::Node>& port,
                                   const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
    OV_ITT_SCOPED_TASK(InferenceEngine::itt::domains::Plugin, "set_tensors");
    if (tensors.size() == 1) {
        set_tensor(port, tensors[0]);
        return;
    }

    OPENVINO_ASSERT_HELPER(::ov::NotImplemented, "", false, "Not Implemented",
                           "set_input_tensors/set_tensors are not supported by this plugin");
}

void SyncInferRequest::check_tensor(const ov::Output<const ov::Node>& port,
                                    const ov::SoPtr<ov::ITensor>& tensor) const {
    if (tensor == nullptr)
        OPENVINO_THROW("The tensor is not initialized!");

    bool is_input = ov::op::util::is_parameter(port.get_node());
    std::string tensor_type = is_input ? "input" : "output";

    OPENVINO_ASSERT(port.get_element_type() == tensor->get_element_type(),
                    "The tensor element type is not corresponding with output element type (",
                    tensor->get_element_type(), " != ", port.get_element_type());
    bool is_dynamic = port.get_partial_shape().is_dynamic();
    OPENVINO_ASSERT(is_dynamic || port.get_shape() == tensor->get_shape(), "The ", tensor_type,
                    " tensor size is not equal to the model ", tensor_type, " type: got ", tensor->get_shape(),
                    " expecting ", port.get_shape(), ".");
    OPENVINO_ASSERT(
            std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr) || tensor->data() != nullptr || is_dynamic,
            "Tensor data equal nullptr!");
}

void SyncInferRequest::check_tensors() const {
    const auto& inputs = _compiledModel->inputs();
    for (size_t i = 0; i < inputs.size(); i++) {
        const std::string& legacyName = _nodeNameToLegacyName.at(inputs[i].get_node()->get_friendly_name());
        check_tensor(inputs[i], _allTensors.at(legacyName));
    }

    const auto& outputs = _compiledModel->outputs();
    for (size_t i = 0; i < outputs.size(); i++) {
        const std::string& legacyName = _nodeNameToLegacyName.at(outputs[i].get_node()->get_friendly_name());
        check_tensor(outputs[i], _allTensors.at(legacyName));
    }
}

void SyncInferRequest::allocate_tensor(std::string tensorName, const IONodeDescriptor& descriptor, void* dataBuffer,
                                       const bool isState) {
    std::shared_ptr<ov::ITensor> tensor;

    check_network_precision(descriptor.precision);

    if (dataBuffer) {
        tensor = ov::make_tensor(descriptor.precision, descriptor.transposedShape.get_shape(), dataBuffer);
    } else {
        tensor = ov::make_tensor(descriptor.precision, descriptor.transposedShape.get_shape());
    }

    if (isState) {
        _variableStates[tensorName] = std::make_shared<vpux::VariableState>(tensorName, tensor);

        // State variables shall be identified by an "assign" prefix in order to avoid a potential tensor name collision
        tensorName = ASSIGN_PREFIX + tensorName;
    }
    _allTensors[tensorName] = std::move(tensor);
}

}  // namespace vpux
