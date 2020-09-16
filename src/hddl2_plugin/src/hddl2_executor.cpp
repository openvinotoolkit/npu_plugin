//
// Copyright 2020 Intel Corporation.
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

#include "hddl2_executor.h"

#include <hddl2_remote_blob.h>
#include <ie_compound_blob.h>
#include <ie_memcpy.h>

#include <blob_factory.hpp>
#include <ie_preprocess.hpp>
#include <ie_remote_context.hpp>
#include <ie_utils.hpp>
#include <vpu/utils/ie_helpers.hpp>

#include "hddl2_exceptions.h"
#include "hddl2_metrics.h"
#include "hddl2_remote_context.h"
#include "hddl_unite/hddl2_unite_graph.h"

namespace vpux {
namespace HDDL2 {

namespace IE = InferenceEngine;
//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
/**
 * @brief If service is not available, we cannot create any executor
 */
static bool isServiceAvailable(const vpu::Logger::Ptr& logger) {
    if (vpu::HDDL2Plugin::HDDL2Metrics::isServiceAvailable()) {
        logger->debug(SERVICE_AVAILABLE.c_str());
        return true;
    } else {
        logger->debug(SERVICE_NOT_AVAILABLE.c_str());
        return false;
    }
}

static vpu::HDDL2Plugin::HDDL2RemoteContext::Ptr castIEContextToHDDL2(const IE::RemoteContext::Ptr& ieContext) {
    vpu::HDDL2Plugin::HDDL2RemoteContext::Ptr pluginContext = nullptr;

    if (ieContext != nullptr) {
        pluginContext = std::dynamic_pointer_cast<vpu::HDDL2Plugin::HDDL2RemoteContext>(ieContext);
        if (pluginContext == nullptr) {
            THROW_IE_EXCEPTION << FAILED_CAST_CONTEXT;
        }
    }
    return pluginContext;
}

// TODO [Track number: S#21391]
// FIXME: does not work for batch != 1
static bool is2DTensor(const IE::SizeVector& dims) {
    size_t ones = std::count(dims.begin(), dims.end(), 1);
    return (dims.size() - ones) == 1;
}

static void copyDataToBlob(const IE::Blob::Ptr& dest, const void* source, const size_t& size) {
    if (source == nullptr) {
        THROW_IE_EXCEPTION << "Source data is nullptr!";
    }
    if (dest->byteSize() != size) {
        THROW_IE_EXCEPTION << "Output size mismatch between HddlUnite: " << size
                           << " and expected output: " << dest->byteSize();
    }
    IE::MemoryBlob::Ptr mblob = IE::as<IE::MemoryBlob>(dest);
    if (!mblob) {
        THROW_IE_EXCEPTION << "Failed output blob type!";
    }
    auto lockedMemory = mblob->wmap();
    void* data = lockedMemory.as<void*>();
    auto result = ie_memcpy(data, dest->byteSize(), source, size);
    if (result != 0) {
        THROW_IE_EXCEPTION << "Failed to copy memory.";
    }
}

static IE::Blob::Ptr prepareInputForInference(const IE::Blob::Ptr& actualInput, const IE::Layout& expectedLayout) {
    if (actualInput == nullptr) {
        THROW_IE_EXCEPTION << "Actual input blob null pointer!";
    }
    if (actualInput->getTensorDesc().getLayout() == expectedLayout ||
        /** Currently we ignore information of what type of remote blob we are using **/
        actualInput->is<IE::RemoteBlob>() ||
        /** Repacking for NV12 Blob is not required, compound blob should be handled other way **/
        // TODO Add repacking for compound blob case
        actualInput->is<IE::NV12Blob>() || actualInput->is<IE::CompoundBlob>()) {
        return actualInput;
    }

    IE::Blob::Ptr inputForInference;

    if (is2DTensor(actualInput->getTensorDesc().getDims())) {
        auto tensorDims = actualInput->getTensorDesc().getDims();
        for (size_t dimInd = actualInput->getTensorDesc().getDims().size(); dimInd < 4; dimInd++) {
            tensorDims.push_back(1);
        }
        IE::TensorDesc TensorDesc = {actualInput->getTensorDesc().getPrecision(), tensorDims, expectedLayout};
        inputForInference = make_blob_with_precision(TensorDesc);
        inputForInference->allocate();

        ie_memcpy(
            inputForInference->buffer(), inputForInference->byteSize(), actualInput->buffer(), actualInput->byteSize());
    } else {
        inputForInference = toLayout(actualInput, expectedLayout);
    }

    return inputForInference;
}
//------------------------------------------------------------------------------
vpux::HDDL2::HDDL2Executor::Ptr HDDL2Executor::prepareExecutor(const vpux::NetworkDescription::Ptr& networkDesc,
    const vpu::HDDL2Config& config, const InferenceEngine::RemoteContext::Ptr& ieContextPtr) {
    auto logger = std::make_shared<vpu::Logger>("Executor", config.logLevel(), vpu::consoleOutput());
    vpux::HDDL2::HDDL2Executor::Ptr executor = nullptr;
    if (!isServiceAvailable(logger)) {
        logger->warning(EXECUTOR_NOT_CREATED.c_str());
        return nullptr;
    }

    auto context = castIEContextToHDDL2(ieContextPtr);

    try {
        executor = std::make_shared<vpux::HDDL2::HDDL2Executor>(networkDesc, config, context);
    } catch (const IE::details::InferenceEngineException& exception) {
        if (exception.hasStatus() && exception.getStatus() == IE::StatusCode::NETWORK_NOT_LOADED) {
            logger->error(FAILED_LOAD_NETWORK.c_str());
        } else {
            logger->error("%s%s", EXECUTOR_NOT_CREATED.c_str(), std::string("\nERROR: ") + exception.what());
        }
    } catch (const std::exception& exception) {
        logger->error("%s%s", EXECUTOR_NOT_CREATED.c_str(), std::string("\nERROR: ") + exception.what());
    }
    return executor;
}

HDDL2Executor::HDDL2Executor(const vpux::NetworkDescription::CPtr& network, const vpu::HDDL2Config& config,
    vpu::HDDL2Plugin::HDDL2RemoteContext::CPtr context)
    : _network(network),
      _context(context),
      _config(config),
      // TODO Make executor logger name unique
      _logger(std::make_shared<vpu::Logger>("Executor", config.logLevel(), vpu::consoleOutput())) {
    loadGraphToDevice();
}

HDDL2Executor::HDDL2Executor(const HDDL2Executor& ex)
    : _network(ex._network), _context(ex._context), _config(ex._config), _uniteGraphPtr(ex._uniteGraphPtr) {}

void HDDL2Executor::setup(const InferenceEngine::ParamMap& params) {
    UNUSED(params);
    THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
}

void HDDL2Executor::push(const InferenceEngine::BlobMap& inputs) { push(inputs, {}); }

void HDDL2Executor::push(const InferenceEngine::BlobMap& inputs, const PreprocMap& preProcMap) {
    // TODO [Design flaw] InferData need to know if preprocessing required on creation [Track number: S#31308]
    bool needUnitePreProcessing = false;
    IE::BlobMap updatedInputs;

    const auto& networkInputs = _network->getInputsInfo();
    const auto& deviceInputs = _network->getDeviceInputsInfo();

    if (inputs.size() != networkInputs.size()) {
        _logger->warning("Amount of blobs and network inputs mismatch!\n"
                         "Blobs: %d, network inputs: %d",
            inputs.size(), networkInputs.size());
    } else if (networkInputs.size() != deviceInputs.size()) {
        _logger->warning("Amount of network inputs and expected device inputs mismatch!\n"
                         "Network inputs: %d, Device inputs: %d",
            networkInputs.size(), deviceInputs.size());
    }

    for (const auto& networkInput : networkInputs) {
        const std::string inputName = networkInput.first;
        auto foundInputBlob = inputs.find(inputName);
        if (foundInputBlob == inputs.end()) {
            THROW_IE_EXCEPTION << "Error: input [" << inputName << "] is not provided.";
        }

        const IE::Blob::Ptr inputBlobPtr = foundInputBlob->second;
        if (preProcMap.find(inputName) != preProcMap.end()) {
            needUnitePreProcessing = true;
        }
        if (inputBlobPtr->is<vpu::HDDL2Plugin::HDDL2RemoteBlob>()) {
            needUnitePreProcessing |= (inputBlobPtr->as<vpu::HDDL2Plugin::HDDL2RemoteBlob>()->getROIPtr() != nullptr);
        }

        const auto deviceInputLayout = deviceInputs.at(inputName)->getLayout();
        updatedInputs[foundInputBlob->first] = prepareInputForInference(foundInputBlob->second, deviceInputLayout);
    }

    // TODO Create HddlUniteInferData inside constructor of executor [Track number: S#37397]
    std::call_once(_onceFlagInferData, [&] {
        _inferDataPtr = std::make_shared<vpu::HDDL2Plugin::HddlUniteInferData>(
            needUnitePreProcessing, _context, _config.getGraphColorFormat(), _network->getDeviceOutputsInfo().size());
    });

    // TODO Should we use deviceInputs instead of networkInputs here?
    for (const auto& networkInput : networkInputs) {
        const std::string inputName = networkInput.first;
        const IE::DataPtr inputDesc = networkInput.second;

        if (preProcMap.find(inputName) != preProcMap.end()) {
            IE::Blob::CPtr blobRequiredPreProcessing;
            InferenceEngine::PreProcessInfo preProcessInfo = preProcMap.find(inputName)->second;
            // TODO preProcessInfo are not used [Track number: S#37393]
            UNUSED(preProcessInfo);
        }
        auto foundInputBlob = updatedInputs.find(inputName);
        if (foundInputBlob == updatedInputs.end()) {
            THROW_IE_EXCEPTION << "Error: input [" << inputName << "] is not provided.";
        }
        const IE::Blob::Ptr inputBlobPtr = foundInputBlob->second;
        _inferDataPtr->prepareUniteInput(inputBlobPtr, inputDesc);
    }

    /// Use what expected on device instead of what expected on IE side
    const auto& deviceOutputs = _network->getDeviceOutputsInfo();
    for (const auto& deviceOutput : deviceOutputs) {
        _inferDataPtr->prepareUniteOutput(deviceOutput.second);
    }

    _uniteGraphPtr->InferAsync(_inferDataPtr);
}

void HDDL2Executor::pull(InferenceEngine::BlobMap& outputs) {
    _inferDataPtr->waitInferDone();
    const auto& networkOutputs = _network->getOutputsInfo();
    const auto& deviceOutputs = _network->getDeviceOutputsInfo();
    for (const auto& networkOutput : networkOutputs) {
        const std::string outputName = networkOutput.first;
        auto foundOutputBlob = outputs.find(outputName);
        if (foundOutputBlob == outputs.end()) {
            THROW_IE_EXCEPTION << "Error: output [" << outputName << "] is not provided.";
        }
        const auto& deviceOutput = deviceOutputs.find(outputName);
        if (deviceOutput == deviceOutputs.end()) {
            THROW_IE_EXCEPTION << "Error: output [" << outputName << "] information for device not found.";
        }
        IE::Blob::Ptr outputBlobPtr = foundOutputBlob->second;

        const std::string outputUniteData = _inferDataPtr->getOutputData(outputName);

        const auto deviceTensorDesc = deviceOutput->second->getTensorDesc();
        const auto outputBlobTensorDesc = outputBlobPtr->getTensorDesc();

        const auto deviceOutputPrecision = deviceTensorDesc.getPrecision();
        const auto blobOutputPrecision = outputBlobTensorDesc.getPrecision();
        const auto deviceOutputLayout = deviceTensorDesc.getLayout();
        const auto blobOutputLayout = outputBlobTensorDesc.getLayout();

        if (deviceOutputPrecision == IE::Precision::FP32 || blobOutputPrecision == IE::Precision::FP32) {
            if (deviceOutputPrecision == IE::Precision::U8 || blobOutputPrecision == IE::Precision::U8) {
                THROW_IE_EXCEPTION << "Error: output precision conversion from " << deviceOutputPrecision << " to "
                                   << blobOutputPrecision << " is not supported.";
            }
            IE::Blob::Ptr deviceOutputBlob = make_blob_with_precision(deviceTensorDesc);
            deviceOutputBlob->allocate();
            copyDataToBlob(deviceOutputBlob, outputUniteData.data(), outputUniteData.size());
            outputBlobPtr = toPrecision(deviceOutputBlob, blobOutputPrecision);
        } else {
            if (deviceOutputPrecision == IE::Precision::U8 && blobOutputPrecision == IE::Precision::FP16) {
                THROW_IE_EXCEPTION << "Error: output precision conversion from " << deviceOutputPrecision << " to "
                                   << blobOutputPrecision << " is not supported.";
            }
            if (outputUniteData.size() != outputBlobPtr->byteSize()) {
                THROW_IE_EXCEPTION << "Output size mismatch between HddlUnite and network expected output";
            }
            copyDataToBlob(outputBlobPtr, outputUniteData.data(), outputUniteData.size());
        }

        // Currently we have outputBlob with device layout and user precision
        if (blobOutputLayout == deviceOutputLayout) {
            outputs[outputName] = outputBlobPtr;
        } else {
            IE::Blob::Ptr rightOutputBlobPtr = nullptr;
            if (is2DTensor(outputBlobTensorDesc.getDims()) || is2DTensor(deviceTensorDesc.getDims())) {
                rightOutputBlobPtr = make_blob_with_precision(outputBlobTensorDesc);
                rightOutputBlobPtr->allocate();
                std::memcpy(IE::as<IE::MemoryBlob>(rightOutputBlobPtr)->wmap().as<uint8_t*>(),
                    IE::as<IE::MemoryBlob>(outputBlobPtr)->rmap().as<uint8_t*>(), outputBlobPtr->byteSize());
            } else {
                // Both of them are not 2D
                if (outputBlobTensorDesc.getDims().size() == 3) {
                    // 3D output - icnet
                    IE::Blob::Ptr tmpBlobPtr = toLayout(
                        outputBlobPtr, blobOutputLayout == IE::Layout::CHW ? IE::Layout::NCHW : IE::Layout::NHWC);
                    rightOutputBlobPtr = make_blob_with_precision(outputBlobTensorDesc);
                    rightOutputBlobPtr->allocate();
                    std::memcpy(IE::as<IE::MemoryBlob>(rightOutputBlobPtr)->wmap().as<uint8_t*>(),
                        IE::as<IE::MemoryBlob>(tmpBlobPtr)->rmap().as<uint8_t*>(), tmpBlobPtr->byteSize());
                } else {
                    // 4D to 4D
                    rightOutputBlobPtr = toLayout(outputBlobPtr, blobOutputLayout);
                }
            }
            outputs[outputName] = rightOutputBlobPtr;
        }
    }
}  // namespace HDDL2

bool HDDL2Executor::isPreProcessingSupported(const InferenceEngine::PreProcessInfo& preProcessInfo) const {
    UNUSED(preProcessInfo);
    THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
}

std::map<std::string, IE::InferenceEngineProfileInfo> HDDL2Executor::getLayerStatistics() {
    return _inferDataPtr->getHDDLUnitePerfCounters();
}

InferenceEngine::Parameter HDDL2Executor::getParameter(const std::string& paramName) const {
    UNUSED(paramName);
    THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
}

void HDDL2Executor::loadGraphToDevice() {
    if (_context == nullptr) {
        _uniteGraphPtr =
            std::make_shared<vpu::HDDL2Plugin::HddlUniteGraph>(_network, _config.device_id(), _config.logLevel());
    } else {
        _uniteGraphPtr = std::make_shared<vpu::HDDL2Plugin::HddlUniteGraph>(_network, _context, _config.logLevel());
    }
}

Executor::Ptr HDDL2Executor::clone() const { return std::make_shared<HDDL2Executor>(*this); }

}  // namespace HDDL2
}  // namespace vpux
