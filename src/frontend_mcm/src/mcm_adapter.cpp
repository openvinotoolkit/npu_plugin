//
// Copyright 2019-2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "mcm_adapter.hpp"

#include <file_utils.h>
#include <sys/stat.h>

#include <ie_icnn_network.hpp>

#if defined(_WIN32)
#include <direct.h>
#define mkdir(dir, mode) _mkdir(dir)
#endif

#include <flatbuffers/flatbuffers.h>

#include <include/mcm/compiler/compilation_unit.hpp>

#include "converters.hpp"
#include "ie_memcpy.h"

using namespace InferenceEngine;
using namespace vpu;

std::unique_ptr<MVCNN::TensorReferenceT> buildTensorReference(const std::string& tensorName,
                                                              const InferenceEngine::TensorDesc& tensorInfo) {
    std::unique_ptr<MVCNN::TensorReferenceT> toBuild =
            std::unique_ptr<MVCNN::TensorReferenceT>(new MVCNN::TensorReferenceT());
    toBuild->name = tensorName;
    const InferenceEngine::SizeVector& dimVec = tensorInfo.getDims();
    for (const size_t& dim : dimVec) {
        toBuild->dimensions.push_back(dim);
    }
    toBuild->strides = layoutToOrderVector(tensorInfo.getLayout());
    toBuild->data_dtype = precisionToMvcnnDType(tensorInfo.getPrecision());
    toBuild->data = nullptr;

    return toBuild;
}

std::unique_ptr<MVCNN::TensorReferenceT> buildTensorReference(const std::string& tensorName,
                                                              const InferenceEngine::TensorDesc& tensorInfo,
                                                              const mv::Data::TensorIterator& opModelTensor) {
    InferenceEngine::TensorDesc newTensorInfo(tensorInfo);

    const InferenceEngine::SizeVector& tensorInfoDimVec = tensorInfo.getDims();
    auto shape = opModelTensor->getShape();

    // For now support case in which dim reduction from 5D to 4D has happened (yolo-v5 cases)
    if (tensorInfoDimVec.size() == 5 && shape.ndims() == 4) {
        // Try to update tensor information in case of 5D to 4D dimension reduction
        InferenceEngine::SizeVector newDimVec;
        for (size_t i = 0; i < shape.ndims(); i++)
            newDimVec.push_back(shape[shape.ndims() - 1 - i]);

        // Update layout and dimensions of vector.
        // Note: current solution has limited implementation to satisfy just yolo-v5 case
        if (newTensorInfo.getLayout() == Layout::NCDHW)
            newTensorInfo.reshape(newDimVec, Layout::NCHW);
        else if (newTensorInfo.getLayout() == Layout::NDHWC)
            newTensorInfo.reshape(newDimVec, Layout::NHWC);
    }

    return buildTensorReference(tensorName, newTensorInfo);
}

bool vpu::MCMAdapter::isMCMCompilerAvailable() {
    return true;
}

vpu::MCMAdapter::MetaInfo vpu::MCMAdapter::deserializeMetaData(const MVCNN::SummaryHeader& header,
                                                               const MCMConfig& config) {
    IE_ASSERT(header.identifier() != nullptr);
    IE_ASSERT(header.in_tensor_desc() != nullptr);
    IE_ASSERT(header.out_tensor_desc() != nullptr);
    Logger::Ptr logger = std::make_shared<Logger>("compileMCM", config.logLevel(), consoleOutput());
    if (logger == nullptr) {
        IE_THROW() << "Logger has not been created";
    }

    const std::string& resultNetworkName = header.identifier()->str();
    logger->debug("networkName: %s", resultNetworkName);

    InferenceEngine::InputsDataMap resultNetworkInputs;
    const auto& inputTensorDesc = *header.in_tensor_desc();
    size_t inputTensorsCount = inputTensorDesc.size();
    logger->debug("inputTensorsCount: %d", inputTensorsCount);
    for (size_t inputIdx = 0; inputIdx < inputTensorsCount; inputIdx++) {
        const auto tensorRef = inputTensorDesc[inputIdx];
        IE_ASSERT(tensorRef != nullptr);
        IE_ASSERT(tensorRef->strides() != nullptr);
        std::ostringstream inputSerializer;
        inputSerializer << "Name: " << tensorRef->name()->str() << std::endl;
        InferenceEngine::SizeVector dimVec;
        std::copy(tensorRef->dimensions()->cbegin(), tensorRef->dimensions()->cend(), std::back_inserter(dimVec));
        inputSerializer << "Dims: {";
        for (const size_t& dim : dimVec) {
            inputSerializer << " " << dim << " ";
        }
        inputSerializer << "}" << std::endl;
        std::vector<float> inputTensorOrder;
        std::copy(tensorRef->strides()->cbegin(), tensorRef->strides()->cend(), std::back_inserter(inputTensorOrder));
        const auto ieLayout = orderVectorToLayout(inputTensorOrder);
        const auto iePrecision = MvcnnDTypeToPrecision(tensorRef->data_dtype());
        inputSerializer << "Layout: " << ieLayout << std::endl;
        inputSerializer << "Precision: " << iePrecision << std::endl;

        InferenceEngine::TensorDesc inputDesc(iePrecision, dimVec, ieLayout);
        InferenceEngine::Data inputData(tensorRef->name()->str(), inputDesc);
        logger->debug("input info:\n%s\n", inputSerializer.str());

        InferenceEngine::InputInfo inputInfo;
        inputInfo.setInputData(std::make_shared<InferenceEngine::Data>(inputData));
        resultNetworkInputs[inputInfo.name()] = std::make_shared<InferenceEngine::InputInfo>(inputInfo);
    }

    InferenceEngine::OutputsDataMap resultNetworkOutputs;
    const auto& outputTensorDesc = *header.out_tensor_desc();
    size_t outputTensorsCount = outputTensorDesc.size();
    logger->debug("outputTensorsCount: %d", outputTensorsCount);
    for (size_t outputIdx = 0; outputIdx < outputTensorsCount; outputIdx++) {
        const auto tensorRef = outputTensorDesc[outputIdx];
        IE_ASSERT(tensorRef != nullptr);
        IE_ASSERT(tensorRef->strides() != nullptr);
        std::ostringstream outputSerializer;
        outputSerializer << "Name: " << tensorRef->name()->str() << std::endl;
        InferenceEngine::SizeVector dimVec;
        std::copy(tensorRef->dimensions()->cbegin(), tensorRef->dimensions()->cend(), std::back_inserter(dimVec));
        outputSerializer << "Dims: {";
        for (const size_t& dim : dimVec) {
            outputSerializer << " " << dim << " ";
        }
        outputSerializer << "}" << std::endl;
        std::vector<float> outputTensorOrder;
        std::copy(tensorRef->strides()->cbegin(), tensorRef->strides()->cend(), std::back_inserter(outputTensorOrder));
        const auto ieLayout = orderVectorToLayout(outputTensorOrder);
        const auto iePrecision = MvcnnDTypeToPrecision(tensorRef->data_dtype());
        outputSerializer << "Layout: " << ieLayout << std::endl;
        outputSerializer << "Precision: " << iePrecision << std::endl;
        logger->debug("output info:\n%s\n", outputSerializer.str());

        InferenceEngine::TensorDesc outputDesc(iePrecision, dimVec, ieLayout);
        InferenceEngine::Data outputData(tensorRef->name()->str(), outputDesc);
        resultNetworkOutputs[outputData.getName()] = std::make_shared<InferenceEngine::Data>(outputData);
    }

    return {resultNetworkName, resultNetworkInputs, resultNetworkOutputs};
}
