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
#include <schema/graphfile/graphfile_generated.h>

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

bool vpu::MCMAdapter::isMCMCompilerAvailable() {
    return true;
}

vpu::MCMAdapter::MetaInfo vpu::MCMAdapter::deserializeMetaData(const std::vector<char>& outBlob,
                                                               const MCMConfig& config) {
    Logger::Ptr logger = std::make_shared<Logger>("compileMCM", config.logLevel(), consoleOutput());
    if (logger == nullptr) {
        IE_THROW() << "Logger has not been created";
    }
    const MVCNN::GraphFile* graphFilePtr = MVCNN::GetGraphFile(outBlob.data());
    MVCNN::GraphFileT graphFileInstance;
    graphFilePtr->UnPackTo(&graphFileInstance);

    const std::string& networkName = graphFileInstance.header->identifier;
    logger->debug("networkName: %s", networkName);

    InferenceEngine::InputsDataMap resultNetworkInputs;
    size_t inputTensorsCount = graphFileInstance.header->in_tensor_desc.size();
    logger->debug("inputTensorsCount: %d", inputTensorsCount);
    for (size_t inputIdx = 0; inputIdx < inputTensorsCount; inputIdx++) {
        std::unique_ptr<MVCNN::TensorReferenceT>& tensorRef = graphFileInstance.header->in_tensor_desc.at(inputIdx);
        std::ostringstream inputSerializer;
        inputSerializer << "Name: " << tensorRef->name << std::endl;
        InferenceEngine::SizeVector dimVec;
        std::copy(tensorRef->dimensions.begin(), tensorRef->dimensions.end(), std::back_inserter(dimVec));
        inputSerializer << "Dims: {";
        for (const size_t& dim : dimVec) {
            inputSerializer << " " << dim << " ";
        }
        inputSerializer << "}" << std::endl;
        InferenceEngine::Layout ieLayout = orderVectorToLayout(tensorRef->strides);
        InferenceEngine::Precision iePrecision = MvcnnDTypeToPrecision(tensorRef->data_dtype);
        inputSerializer << "Layout: " << ieLayout << std::endl;
        inputSerializer << "Precision: " << iePrecision << std::endl;

        InferenceEngine::TensorDesc inputDesc(iePrecision, dimVec, ieLayout);
        InferenceEngine::Data inputData(tensorRef->name, inputDesc);
        logger->debug("input info:\n%s\n", inputSerializer.str());

        InferenceEngine::InputInfo inputInfo;
        inputInfo.setInputData(std::make_shared<InferenceEngine::Data>(inputData));
        resultNetworkInputs[inputInfo.name()] = std::make_shared<InferenceEngine::InputInfo>(inputInfo);
    }

    InferenceEngine::OutputsDataMap resultNetworkOutputs;
    size_t outputTensorsCount = graphFileInstance.header->out_tensor_desc.size();
    logger->debug("outputTensorsCount: %d", outputTensorsCount);
    for (size_t outputIdx = 0; outputIdx < outputTensorsCount; outputIdx++) {
        std::unique_ptr<MVCNN::TensorReferenceT>& tensorRef = graphFileInstance.header->out_tensor_desc.at(outputIdx);
        std::ostringstream outputSerializer;
        outputSerializer << "Name: " << tensorRef->name << std::endl;
        InferenceEngine::SizeVector dimVec;
        std::copy(tensorRef->dimensions.begin(), tensorRef->dimensions.end(), std::back_inserter(dimVec));
        outputSerializer << "Dims: {";
        for (const size_t& dim : dimVec) {
            outputSerializer << " " << dim << " ";
        }
        outputSerializer << "}" << std::endl;
        InferenceEngine::Layout ieLayout = orderVectorToLayout(tensorRef->strides);
        InferenceEngine::Precision iePrecision = MvcnnDTypeToPrecision(tensorRef->data_dtype);
        outputSerializer << "Layout: " << ieLayout << std::endl;
        outputSerializer << "Precision: " << iePrecision << std::endl;
        logger->debug("output info:\n%s\n", outputSerializer.str());

        InferenceEngine::TensorDesc outputDesc(iePrecision, dimVec, ieLayout);
        InferenceEngine::Data outputData(tensorRef->name, outputDesc);
        resultNetworkOutputs[outputData.getName()] = std::make_shared<InferenceEngine::Data>(outputData);
    }

    return {networkName, resultNetworkInputs, resultNetworkOutputs};
}
