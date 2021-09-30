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
                                                              const mv::QuantizationParams& quantParams,
                                                              const bool forcePluginInputQuantization) {
    auto mainData = buildTensorReference(tensorName, tensorInfo);
    std::unique_ptr<MVCNN::TensorReferenceT> toBuild = buildTensorReference(tensorName, tensorInfo);
    const auto epsilon = std::numeric_limits<double>::epsilon();
    const int64_t defaultZeroPoint = 0L;
    const double defaultScale = 1.;
    const auto isPluginInputQuantization =
            forcePluginInputQuantization
                    ? ((quantParams.getZeroPoint().size() >= 1 &&
                        (quantParams.getZeroPoint()[0] != defaultZeroPoint)) ||
                       (quantParams.getScale().size() >= 1 && fabs(quantParams.getScale()[0] - defaultScale) > epsilon))
                    : false;
    // Plugin input quantization flag
    // Consider to use quant_mult parameter as a flag
    toBuild->quant_mult.push_back(static_cast<uint16_t>(isPluginInputQuantization));
    if (isPluginInputQuantization) {
        // Zero point
        const float minU8 = static_cast<float>(std::numeric_limits<uint8_t>().lowest());
        const float maxU8 = static_cast<float>(std::numeric_limits<uint8_t>().max());
        const int64_t zpValue = quantParams.getZeroPoint()[0];
        const uint8_t zeroPoint = static_cast<uint8_t>(zpValue < minU8 ? minU8 : (zpValue > maxU8 ? maxU8 : zpValue));
        toBuild->quant_zero = {zeroPoint};
        // Scale value
        // Consider to use quant_shift parameter as a scale value (fp32)
        std::vector<uint8_t> floatStorage(sizeof(float) / sizeof(uint8_t));
        float* floatValue = reinterpret_cast<float*>(floatStorage.data());
        const float scale = quantParams.getScale().size() >= 1 ? static_cast<float>(quantParams.getScale()[0]) : 1.f;
        *floatValue = scale;
        toBuild->quant_shift = floatStorage;
    }

    return toBuild;
}

std::unique_ptr<MVCNN::TensorReferenceT> buildTensorReference(const std::string& tensorName,
                                                              const InferenceEngine::TensorDesc& tensorInfo,
                                                              const mv::Data::TensorIterator& opModelTensor,
                                                              const MCMConfig& config) {
    InferenceEngine::TensorDesc newTensorInfo(tensorInfo);

    const InferenceEngine::SizeVector& tensorInfoDimVec = tensorInfo.getDims();
    const auto shape = opModelTensor->getShape();
    const auto quantParams = opModelTensor->getQuantParams();

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

    return buildTensorReference(tensorName, newTensorInfo, quantParams, config.forcePluginInputQuantization());
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

    vpux::QuantizationParamMap resultQuantParamMap;
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
        const InferenceEngine::Layout ieLayout = getLayout(tensorRef);
        const auto iePrecision = MvcnnDTypeToPrecision(tensorRef->data_dtype());
        inputSerializer << "Layout: " << ieLayout << std::endl;
        inputSerializer << "Precision: " << iePrecision << std::endl;

        InferenceEngine::TensorDesc inputDesc(iePrecision, dimVec, ieLayout);
        InferenceEngine::Data inputData(tensorRef->name()->str(), inputDesc);
        logger->debug("input info:\n%s\n", inputSerializer.str());

        InferenceEngine::InputInfo inputInfo;
        inputInfo.setInputData(std::make_shared<InferenceEngine::Data>(inputData));
        resultNetworkInputs[inputInfo.name()] = std::make_shared<InferenceEngine::InputInfo>(inputInfo);
        const auto isQuantFlagDefined = tensorRef->quant_mult() != nullptr && tensorRef->quant_mult()->size() == 1;
        const auto pluginQuantization = isQuantFlagDefined && static_cast<bool>(*tensorRef->quant_mult()->cbegin());
        vpux::QuantizationParam quantParam{pluginQuantization};
        if (pluginQuantization) {
            const auto floatPackedSize = sizeof(float) / sizeof(uint8_t);
            IE_ASSERT(tensorRef->quant_shift() != nullptr);
            IE_ASSERT(tensorRef->quant_shift()->size() == floatPackedSize);
            IE_ASSERT(tensorRef->quant_zero() != nullptr);
            IE_ASSERT(tensorRef->quant_zero()->size() == 1);
            const auto scale = *(reinterpret_cast<const float*>(tensorRef->quant_shift()->data()));
            IE_ASSERT(scale != 0.f);
            quantParam._scale = 1.f / scale;
            quantParam._zeroPoint = *tensorRef->quant_zero()->cbegin();
        }
        resultQuantParamMap.insert({inputInfo.name(), quantParam});
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
        const InferenceEngine::Layout ieLayout = getLayout(tensorRef);
        const auto iePrecision = MvcnnDTypeToPrecision(tensorRef->data_dtype());
        outputSerializer << "Layout: " << ieLayout << std::endl;
        outputSerializer << "Precision: " << iePrecision << std::endl;
        logger->debug("output info:\n%s\n", outputSerializer.str());

        InferenceEngine::TensorDesc outputDesc(iePrecision, dimVec, ieLayout);
        InferenceEngine::Data outputData(tensorRef->name()->str(), outputDesc);
        resultNetworkOutputs[outputData.getName()] = std::make_shared<InferenceEngine::Data>(outputData);
    }

    return {resultNetworkName, resultNetworkInputs, resultNetworkOutputs, resultQuantParamMap};
}
