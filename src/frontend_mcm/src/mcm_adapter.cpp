//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "mcm_adapter.hpp"

#include "vpux/al/config/common.hpp"
#include "vpux/al/config/mcm_compiler.hpp"
#include "vpux/utils/core/logger.hpp"

#include <file_utils.h>
#include <sys/stat.h>

#include <precision_utils.h>
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
using namespace vpux;

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
    const auto nonTrivialZeroPoint =
            quantParams.getZeroPoint().size() >= 1 && quantParams.getZeroPoint()[0] != defaultZeroPoint;
    const auto nonTrivialScale =
            quantParams.getScale().size() >= 1 && fabs(quantParams.getScale()[0] - defaultScale) > epsilon;
    const auto nonTrivialQuantParam = nonTrivialZeroPoint || nonTrivialScale;
    const auto isPluginInputQuantization = forcePluginInputQuantization ? nonTrivialQuantParam : false;
    // Plugin input quantization flag
    // Consider to use existence of quant_mult and quant_zero parameters as a flag of plugin quantization mode
    toBuild->quant_zero.clear();
    toBuild->quant_mult.clear();
    if (isPluginInputQuantization) {
        // Zero point
        const float minU8 = static_cast<float>(std::numeric_limits<uint8_t>().lowest());
        const float maxU8 = static_cast<float>(std::numeric_limits<uint8_t>().max());
        const int64_t zpValue = nonTrivialZeroPoint ? quantParams.getZeroPoint()[0] : defaultZeroPoint;
        const uint8_t zeroPoint = static_cast<uint8_t>(zpValue < minU8 ? minU8 : (zpValue > maxU8 ? maxU8 : zpValue));
        toBuild->quant_zero = {zeroPoint};
        // Scale value
        // Consider to use quant_mult parameter as a scale value (fp16)
        const float scaleFP32 = static_cast<float>(nonTrivialScale ? quantParams.getScale()[0] : defaultScale);
        const auto scaleFP16 = PrecisionUtils::f32tof16(scaleFP32);
        toBuild->quant_mult = {static_cast<uint16_t>(scaleFP16)};
    }

    return toBuild;
}

std::unique_ptr<MVCNN::TensorReferenceT> buildTensorReference(const std::string& tensorName,
                                                              const InferenceEngine::TensorDesc& tensorInfo,
                                                              const mv::Data::TensorIterator& opModelTensor,
                                                              const Config& config) {
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

    return buildTensorReference(tensorName, newTensorInfo, quantParams,
                                config.get<MCM_FORCE_PLUGIN_INPUT_QUANTIZATION>());
}

bool vpu::MCMAdapter::isMCMCompilerAvailable() {
    return true;
}

vpu::MCMAdapter::MetaInfo vpu::MCMAdapter::deserializeMetaData(const MVCNN::SummaryHeader& header,
                                                               const Config& config) {
    IE_ASSERT(header.identifier() != nullptr);
    IE_ASSERT(header.in_tensor_desc() != nullptr);
    IE_ASSERT(header.out_tensor_desc() != nullptr);

    vpux::Logger logger("compileMCM", config.get<LOG_LEVEL>());

    const std::string& resultNetworkName = header.identifier()->str();
    logger.debug("networkName: {0}", resultNetworkName);

    vpux::QuantizationParamMap resultQuantParamMap;
    InferenceEngine::InputsDataMap resultNetworkInputs;
    const auto& inputTensorDesc = *header.in_tensor_desc();
    size_t inputTensorsCount = inputTensorDesc.size();
    logger.debug("inputTensorsCount: {0}", inputTensorsCount);
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
        logger.debug("input info: {0}", inputSerializer.str());

        InferenceEngine::InputInfo inputInfo;
        inputInfo.setInputData(std::make_shared<InferenceEngine::Data>(inputData));
        resultNetworkInputs[inputInfo.name()] = std::make_shared<InferenceEngine::InputInfo>(inputInfo);
        const auto isQuantZeroDefined = tensorRef->quant_zero() && tensorRef->quant_zero()->size() == 1;
        const auto isQuantScaleDefined = tensorRef->quant_mult() && tensorRef->quant_mult()->size() == 1;
        const auto pluginQuantization = isQuantZeroDefined && isQuantScaleDefined;
        vpux::Optional<vpux::QuantizationParam> quantParam{vpux::None};
        if (pluginQuantization) {
            quantParam = vpux::QuantizationParam{};
            const ie_fp16 scaleFP16 = static_cast<ie_fp16>(*tensorRef->quant_mult()->cbegin());
            const float scaleFP32 = PrecisionUtils::f16tof32(scaleFP16);
            IE_ASSERT(scaleFP32 != 0.f);
            quantParam.getValue()._reverseScale = 1.f / scaleFP32;
            quantParam.getValue()._zeroPoint = *tensorRef->quant_zero()->cbegin();
        }
        resultQuantParamMap.insert({inputInfo.name(), quantParam});
    }

    InferenceEngine::OutputsDataMap resultNetworkOutputs;
    const auto& outputTensorDesc = *header.out_tensor_desc();
    size_t outputTensorsCount = outputTensorDesc.size();
    logger.debug("outputTensorsCount: {0}", outputTensorsCount);
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
        logger.debug("output info: {0}", outputSerializer.str());

        InferenceEngine::TensorDesc outputDesc(iePrecision, dimVec, ieLayout);
        InferenceEngine::Data outputData(tensorRef->name()->str(), outputDesc);
        resultNetworkOutputs[outputData.getName()] = std::make_shared<InferenceEngine::Data>(outputData);
    }

    return {resultNetworkName, resultNetworkInputs, resultNetworkOutputs, resultQuantParamMap};
}
