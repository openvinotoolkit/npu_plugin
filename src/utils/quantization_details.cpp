// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "quantization_details.hpp"

#include "network_helper.hpp"

using namespace vpu::details;

QuantizationDetails::QuantizationDetails()
        : levels(),
          inputLowValues({}),
          inputHighValues({}),
          outputLowValues({}),
          outputHighValues({}),
          inputIntervalsCount(0),
          outputIntervalsCount(0),
          outputChannelsCount(0) {
}

QuantizationDetails::QuantizationDetails(const QuantizationDetails& quantizationDetails)
        : levels(quantizationDetails.levels),
          inputLowValues(quantizationDetails.inputLowValues),
          inputHighValues(quantizationDetails.inputHighValues),
          outputLowValues(quantizationDetails.outputLowValues),
          outputHighValues(quantizationDetails.outputHighValues),
          inputIntervalsCount(quantizationDetails.inputIntervalsCount),
          outputIntervalsCount(quantizationDetails.outputIntervalsCount),
          outputChannelsCount(quantizationDetails.outputChannelsCount) {
}

QuantizationDetails::QuantizationDetails(const size_t levels, const std::vector<float>& inputLowValues,
                                         const std::vector<float>& inputHighValues,
                                         const std::vector<float>& outputLowValues,
                                         const std::vector<float>& outputHighValues, const size_t inputIntervalsCount,
                                         const size_t outputIntervalsCount, const size_t outputChannelsCount)
        : levels(levels),
          inputLowValues(inputLowValues),
          inputHighValues(inputHighValues),
          outputLowValues(outputLowValues),
          outputHighValues(outputHighValues),
          inputIntervalsCount(inputIntervalsCount),
          outputIntervalsCount(outputIntervalsCount),
          outputChannelsCount(outputChannelsCount) {
}

IE_SUPPRESS_DEPRECATED_START
bool QuantizationDetails::outputLayoutIsSupported(const ie::CNNLayer& quantize) {
    std::vector<float> outputLowValues;
    std::vector<float> outputHighValues;
    size_t outputIntervalsCount;
    getOutputIntervals(quantize, outputLowValues, outputHighValues, outputIntervalsCount);

    const size_t outputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(
            quantize, CNNNetworkHelper::onWeights(quantize) && CNNNetworkHelper::onConstWeightsPath(quantize));
    if ((outputIntervalsCount != 1ul) && (outputIntervalsCount != outputChannelsCount)) {
        return false;
    }

    return true;
}

void QuantizationDetails::getInputIntervals(const ie::CNNLayer& quantize, std::vector<float>& inputLowValues,
                                            std::vector<float>& inputHighValues, size_t& inputIntervalsCount) {
    if (quantize.insData.size() != 5) {
        THROW_VPU_LPT_EXCEPTION(quantize) << "Unexpected inputs size " << quantize.insData.size();
    }

    const ie::DataPtr inputLowData = quantize.insData[1].lock();
    if (inputLowData == nullptr) {
        THROW_VPU_LPT_EXCEPTION(quantize) << "input low data is absent";
    }
    const ie::CNNLayerPtr inputLowLayer = getCreatorLayer(inputLowData).lock();
    validate(inputLowLayer);
    const std::vector<float> inputLowBlobValues = getBlobValue(inputLowLayer);
    inputLowValues.insert(inputLowValues.end(), inputLowBlobValues.begin(), inputLowBlobValues.end());

    const ie::DataPtr inputHighData = quantize.insData[2].lock();
    if (inputHighData == nullptr) {
        THROW_VPU_LPT_EXCEPTION(quantize) << "input high data is absent";
    }
    const ie::CNNLayerPtr inputHighLayer = getCreatorLayer(inputHighData).lock();
    validate(inputHighLayer);
    const std::vector<float> inputHighBlobValues = getBlobValue(inputHighLayer);
    inputHighValues.insert(inputHighValues.end(), inputHighBlobValues.begin(), inputHighBlobValues.end());

    if (inputLowValues.size() != inputHighValues.size()) {
        THROW_VPU_LPT_EXCEPTION(quantize) << "Quantize input values sizes are not equal for layer " << quantize.name;
    }

    inputIntervalsCount = inputLowValues.size();
}

void QuantizationDetails::getOutputIntervals(const ie::CNNLayer& quantize, std::vector<float>& outputLowValues,
                                             std::vector<float>& outputHighValues, size_t& outputIntervalsCount) {
    if (quantize.insData.size() != 5) {
        THROW_VPU_LPT_EXCEPTION(quantize) << "unexpected inputs size " << quantize.insData.size();
    }

    const ie::DataPtr outputLowData = quantize.insData[3].lock();
    if (outputLowData == nullptr) {
        THROW_VPU_LPT_EXCEPTION(quantize) << "output low data is absent";
    }
    const ie::CNNLayerPtr outputLowLayer = getCreatorLayer(outputLowData).lock();
    validate(outputLowLayer);
    const std::vector<float>& outputLowBlobValues = getBlobValue(outputLowLayer);
    outputLowValues.insert(outputLowValues.end(), outputLowBlobValues.begin(), outputLowBlobValues.end());

    const ie::DataPtr outputHighData = quantize.insData[4].lock();
    if (outputHighData == nullptr) {
        THROW_VPU_LPT_EXCEPTION(quantize) << "output high data is absent";
    }
    const ie::CNNLayerPtr outputHighLayer = getCreatorLayer(outputHighData).lock();
    validate(outputHighLayer);
    const std::vector<float> outputHighBlobValues = getBlobValue(outputHighLayer);
    outputHighValues.insert(outputHighValues.end(), outputHighBlobValues.begin(), outputHighBlobValues.end());

    if (outputLowValues.size() != outputHighValues.size()) {
        THROW_VPU_LPT_EXCEPTION(quantize) << "Quantize output values sizes are not equal for layer " << quantize.name;
    }

    outputIntervalsCount = outputLowValues.size();
}

QuantizationDetails QuantizationDetails::getDetails(const ie::CNNLayer& quantize) {
    std::vector<float> inputLowValues;
    std::vector<float> inputHighValues;
    size_t inputIntervalsCount;
    getInputIntervals(quantize, inputLowValues, inputHighValues, inputIntervalsCount);

    std::vector<float> outputLowValues;
    std::vector<float> outputHighValues;
    size_t outputIntervalsCount;
    getOutputIntervals(quantize, outputLowValues, outputHighValues, outputIntervalsCount);

    const size_t outputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(
            quantize, CNNNetworkHelper::onWeights(quantize) && CNNNetworkHelper::onConstWeightsPath(quantize));
    if (!outputLayoutIsSupported(quantize)) {
        THROW_VPU_LPT_EXCEPTION(quantize)
                << "Expected output channels count " << outputIntervalsCount << " but found " << outputChannelsCount;
    }

    if (!quantize.CheckParamPresence("levels")) {
        THROW_VPU_LPT_EXCEPTION(quantize) << "Parameter 'levels' is absent";
    }

    return QuantizationDetails(quantize.GetParamAsInt("levels"), inputLowValues, inputHighValues, outputLowValues,
                               outputHighValues, inputIntervalsCount, outputIntervalsCount, outputChannelsCount);
}

void QuantizationDetails::validate(const ie::CNNLayerPtr& constantLayer) {
    if (constantLayer == nullptr) {
        THROW_IE_EXCEPTION << "Quantize layer input is absent";
    }

    if (constantLayer->blobs.size() == 0) {
        THROW_IE_EXCEPTION << "Quantize layer input '" << constantLayer->name << "' doesn't have blobs";
    }

    if (constantLayer->blobs.size() > 1) {
        THROW_IE_EXCEPTION << "Quantize layer input '" << constantLayer->name << "' has too much blobs";
    }

    const auto blob = constantLayer->blobs.begin()->second;
    // const auto byteSize = blob->byteSize();
    // if ((blob->getTensorDesc().getDims().size() != 0) &&
    //     (blob->getTensorDesc().getDims().size() != 1) &&
    //     (blob->getTensorDesc().getDims().size() != 4)) {
    //     THROW_IE_EXCEPTION << "Quantize layer input '" << constantLayer->name << "' blob dimensions are not correct";
    // }

    const auto tensorDesc = blob->getTensorDesc();
    // if ((tensorDesc.getLayout() != Layout::SCALAR) &&
    //     (tensorDesc.getLayout() != Layout::C) &&
    //     ((tensorDesc.getLayout() != Layout::NCHW))) {
    //     THROW_IE_EXCEPTION << "Quantize layer input '" << constantLayer->name << "' layout not correct";
    // }

    // const auto dims = tensorDesc.getDims();
    // if ((dims.size() != 0) && (dims.size() != 1) && (dims.size() != 4)) {
    //     THROW_IE_EXCEPTION << "Quantize layer input '" << constantLayer->name << "' blob dimensions size " <<
    //     dims.size() << " not correct";
    // }

    // ConstTensorDesc::validate(tensorDesc.getLayout(), tensorDesc.getDims());
}

std::vector<float> QuantizationDetails::getBlobValue(const ie::CNNLayerPtr& constantLayer) {
    if (constantLayer->blobs.empty()) {
        THROW_VPU_LPT_EXCEPTION((*constantLayer)) << "blobs are empty";
    }
    const auto blob = constantLayer->blobs.begin()->second;
    auto buffer = CNNNetworkHelper::getFloatData(blob);
    return std::vector<float>(buffer.get(), buffer.get() + blob->size());
}
