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

#include <precision_utils.h>

#include <algorithm>
#include <blob_factory.hpp>
#include <frontend_mcm.hpp>
#include <graph_tools.hpp>
#include <ie_itt.hpp>
#include <ie_util_internal.hpp>
#include <limits>
#include <low_precision_transformations/network_helper.hpp>
#include <low_precision_transformations/transformer.hpp>
#include <memory>
#include <parse_layers_helpers.hpp>
#include <quantization_helpers.hpp>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <vpu/utils/error.hpp>

#include "dims_parser.hpp"
#include "ie_macro.hpp"

#ifdef ENABLE_MCM_COMPILER

#include <converters.hpp>
#include <custom_layer/custom_parser.hpp>
#include <include/mcm/tensor/tiling.hpp>

using namespace InferenceEngine;
using namespace InferenceEngine::details;
namespace vpu {

namespace {

typedef void (FrontEndMcm::*parser_t)(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);

// clang-format off

        ie::details::caseless_map<std::string, parser_t> g_mcm_parsers = {
                {"Convolution",        &FrontEndMcm::parseConvolution},
                {"Pooling",            &FrontEndMcm::parsePooling},
                {"ReLU",               &FrontEndMcm::parseReLU},
                {"Clamp",              &FrontEndMcm::parseClamp},
                {"FullyConnected",     &FrontEndMcm::parseFullyConnected},
                {"SoftMax",            &FrontEndMcm::parseSoftMax},
                {"GRN",                &FrontEndMcm::parseGRN},
                {"MVN",                &FrontEndMcm::parseMVN},
                {"Norm",               &FrontEndMcm::parseNorm},
                {"Concat",             &FrontEndMcm::parseConcat},
                {"Eltwise",            &FrontEndMcm::parseEltwise},
                {"Split",              &FrontEndMcm::parseSplit},
                {"Sigmoid",            &FrontEndMcm::parseSigmoid},
                {"TanH",               &FrontEndMcm::parseTanH},
                {"PReLU",              &FrontEndMcm::parsePReLU},
                {"Bias",               &FrontEndMcm::parseBias},
                // Caffe Slice is transformed to Split by IE
                {"Slice",              &FrontEndMcm::parseSplit},
                {"BatchNormalization", &FrontEndMcm::parseBatchNorm},
                {"ScaleShift",         &FrontEndMcm::parseScale},
                {"Deconvolution",      &FrontEndMcm::parseDeconvolution},
                {"Power",              &FrontEndMcm::parsePower},
                {"Copy",               &FrontEndMcm::parseCopy},
                {"Reshape",            &FrontEndMcm::parseReshape},
                {"Squeeze",            &FrontEndMcm::parseReshape},
                {"Unsqueeze",          &FrontEndMcm::parseReshape},
                {"ELU",                &FrontEndMcm::parseELU},
                // Flatten is represented as Reshape in KMB model
                {"Flatten",            &FrontEndMcm::parseReshape},
                {"Crop",               &FrontEndMcm::parseCrop},
                {"Tile",               &FrontEndMcm::parseTile},
                {"Normalize",          &FrontEndMcm::parseNormalize},
                {"PriorBox",           &FrontEndMcm::parsePriorBox},
                {"PriorBoxClustered",  &FrontEndMcm::parsePriorBoxClustered},
                {"Permute",            &FrontEndMcm::parsePermute},
                {"DetectionOutput",    &FrontEndMcm::parseDetectionOutput},
                {"RegionYolo",         &FrontEndMcm::parseRegionYolo},
                {"ReorgYolo",          &FrontEndMcm::parseReorgYolo},
                {"CTCGreedyDecoder",   &FrontEndMcm::parseCTCDecoder},
                {"Proposal",           &FrontEndMcm::parseProposal},
                {"ROIPooling",         &FrontEndMcm::parseROIPooling},
                {"PSROIPooling",       &FrontEndMcm::parsePSROIPooling},
                {"Interp",             &FrontEndMcm::parseInterp},
                {"Custom",             &FrontEndMcm::parseCustom},
                {"MTCNN",              &FrontEndMcm::parseMTCNN},
                {"LSTMCell",           &FrontEndMcm::parseLSTMCell},
                {"Pad",                &FrontEndMcm::parsePad},
                {"Resample",           &FrontEndMcm::parseResample},
                {"ArgMax",             &FrontEndMcm::parseArgMax},
                {"TopK",               &FrontEndMcm::parseTopK},
                {"FakeQuantize",       &FrontEndMcm::parseFakeQuantize},
                {"Const",              &FrontEndMcm::parseConst},
        };

// clang-format on

}  // namespace

void FrontEndMcm::buildInitialModel(ie::ICNNNetwork& network) {
    if (!_config.customLayers().empty()) {
        _customLayers = CustomLayer::loadFromFile(_config.customLayers());
    }

    runCommonPasses(network);
    for (const auto& layer : _parsedNetwork.orderedLayers) {
        IE_ASSERT(layer != nullptr);
        _logger->debug("Try to parse layer %s", layer->name);

        const auto parser = [&] {
            const auto customLayersForType = _customLayers.find(layer->type);

            if (customLayersForType != _customLayers.end()) {
                const auto suitableLayers = getSuitableCustomLayers(customLayersForType->second, layer);
                if (!suitableLayers.empty()) {
                    return g_mcm_parsers.at("Custom");
                }
            }

            const auto it = g_mcm_parsers.find(layer->type);
            if (it == g_mcm_parsers.end()) {
                VPU_THROW_EXCEPTION << "Cannot convert layer \"" << layer->name << "\" due to unsupported layer type \""
                                    << layer->type << "\"";
            }

            return it->second;
        }();

        IE_ASSERT(parser != nullptr);

        McmNodeVector inputs;
        getInputData(layer, inputs);
        (this->*parser)(layer, inputs);
    }
    parseOutputData();
}

std::set<std::string> FrontEndMcm::checkSupportedLayers(ie::ICNNNetwork& network) {
    runCommonPasses(network);

    std::set<std::string> layerNames;

    for (const auto& layer : _parsedNetwork.orderedLayers) {
        IE_ASSERT(layer != nullptr);

        _logger->debug("Try to parse layer %s", layer->name);

        McmNodeVector inputs;
        getInputData(layer, inputs);

        auto it = g_mcm_parsers.find(layer->type);
        if (it != g_mcm_parsers.end()) {
            try {
                // If we can create and have not thrown exception, then layer is supported.
                auto parser = it->second;
                IE_ASSERT(parser != nullptr);

                (this->*parser)(layer, inputs);

                layerNames.insert(layer->name);
            } catch (const ie::details::InferenceEngineException&) {
                // TODO: continue instead?
                break;
            }
        }
    }

    return layerNames;
}

void FrontEndMcm::parseNetworkDFS(const ie::ICNNNetwork& network, ParsedNetwork& parsedNetwork) {
    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "parseNetworkDFS");

    ie::details::CaselessEq<std::string> cmp;

    //
    // Collect all network input data.
    //

    network.getInputsInfo(parsedNetwork.networkInputs);
    network.getOutputsInfo(parsedNetwork.networkOutputs);

    std::unordered_set<ie::DataPtr> allInputDatas;
    for (const auto& netInput : parsedNetwork.networkInputs) {
        auto inputInfo = netInput.second;
        IE_ASSERT(inputInfo != nullptr);

        auto inputData = inputInfo->getInputData();
        IE_ASSERT(inputData != nullptr);

        allInputDatas.insert(inputData);
    }

    //
    // Collect all network const data.
    //

    SmallVector<ie::CNNLayerPtr> initialLayers;

    for (const auto& layer : ie::CNNNetGetAllInputLayers(network)) {
        IE_ASSERT(layer != nullptr);

        if (!cmp(layer->type, "Const")) continue;

        if (layer->outData.size() != 1) {
            VPU_THROW_EXCEPTION << "Const layer " << layer->name << " has unsupported number of outputs "
                                << layer->outData.size();
        }

        if (layer->blobs.size() != 1) {
            VPU_THROW_EXCEPTION << "Const layer " << layer->name << " has unsupported number of blobs "
                                << layer->blobs.size();
        }

        initialLayers.emplace_back(std::move(layer));
    }

    //
    // Collect initial layers.
    //

    std::unordered_set<ie::CNNLayerPtr> visitedInitialLayers;

    for (const auto& inputData : allInputDatas) {
        for (const auto& consumer : getInputTo(inputData)) {
            auto initialLayer = consumer.second;
            IE_ASSERT(initialLayer != nullptr);

            if (visitedInitialLayers.count(initialLayer) > 0) continue;

            bool allInputsAvailable = true;
            for (const auto& in : initialLayer->insData) {
                auto input = in.lock();
                IE_ASSERT(input != nullptr);

                if (allInputDatas.count(input) == 0) {
                    allInputsAvailable = false;
                    break;
                }
            }

            if (allInputsAvailable) {
                visitedInitialLayers.insert(initialLayer);
                initialLayers.emplace_back(std::move(initialLayer));
            }
        }
    }

    IE_ASSERT(!initialLayers.empty());

    //
    // Run recursive DFS algorithm.
    //

    std::sort(
        initialLayers.begin(), initialLayers.end(), [](const ie::CNNLayerPtr& left, const ie::CNNLayerPtr& right) {
            ie::details::CaselessLess<std::string> cmp;
            return cmp(left->name, right->name);
        });

    InferenceEngine::CNNNetForestDFS(
        initialLayers,
        [&parsedNetwork](const ie::CNNLayerPtr& layer) {
            parsedNetwork.orderedLayers.emplace_back(layer);
        },
        false);

    std::reverse(parsedNetwork.orderedLayers.begin(), parsedNetwork.orderedLayers.end());
}

namespace {
// TODO: Move this function to utils
template <typename ResultType>
std::vector<ResultType> packBlobToVector(ie::Blob::Ptr blobPtr, size_t expectedSize) {
    IE_ASSERT(blobPtr != nullptr);

    if (expectedSize == 0) {
        expectedSize = blobPtr->size();
    }

    std::vector<ResultType> blobData(expectedSize, 0);

    // TODO: Make the ASSERT on equality after correction of blob creation in tests
    IE_ASSERT(expectedSize <= blobPtr->size());

    ie::Precision blobPrecision = blobPtr->getTensorDesc().getPrecision();

    // TODO: add proper layout handling. for now, weights are assumed to have OIYX
    if (blobPrecision == ie::Precision::FP16) {
        const auto* blobDataFP16 = blobPtr->cbuffer().as<const fp16_t*>();
        IE_ASSERT(blobDataFP16 != nullptr);

        for (size_t pos = 0; pos < expectedSize; pos++) {
            ResultType val = ie::PrecisionUtils::f16tof32(blobDataFP16[pos]);
            blobData[pos] = val;
        }
    } else if (blobPrecision == ie::Precision::FP32) {
        const auto* blobDataFP32 = blobPtr->cbuffer().as<const float*>();
        IE_ASSERT(blobDataFP32 != nullptr);

        for (size_t pos = 0; pos < expectedSize; pos++) {
            ResultType val = blobDataFP32[pos];
            blobData[pos] = val;
        }
    } else if (blobPrecision == ie::Precision::U8) {
        const auto* blobDataU8 = blobPtr->cbuffer().as<const uint8_t*>();
        IE_ASSERT(blobDataU8 != nullptr);

        for (size_t pos = 0; pos < expectedSize; pos++) {
            ResultType val = blobDataU8[pos];
            blobData[pos] = val;
        }
    } else if (blobPrecision == ie::Precision::I8) {
        const auto* blobDataI8 = blobPtr->cbuffer().as<const int8_t*>();
        IE_ASSERT(blobDataI8 != nullptr);

        for (size_t pos = 0; pos < expectedSize; pos++) {
            ResultType val = blobDataI8[pos];
            blobData[pos] = val;
        }
    } else if (blobPrecision == ie::Precision::I32) {
        const auto* blobDataI32 = blobPtr->cbuffer().as<const int32_t*>();
        IE_ASSERT(blobDataI32 != nullptr);

        for (size_t pos = 0; pos < expectedSize; pos++) {
            ResultType val = blobDataI32[pos];
            blobData[pos] = val;
        }
    } else if (blobPrecision == ie::Precision::I64) {
        const auto* blobDataI64 = blobPtr->cbuffer().as<const int64_t*>();
        IE_ASSERT(blobDataI64 != nullptr);

        for (size_t pos = 0; pos < expectedSize; pos++) {
            ResultType val = blobDataI64[pos];
            blobData[pos] = val;
        }
    } else {
        THROW_IE_EXCEPTION << "precision '" << blobPrecision << "' is not supported";
    }

    return blobData;
}
}  // namespace

static bool inputsHasSameScales(
    const std::vector<InferenceEngine::CNNLayerPtr>& inputs, const size_t& maxValues, const size_t& maxValuesIdx) {
    for (size_t i = 0; i < inputs.size(); i++) {
        auto quantizationParams1 = QuantizationDetails::getDetails(*inputs[i]);
        auto quantizationParams2 = QuantizationDetails::getDetails(*inputs[maxValuesIdx]);
        for (size_t c = 0; c < maxValues; c++) {
            size_t c1 = quantizationParams1.outputHighValues.size() == 1 ? 0 : c;
            size_t c2 = c;
            if ((quantizationParams1.outputHighValues[c1] - quantizationParams1.outputLowValues[c1]) !=
                (quantizationParams2.outputHighValues[c2] - quantizationParams2.outputLowValues[c2])) {
                return false;
            }
        }
        if (quantizationParams1.levels != quantizationParams2.levels) {
            return false;
        }
    }

    return true;
}

static bool inputsHasSameScalesAndZeroPoints(const std::vector<InferenceEngine::CNNLayerPtr>& inputs) {
    if (inputs.size() < 2) return true;

    auto ol = QuantizationDetails::getDetails(*inputs[0]).outputLowValues[0];
    auto oh = QuantizationDetails::getDetails(*inputs[0]).outputHighValues[0];
    auto levels = QuantizationDetails::getDetails(*inputs[0]).levels;
    for (size_t i = 0; i < inputs.size(); i++) {
        auto quantizationParams = QuantizationDetails::getDetails(*inputs[i]);
        for (size_t c = 0; c < quantizationParams.outputLowValues.size(); c++) {
            if ((quantizationParams.outputLowValues[c] != ol) || (quantizationParams.outputHighValues[c] != oh)) {
                return false;
            }
        }
        if (quantizationParams.levels != levels) {
            return false;
        }
    }

    return true;
}

static void setFakeQuantizeScales(const InferenceEngine::CNNLayerPtr& fakeQuantizeLayer, const size_t& maxLevels,
    const std::vector<double>& maxRange) {
    auto quantizationParams = QuantizationDetails::getDetails(*fakeQuantizeLayer);
    std::vector<float> scaledInputLowValues(quantizationParams.inputLowValues.size());
    std::vector<float> scaledInputHighValues(quantizationParams.inputLowValues.size());
    std::vector<float> scaledOutputLowValues(quantizationParams.outputLowValues.size());
    std::vector<float> scaledOutputHighValues(quantizationParams.outputLowValues.size());

    for (size_t i = 0; i < quantizationParams.inputLowValues.size(); i++) {
        double range = quantizationParams.inputHighValues[i] - quantizationParams.inputLowValues[i];
        double updatedInputLow = quantizationParams.inputLowValues[i] * maxRange[i] / range;
        scaledInputLowValues[i] = static_cast<float>(updatedInputLow);
        scaledInputHighValues[i] = static_cast<float>(updatedInputLow + maxRange[i]);
    }

    for (size_t i = 0; i < quantizationParams.outputLowValues.size(); i++) {
        double range = quantizationParams.outputHighValues[i] - quantizationParams.outputLowValues[i];
        double updatedOutputLow = quantizationParams.outputLowValues[i] * maxRange[i] / range;
        scaledOutputLowValues[i] = static_cast<float>(updatedOutputLow);
        scaledOutputHighValues[i] = static_cast<float>(updatedOutputLow + maxRange[i]);
    }

    fakeQuantizeLayer->params["levels"] = std::to_string(maxLevels);
    CNNNetworkHelper::updateBlobs(*fakeQuantizeLayer, 1, scaledInputLowValues);
    CNNNetworkHelper::updateBlobs(*fakeQuantizeLayer, 2, scaledInputHighValues);
    CNNNetworkHelper::updateBlobs(*fakeQuantizeLayer, 3, scaledOutputLowValues);
    CNNNetworkHelper::updateBlobs(*fakeQuantizeLayer, 4, scaledOutputHighValues);
}

static void setFakeQuantizeParams(const InferenceEngine::CNNLayerPtr& fakeQuantizeLayer, const size_t& maxLevels,
    const double& minVal, const double& maxVal) {
    auto quantizationParams = QuantizationDetails::getDetails(*fakeQuantizeLayer);
    std::vector<float> scaledInputLowValues(quantizationParams.inputLowValues.size());
    std::vector<float> scaledInputHighValues(quantizationParams.inputLowValues.size());
    std::vector<float> scaledOutputLowValues(quantizationParams.outputLowValues.size());
    std::vector<float> scaledOutputHighValues(quantizationParams.outputLowValues.size());

    for (size_t i = 0; i < quantizationParams.inputLowValues.size(); i++) {
        scaledInputLowValues[i] = minVal;
        scaledInputHighValues[i] = maxVal;
    }

    for (size_t i = 0; i < quantizationParams.outputLowValues.size(); i++) {
        scaledOutputLowValues[i] = minVal;
        scaledOutputHighValues[i] = maxVal;
    }

    fakeQuantizeLayer->params["levels"] = std::to_string(maxLevels);
    CNNNetworkHelper::updateBlobs(*fakeQuantizeLayer, 1, scaledInputLowValues);
    CNNNetworkHelper::updateBlobs(*fakeQuantizeLayer, 2, scaledInputHighValues);
    CNNNetworkHelper::updateBlobs(*fakeQuantizeLayer, 3, scaledOutputLowValues);
    CNNNetworkHelper::updateBlobs(*fakeQuantizeLayer, 4, scaledOutputHighValues);
}

namespace {
int appendFQfromInput(const CNNLayerPtr inputLayer, std::vector<CNNLayerPtr>& result) {
    int appended = 0;
    std::set<CNNLayerPtr> visited;
    std::stack<CNNLayerPtr> layers;
    layers.push(inputLayer);
    while (!layers.empty()) {
        auto input = layers.top();
        layers.pop();
        visited.insert(input);
        if ((input->type == "FakeQuantize") && (CNNNetworkHelper::getParent(*input)->type != "Const")) {
            appended++;
            result.push_back(input);
        } else {
            auto inputs = CNNNetworkHelper::getParents(*input);
            for (auto&& newInput : inputs) {
                if (!visited.count(newInput)) {
                    layers.push(newInput);
                }
            }
        }
    }
    return appended;
}
}  // namespace

void FrontEndMcm::alignEltwiseScales(ie::CNNNetwork& network) {
    ie::details::CNNNetworkIterator i(&static_cast<const ie::ICNNNetwork&>(network)), end;
    for (; i != end; ++i) {
        auto layer = *i;
        if (layer->type == "Eltwise") {
            std::vector<CNNLayerPtr> inputs;
            auto layerInputs = CNNNetworkHelper::getParents(*layer);
            bool canBeAlligned = true;
            for (auto& input : layerInputs) {
                int appended = appendFQfromInput(input, inputs);
                if (appended == 0) {  // Input is not quantized
                    _logger->debug("Input %s of layer %s is not quantized. Eltwise scales will not be aligned",
                        input->name, layer->name);
                    canBeAlligned = false;
                    break;
                }
            }
            if (!canBeAlligned) {
                continue;
            }
            size_t maxValues = 1;
            size_t maxValuesIdx = 0;
            for (size_t i = 0; i < inputs.size(); i++) {
                IE_ASSERT(inputs[i]->type == "FakeQuantize");
                if (maxValues < QuantizationDetails::getDetails(*inputs[i]).outputLowValues.size()) {
                    maxValues = QuantizationDetails::getDetails(*inputs[i]).outputLowValues.size();
                    maxValuesIdx = i;
                }
            }

            if (inputsHasSameScales(inputs, maxValues, maxValuesIdx)) {
                continue;
            }

            size_t maxLevels = 0;
            std::vector<double> maxRange(maxValues, 0.0);
            for (const auto& input : inputs) {
                auto quantizationParams = QuantizationDetails::getDetails(*input);
                if (maxLevels < quantizationParams.levels) maxLevels = quantizationParams.levels;

                for (size_t i = 0; i < maxValues; i++) {
                    size_t c = quantizationParams.outputHighValues.size() == 1 ? 0 : i;
                    double range = quantizationParams.outputHighValues[c] - quantizationParams.outputLowValues[c];
                    if (maxRange[i] < range) maxRange[i] = range;
                }
            }

            for (const auto& input : inputs) {
                setFakeQuantizeScales(input, maxLevels, maxRange);
            }
        }
    }
}

bool FrontEndMcm::needsConcatScaleAlignment(const ie::CNNLayerPtr& layer) {
    auto inputs = CNNNetworkHelper::getParents(*layer);
    for (auto& input : inputs) {
        if (input->type == "PriorBox") {
            return false;
        }
    }

    return true;
}

void FrontEndMcm::alignConcatScales(ie::CNNNetwork& network) {
    ie::details::CNNNetworkIterator i(&static_cast<const ie::ICNNNetwork&>(network)), end;
    for (; i != end; ++i) {
        auto layer = *i;
        if (layer->type == "Concat") {
            if (!needsConcatScaleAlignment(layer)) {
                continue;
            }
            std::vector<CNNLayerPtr> inputs;
            auto layerInputs = CNNNetworkHelper::getParents(*layer);
            bool canBeAlligned = true;
            for (auto& input : layerInputs) {
                int appended = appendFQfromInput(input, inputs);
                if (appended == 0) {  // Input is not quantized
                    _logger->debug("Input %s of layer %s is not quantized. Concat scales will not be aligned",
                        input->name, layer->name);
                    canBeAlligned = false;
                    break;
                }
            }
            if (!canBeAlligned) {
                continue;
            }
            for (auto& input : inputs) {
                IE_ASSERT(input->type == "FakeQuantize");
            }

            if (inputsHasSameScalesAndZeroPoints(inputs)) {
                continue;
            }

            size_t maxLevels = 0;
            double minVal = std::numeric_limits<double>::max();
            double maxVal = std::numeric_limits<double>::min();
            for (const auto& input : inputs) {
                auto quantizationParams = QuantizationDetails::getDetails(*input);
                if (maxLevels < quantizationParams.levels) maxLevels = quantizationParams.levels;

                for (size_t i = 0; i < quantizationParams.outputLowValues.size(); i++) {
                    double ol = quantizationParams.outputLowValues[i];
                    double oh = quantizationParams.outputHighValues[i];
                    if (minVal > ol) minVal = ol;
                    if (maxVal < oh) maxVal = oh;
                }
            }
            for (const auto& input : inputs) {
                setFakeQuantizeParams(input, maxLevels, minVal, maxVal);
            }
        }
    }
}

namespace {
template <typename T>
bool needAlignZeroPoints(std::vector<T> lowValues, std::vector<T> highValues, const float levels) {
    auto firstZP =
        QuantizationHelpers::calculateZeroPoint(highValues[0], lowValues[0], levels, InferenceEngine::Precision::U8);
    for (size_t i = 1; i < lowValues.size(); i++) {
        auto zp = QuantizationHelpers::calculateZeroPoint(
            highValues[i], lowValues[i], levels, InferenceEngine::Precision::U8);
        if (firstZP != zp) {
            return true;
        }
    }
    return false;
}

bool isFakeQuantizeOnWeights(const InferenceEngine::CNNLayerPtr& fakeQuantizeLayer) {
    InferenceEngine::DataPtr inputData = fakeQuantizeLayer->insData[0].lock();
    IE_ASSERT(inputData != nullptr);
    auto parentLayer = getCreatorLayer(inputData).lock();

    //  Check that FQ on weights
    return parentLayer->type == "Const" ? true : false;
}
}  // namespace

void FrontEndMcm::alignZeroPointsOnWeights(ie::CNNNetwork& network) {
    ie::details::CNNNetworkIterator i(&static_cast<const ie::ICNNNetwork&>(network)), end;
    for (; i != end; ++i) {
        auto layer = *i;
        if (layer->type == "FakeQuantize") {
            if (!isFakeQuantizeOnWeights(layer)) {
                continue;
            }

            auto quantizationParams = QuantizationDetails::getDetails(*layer);
            float levels = quantizationParams.levels;

            auto numberOfQuantParams = quantizationParams.outputLowValues.size();
            if (!needAlignZeroPoints(quantizationParams.outputLowValues, quantizationParams.outputHighValues, levels)) {
                continue;
            }
            double sumOfZeroPoints = 0;

            for (size_t i = 0; i < numberOfQuantParams; i++) {
                float ol = quantizationParams.outputLowValues[i];
                float oh = quantizationParams.outputHighValues[i];

                float x = -(levels - 1) * ol / (oh - ol);

                // re-calculate ZP for weights, we use U8 for weights
                sumOfZeroPoints += x;
            }
            auto avgZeroPoints = std::round(sumOfZeroPoints / numberOfQuantParams);

            // NOTE: ol is always negative value
            std::vector<float> newLowValues(numberOfQuantParams);
            std::vector<float> newHighValues(numberOfQuantParams);
            for (size_t i = 0; i < quantizationParams.outputLowValues.size(); i++) {
                float ol = quantizationParams.outputLowValues[i];
                float oh = quantizationParams.outputHighValues[i];

                float zpl = oh * avgZeroPoints / (avgZeroPoints - (levels - 1));
                float zph = ol - ol * (levels - 1) / avgZeroPoints;

                ol = std::min(ol, zpl);
                oh = std::max(oh, zph);
                newLowValues[i] = ol;
                newHighValues[i] = oh;
            }
            CNNNetworkHelper::updateBlobs(*layer, 1, newLowValues);
            CNNNetworkHelper::updateBlobs(*layer, 2, newHighValues);
            CNNNetworkHelper::updateBlobs(*layer, 3, newLowValues);
            CNNNetworkHelper::updateBlobs(*layer, 4, newHighValues);
        }
    }
}

void FrontEndMcm::runCommonPasses(ie::ICNNNetwork& network) {
    auto cnnNet = ie::CNNNetwork(std::shared_ptr<ie::ICNNNetwork>(&network, [](ie::ICNNNetwork*) {}));
    bool _isQuantized = QuantizationHelpers::isCNNNetworkQuantized(cnnNet);
    if (_isQuantized) {
        if (_config.eltwiseScalesAlignment()) {
            alignEltwiseScales(cnnNet);
        }
        if (_config.concatScalesAlignment()) {
            alignConcatScales(cnnNet);
        }
        if (_config.zeroPointsOnWeightsAlignment()) {
            alignZeroPointsOnWeights(cnnNet);
        }
    }

    if (!_config.serializeCNNBeforeCompileFile().empty()) {
        std::string origFileName = _config.serializeCNNBeforeCompileFile();
        auto baseFileName = (origFileName.substr(origFileName.length() - 4, 4) == ".xml")
                                ? origFileName.substr(0, origFileName.length() - 4)
                                : origFileName;

        cnnNet.serialize(baseFileName + ".xml", baseFileName + ".bin");
    }

    parseNetworkDFS(cnnNet, _parsedNetwork);
    parseInputData();
}

McmNode FrontEndMcm::getMcmData(const ie::DataPtr& ieData) {
    IE_ASSERT(ieData != nullptr);
    auto it = _ieToMcmMap.find(ieData);
    if (it == _ieToMcmMap.end()) {
        return nullptr;
    }

    return it->second;
}

void FrontEndMcm::bindData(const McmNode& data, const ie::DataPtr& ieData) {
    IE_ASSERT(_modelMcm.isValid(data->getMcmNode()));
    IE_ASSERT(_modelMcm.isValid(_modelMcm.getSourceOp(data->getMcmNode())));
    _ieToMcmMap[ieData] = data;
    data->setOrigData(ieData);
}

void FrontEndMcm::bindOutput(mv::Data::TensorIterator node, ie::DataPtr& layerOutput) {
    IE_ASSERT(layerOutput != nullptr);
    auto layer = std::make_shared<McmNodeObject>(node, layerOutput->getTensorDesc());
    _nodes.push_back(layer);
    bindData(layer, layerOutput);
}

void FrontEndMcm::getInputData(const ie::CNNLayerPtr& layer, McmNodeVector& inputs) {
    IE_ASSERT(layer != nullptr);
    inputs.resize(layer->insData.size());
    for (size_t i = 0; i < layer->insData.size(); ++i) {
        auto layerInput = layer->insData[i].lock();
        IE_ASSERT(layerInput != nullptr);
        inputs[i] = getMcmData(layerInput);
        IE_ASSERT(inputs[i] != nullptr);
    }
}

std::string getDimLabel(size_t dimIndex, ie::Layout ieLayout) {
    std::ostringstream ostr;
    ostr << ieLayout;
    const auto layoutStr = ostr.str();
    IE_ASSERT(dimIndex < layoutStr.size());
    return std::string(1, layoutStr[dimIndex]);
}

constexpr char FINISH_PARSING_STR[] = "Parsed to mcmModel as '%s";

void logParsingStartHelper(Logger::Ptr logger, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    logger->debug("Start parsing '%s' layer: '%s'", layer->type, layer->name);

    if (inputs.empty()) {
        logger->debug("Layer has no input");
    } else {
        for (size_t i = 0; i < inputs.size(); ++i)
            logger->debug("Layer input %d: '%s'", i, inputs[i]->getMcmNode()->getName());
    }
}

double inf = std::numeric_limits<double>::infinity();

// PATCH(mcm2): This is needed to avoid a static initializer fiasco.
static const mv::QuantizationParams& initialQuantParams() {
    static mv::QuantizationParams init{{0}, {1}, {-inf}, {inf}};
    return init;
};

bool isInputPrecisionSupported(const ie::Precision& inputPrecision) {
    const std::set<ie::Precision> supportedInPrecisions = {ie::Precision::U8, ie::Precision::FP16, ie::Precision::FP32};
    return supportedInPrecisions.find(inputPrecision) != supportedInPrecisions.end();
}

bool isInputLayoutSupported(const ie::Layout& inputLayout) {
    const std::set<ie::Layout> supportedInLayouts = {
        ie::Layout::NHWC, ie::Layout::NCHW, ie::Layout::CHW, ie::Layout::NC, ie::Layout::C};
    return supportedInLayouts.find(inputLayout) != supportedInLayouts.end();
}

bool isOutputPrecisionSupported(const ie::Precision& outputPrecision) {
    std::set<ie::Precision> supportedOutPrecisions = {ie::Precision::U8, ie::Precision::FP16, ie::Precision::FP32};
    return supportedOutPrecisions.find(outputPrecision) != supportedOutPrecisions.end();
}

bool isOutputLayoutSupported(const ie::Layout& outputLayout) {
    std::set<ie::Layout> supportedOutLayouts = {
        ie::Layout::NHWC, ie::Layout::NCHW, ie::Layout::CHW, ie::Layout::NC, ie::Layout::C};
    return supportedOutLayouts.find(outputLayout) != supportedOutLayouts.end();
}

void FrontEndMcm::parseInputData() {
    _logger->debug("Try to parse network input");

    for (const auto& inputInfo : _parsedNetwork.networkInputs) {
        auto netInput = inputInfo.second;
        IE_ASSERT(netInput != nullptr);

        auto ieData = netInput->getInputData();
        IE_ASSERT(ieData != nullptr);

        const auto& dataDesc = ieData->getTensorDesc();
        mv::Shape inputShape(getWHCN(dataDesc).getDims());

        auto inputLayerPtr = getCreatorLayer(ieData).lock();

        const InferenceEngine::Layout inputLayout = ieData->getTensorDesc().getLayout();
        if (!isInputLayoutSupported(inputLayout)) {
            VPU_THROW_EXCEPTION << "Input layout is not supported: " << ieData->getTensorDesc().getLayout();
        }

        const InferenceEngine::Precision inputPrecision = ieData->getTensorDesc().getPrecision();
        if (!isInputPrecisionSupported(inputPrecision)) {
            VPU_THROW_EXCEPTION << "Input data type is not supported: " << ieData->getTensorDesc().getPrecision();
        }

        bool networkInput = true;

        const auto mcmInputOrder = [&] {
            if (inputLayout == NCHW && _config.allowNCHWLayoutForMcmModelInput()) {
                return layoutToOrder(Layout::NCHW);
            }
            return layoutToOrder(Layout::NHWC);
        }();

        auto mvInput = _modelMcm.input(inputShape, precisionToDType(inputPrecision), mcmInputOrder,
            initialQuantParams(), networkInput, netInput->name());
        bindOutput(mvInput, ieData);
        _logger->debug("Network input '%s'(orig: '%s') parsed to mcmModel", mvInput->getName(), netInput->name());
    }
}

void FrontEndMcm::parseOutputData() {
    _logger->debug("Try to parse network output");

    for (const auto& outputInfo : _parsedNetwork.networkOutputs) {
        auto ieData = outputInfo.second;

        IE_ASSERT(ieData != nullptr);

        auto lastLayerOut = getMcmData(ieData);
        if (lastLayerOut == nullptr) {
            lastLayerOut = _nodes.back();
        }
        IE_ASSERT(lastLayerOut != nullptr);
        auto name = lastLayerOut->getMcmNode()->getName();

        const auto outputPrecision = ieData->getTensorDesc().getPrecision();
        if (!isOutputPrecisionSupported(outputPrecision)) {
            VPU_THROW_EXCEPTION << "Output data type is not supported: " << outputPrecision;
        }

        // TODO: kmbPlugin already has a function convert_data_type() for matching IE precision to mcm, but
        // in this case we can't use due to limitations on mcm level (not all precisions are supported).
        // mcmCompiler right now support only 2 types of precisions for output: U8 and FP16
        // for avoid this limitations plugin has a WA: translate FP32 output like a FP16 and convert output blob
        // in getResult() function after the inference.
        mv::DType outputType;
        switch (outputPrecision) {
        case ie::Precision::UNSPECIFIED:
            outputType = mv::DType("Default");
            break;
        case ie::Precision::U8:
            outputType = mv::DType("UInt8");
            break;
        case ie::Precision::FP16:
            outputType = mv::DType("Float16");
            break;
        case ie::Precision::FP32:
            outputType = mv::DType("Float16");
            break;
        default:
            VPU_THROW_EXCEPTION << "Data type handling is not implemented" << outputPrecision.name();
        }

        const InferenceEngine::Layout outputLayout = ieData->getTensorDesc().getLayout();
        if (!isOutputLayoutSupported(outputLayout)) {
            VPU_THROW_EXCEPTION << "Output layout is not supported: " << outputLayout;
        }

        // FIXME: REMOVE_ME postfix was added to make output name unique.
        // compiler will fail if output name is equal to some layer name
        // S#34832
        auto mvOutput = _modelMcm.output(
            lastLayerOut->getMcmNode(), outputType, {{}, {}, {}, {}}, true, outputInfo.first + "REMOVE_ME");

        _output = std::make_shared<McmNodeObject>(mvOutput, lastLayerOut->desc());
        _nodes.push_back(_output);
    }
}

namespace {

void cvtPaddingsFromCeilToFloorMode(
    int input_size_ceil, int output_size, int kernel, int stride, int& pad_start, int& pad_end) {
    const auto input_size_floor = mv::Tiling::inferInputSize(output_size, pad_start, pad_end, kernel, stride);

    pad_end = pad_end + (input_size_floor - input_size_ceil);
    pad_end = std::max(pad_end, 0);
}

}  // namespace

void FrontEndMcm::parseConvolution(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    auto input = inputs[0];

    auto convLayer = std::dynamic_pointer_cast<ie::ConvolutionLayer>(layer);
    IE_ASSERT(convLayer != nullptr);

    logParsingStartHelper(_logger, layer, {input});

    int kernelSizeX = convLayer->_kernel_x;
    int kernelSizeY = convLayer->_kernel_y;

    int kernelStrideX = convLayer->_stride_x;
    int kernelStrideY = convLayer->_stride_y;

    auto paddings = getPaddings(*convLayer);
    int padLeft = paddings.begin.exist(ie::X_AXIS) ? paddings.begin[ie::X_AXIS] : 0;
    int padRight = paddings.end.exist(ie::X_AXIS) ? paddings.end[ie::X_AXIS] : padLeft;
    int padTop = paddings.begin.exist(ie::Y_AXIS) ? paddings.begin[ie::Y_AXIS] : 0;
    int padBottom = paddings.end.exist(ie::Y_AXIS) ? paddings.end[ie::Y_AXIS] : padTop;

    int dilationX = convLayer->_dilation_x;
    int dilationY = convLayer->_dilation_y;
    if (dilationX != dilationY) {
        VPU_THROW_EXCEPTION << "kmb Convolution supports only equal dilationX and dilationY";
    }

    size_t groupSize = convLayer->_group;

    auto layerOutput = layer->outData[0];
    IE_ASSERT(layerOutput != nullptr);
    auto outDesc = layerOutput->getTensorDesc();
    cvtPaddingsFromCeilToFloorMode(input->origData()->getDims().at(3), outDesc.getDims().at(3),
        kernelSizeX * dilationX - (dilationX - 1), kernelStrideX, padLeft, padRight);
    cvtPaddingsFromCeilToFloorMode(input->origData()->getDims().at(2), outDesc.getDims().at(2),
        kernelSizeY * dilationY - (dilationY - 1), kernelStrideY, padTop, padBottom);

    mv::DType convolutionDataType("Default");
    mv::Data::TensorIterator mvConv;
    mv::Data::TensorIterator mvConvOnly;

    size_t inputGroupSize, outputGroupSize, stub;
    parseDims(input->desc(), stub, inputGroupSize, stub, stub);
    parseDims(outDesc, stub, outputGroupSize, stub, stub);

    bool isDepthWiseConv = groupSize > 1 && groupSize == inputGroupSize && groupSize == outputGroupSize;
    auto mvWeights = inputs[1]->getMcmNode();

    if (isDepthWiseConv) {
        // TODO: Need align API in mcmCompiler
        // mcm expects (1,*,*,*) shape for depthwise weights, but Openvino has a (*,1,*,*)

        auto sourceWeightsOp = _modelMcm.getSourceOp(mvWeights);
        auto constWeightTensor = mvWeights;
        if (sourceWeightsOp->getOpType() == "FakeQuantize") {
            constWeightTensor = sourceWeightsOp->getInputTensor(0);
            sourceWeightsOp = _modelMcm.getSourceOp(constWeightTensor);
        }
        constWeightTensor->set<bool>("is_depthwise_weights", true);
        sourceWeightsOp->set<bool>("is_depthwise_weights", true);
        const std::initializer_list<std::size_t> newWeightsShape = {
            static_cast<std::size_t>(kernelSizeX), static_cast<std::size_t>(kernelSizeY), inputGroupSize, 1lu};

        constWeightTensor->setShape(newWeightsShape);
        mvWeights->setShape(newWeightsShape);
        sourceWeightsOp->set<mv::Shape>("shape", newWeightsShape);

        mvConv = _modelMcm.depthwiseConv(input->getMcmNode(), mvWeights,
            {static_cast<uint16_t>(kernelStrideX), static_cast<uint16_t>(kernelStrideY)},
            {static_cast<uint16_t>(padLeft), static_cast<uint16_t>(padRight), static_cast<uint16_t>(padTop),
                static_cast<uint16_t>(padBottom)},
            static_cast<unsigned>(dilationX), convolutionDataType, initialQuantParams(), convLayer->name);
    } else {
        mvConv = _modelMcm.conv(input->getMcmNode(), mvWeights,
            {static_cast<uint16_t>(kernelStrideX), static_cast<uint16_t>(kernelStrideY)},
            {static_cast<uint16_t>(padLeft), static_cast<uint16_t>(padRight), static_cast<uint16_t>(padTop),
                static_cast<uint16_t>(padBottom)},
            static_cast<unsigned>(dilationX), static_cast<unsigned>(groupSize), convolutionDataType,
            initialQuantParams(), convLayer->name);
    }

    //  Need quantize bias, this logic provide by MCM team, need check
    if (inputs.size() == 3) {
        mvConvOnly = mvConv;
        auto mvBiases = inputs[2]->getMcmNode();
        mvConv =
            _modelMcm.bias(mvConvOnly, mvBiases, mv::DType("Default"), initialQuantParams(), convLayer->name + ":bias");
        _logger->debug(
            "'%s' layer '%s': Bias part (%s) added to mcmModel", convLayer->type, convLayer->name, mvConv->getName());
    }

    bindOutput(mvConv, layerOutput);
    _logger->debug(FINISH_PARSING_STR, mvConv->getName());
}

void FrontEndMcm::parsePooling(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    IE_ASSERT(inputs.size() == 1);

    auto input = inputs[0];
    auto poolLayer = std::dynamic_pointer_cast<ie::PoolingLayer>(layer);
    IE_ASSERT(poolLayer != nullptr);

    const auto rounding_type = layer->GetParamAsString("rounding_type", "floor");

    logParsingStartHelper(_logger, layer, inputs);

    int kernelSizeX = poolLayer->_kernel_x;
    int kernelSizeY = poolLayer->_kernel_y;

    int kernelStrideX = poolLayer->_stride_x;
    int kernelStrideY = poolLayer->_stride_y;

    auto paddings = getPaddings(*poolLayer);
    int padLeft = paddings.begin.exist(ie::X_AXIS) ? paddings.begin[ie::X_AXIS] : 0;
    int padRight = paddings.end.exist(ie::X_AXIS) ? paddings.end[ie::X_AXIS] : padLeft;
    int padTop = paddings.begin.exist(ie::Y_AXIS) ? paddings.begin[ie::Y_AXIS] : 0;
    int padBottom = paddings.end.exist(ie::Y_AXIS) ? paddings.end[ie::Y_AXIS] : padTop;

    auto poolType = poolLayer->_type;

    auto layerOutput = layer->outData[0];
    IE_ASSERT(layerOutput != nullptr);

    auto outDesc = layerOutput->getTensorDesc();

    cvtPaddingsFromCeilToFloorMode(
        input->origData()->getDims().at(3), outDesc.getDims().at(3), kernelSizeX, kernelStrideX, padLeft, padRight);
    cvtPaddingsFromCeilToFloorMode(
        input->origData()->getDims().at(2), outDesc.getDims().at(2), kernelSizeY, kernelStrideY, padTop, padBottom);

    mv::Data::TensorIterator mvPooling;
    if (poolType == ie::PoolingLayer::AVG) {
        mvPooling = _modelMcm.averagePool(inputs[0]->getMcmNode(),
            {static_cast<uint16_t>(kernelSizeX), static_cast<uint16_t>(kernelSizeY)},
            {static_cast<uint16_t>(kernelStrideX), static_cast<uint16_t>(kernelStrideY)},
            {static_cast<uint16_t>(padLeft), static_cast<uint16_t>(padRight), static_cast<uint16_t>(padTop),
                static_cast<uint16_t>(padBottom)},
            poolLayer->_exclude_pad, mv::DType("Default"), initialQuantParams(), poolLayer->name);
    } else {
        mvPooling = _modelMcm.maxPool(inputs[0]->getMcmNode(),
            {static_cast<uint16_t>(kernelSizeX), static_cast<uint16_t>(kernelSizeY)},
            {static_cast<uint16_t>(kernelStrideX), static_cast<uint16_t>(kernelStrideY)},
            {static_cast<uint16_t>(padLeft), static_cast<uint16_t>(padRight), static_cast<uint16_t>(padTop),
                static_cast<uint16_t>(padBottom)},
            poolLayer->_exclude_pad, mv::DType("Default"), initialQuantParams(), poolLayer->name);
    }

    bindOutput(mvPooling, layerOutput);
    _logger->debug(FINISH_PARSING_STR, mvPooling->getName());
}

void FrontEndMcm::parseFullyConnected(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    auto FClayer = std::dynamic_pointer_cast<ie::FullyConnectedLayer>(layer);
    IE_ASSERT(FClayer != nullptr);

    logParsingStartHelper(_logger, layer, inputs);

    auto input = inputs[0];

    size_t dimC, dimY, dimX, stub;
    parseDims(input->desc(), stub, dimC, dimY, dimX, 1);

    auto layerOutput = FClayer->outData[0];
    IE_ASSERT(layerOutput != nullptr);

    auto mvWeights = inputs[1]->getMcmNode();
    mv::DType layerDataType("Default");
    auto mvFullyConnected =
        _modelMcm.fullyConnected(input->getMcmNode(), mvWeights, layerDataType, initialQuantParams(), FClayer->name);

    if (inputs.size() == 3) {
        auto mvBiases = inputs[2]->getMcmNode();
        mvFullyConnected = _modelMcm.bias(
            mvFullyConnected, mvBiases, mv::DType("Default"), initialQuantParams(), FClayer->name + ":bias");
        _logger->debug("'%s' layer '%s': Bias part (%s) added to mcmModel", FClayer->type, FClayer->name,
            mvFullyConnected->getName());
    }

    bindOutput(mvFullyConnected, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, mvFullyConnected->getName());
}

void FrontEndMcm::parseReLU(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    IE_ASSERT(inputs.size() == 1);

    auto reluLayer = std::dynamic_pointer_cast<ie::ReLULayer>(layer);
    IE_ASSERT(reluLayer != nullptr);

    logParsingStartHelper(_logger, layer, inputs);

    float negativeSlope = reluLayer->negative_slope;
    mv::Data::TensorIterator mvRelu;
    if (std::fabs(negativeSlope) < std::numeric_limits<float>::epsilon()) {
        mvRelu = _modelMcm.relu(inputs[0]->getMcmNode(), mv::DType("Default"), initialQuantParams(), reluLayer->name);
    } else {
        mvRelu = _modelMcm.leakyRelu(
            inputs[0]->getMcmNode(), negativeSlope, mv::DType("Default"), initialQuantParams(), reluLayer->name);
    }

    bindOutput(mvRelu, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, mvRelu->getName());
}

void FrontEndMcm::parseSoftMax(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    IE_ASSERT(inputs.size() == 1);

    auto softMaxLayer = std::dynamic_pointer_cast<ie::SoftMaxLayer>(layer);
    IE_ASSERT(softMaxLayer != nullptr);

    IE_ASSERT(static_cast<size_t>(softMaxLayer->axis) < inputs[0]->desc().getDims().size());

    logParsingStartHelper(_logger, layer, inputs);

    const auto ieLayout = ie::TensorDesc::getLayoutByDims(inputs[0]->desc().getDims());
    std::string mcmAxis = getDimLabel(softMaxLayer->axis, ieLayout);
    auto mvSoftmax = _modelMcm.softmax(
        inputs[0]->getMcmNode(), mcmAxis, mv::DType("Default"), initialQuantParams(), softMaxLayer->name);

    bindOutput(mvSoftmax, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, mvSoftmax->getName());
}

void FrontEndMcm::parseNorm(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    IE_ASSERT(inputs.size() == 1);
    auto normLayer = std::dynamic_pointer_cast<ie::NormLayer>(layer);
    IE_ASSERT(normLayer != nullptr);

    logParsingStartHelper(_logger, layer, inputs);

    auto alpha = static_cast<double>(normLayer->_alpha);
    auto beta = static_cast<double>(normLayer->_beta);
    std::string region = std::to_string(normLayer->_k);
    auto mvLRN = _modelMcm.norm(inputs[0]->getMcmNode(), alpha, beta, region, normLayer->_size, mv::DType("Default"),
        initialQuantParams(), normLayer->name);

    bindOutput(mvLRN, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, mvLRN->getName());
}

void FrontEndMcm::parseScaleImpl(
    const ie::CNNLayerPtr& layer, const McmNodeVector& inputs, std::vector<double>& weights, ie::Blob::Ptr biases) {
    logParsingStartHelper(_logger, layer, inputs);

    auto input = inputs[0];

    mv::Shape weightsShape = {weights.size()};
    auto mvWeights = _modelMcm.constant(
        weights, weightsShape, mv::DType("Float32"), mv::Order::getColMajorID(1), initialQuantParams());

    auto scale =
        _modelMcm.scale(input->getMcmNode(), mvWeights, mv::DType("Default"), initialQuantParams(), layer->name);
    auto scaleShift = scale;

    std::vector<double> biasData;
    if (biases != nullptr) {
        biasData = packBlobToVector<double>(biases, biases->size());
        mv::Shape shiftShape{biases->size()};
        auto shiftData = _modelMcm.constant(
            biasData, shiftShape, mv::DType("Float32"), mv::Order::getColMajorID(1), initialQuantParams());
        // TODO: return logging
        scaleShift =
            _modelMcm.bias(scale, shiftData, mv::DType("Default"), initialQuantParams(), layer->name + ":bias");
    }

    bindOutput(scaleShift, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, scaleShift->getName());
}

void FrontEndMcm::parseScale(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    IE_ASSERT(inputs.size() == 1);
    auto scaleLayer = std::dynamic_pointer_cast<ie::ScaleShiftLayer>(layer);
    IE_ASSERT(scaleLayer != nullptr);
    IE_ASSERT(scaleLayer->_weights != nullptr);

    if (scaleLayer->_broadcast != 0) {
        VPU_THROW_EXCEPTION << "Layer " << scaleLayer->name << " doesn't support broadcast param";
    }
    auto input = inputs[0];
    size_t dimC, stub;
    parseDims(input->desc(), stub, dimC, stub, stub);

    std::vector<double> scaleData = packBlobToVector<double>(scaleLayer->_weights, scaleLayer->_weights->size());

    parseScaleImpl(layer, inputs, scaleData, scaleLayer->_biases);
}

void FrontEndMcm::parsePermute(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    IE_ASSERT(inputs.size() == 1);

    logParsingStartHelper(_logger, layer, inputs);

    auto ieOrder = layer->GetParamAsInts("order");

    std::string newOrder;

    // 4d NCHW inputs are supported
    const auto ieLayout = ie::TensorDesc::getLayoutByDims(inputs[0]->desc().getDims());
    for (size_t i = 0; i < ieOrder.size(); i++) {
        newOrder += getDimLabel(ieOrder[ieOrder.size() - 1 - i], ieLayout);
    }

    auto mvPerm = _modelMcm.permute(
        inputs[0]->getMcmNode(), mv::Order(newOrder), mv::DType("Default"), initialQuantParams(), layer->name);

    // Workaround to avoid parsing stage crash:
    // 'ArgumentError: attribute identifer quantParams - Undefined identifier'
    // [Track number: D#2284, D#2237]
    mvPerm->set<mv::QuantizationParams>("quantParams", initialQuantParams());

    bindOutput(mvPerm, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, mvPerm->getName());
}

void FrontEndMcm::parseEltwise(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    auto eltwiseLayer = std::dynamic_pointer_cast<ie::EltwiseLayer>(layer);
    IE_ASSERT(eltwiseLayer != nullptr);

    logParsingStartHelper(_logger, layer, inputs);
    mv::Data::TensorIterator mvEltwise;
    std::vector<mv::Data::TensorIterator> mvInputs;
    for (const auto& input : inputs) {
        mvInputs.push_back(input->getMcmNode());
    }

    if (inputs.size() > 2) {
        VPU_THROW_EXCEPTION << eltwiseLayer->name
                            << "Eltwise operations with more than 2 operands is not supported by kmbPlugin";
    }

    if (eltwiseLayer->_operation == ie::EltwiseLayer::eOperation::Sub ||
        eltwiseLayer->_operation == ie::EltwiseLayer::eOperation::Sum) {
        for (size_t i = 0; i < eltwiseLayer->coeff.size(); ++i) {
            if (std::abs(eltwiseLayer->coeff[i]) != 1.0f) {
                VPU_THROW_EXCEPTION
                    << eltwiseLayer->name
                    << " Eltwise Sum/Sub operations with such coefficients is not supported by kmbPlugin";
            }
        }
    }

    switch (eltwiseLayer->_operation) {
    case ie::EltwiseLayer::eOperation::Sub:
        mvEltwise =
            _modelMcm.eltwise(mvInputs, "Subtract", mv::DType("Default"), initialQuantParams(), eltwiseLayer->name);
        break;
    case ie::EltwiseLayer::eOperation::Sum:
        mvEltwise = _modelMcm.eltwise(mvInputs, "Add", mv::DType("Default"), initialQuantParams(), eltwiseLayer->name);
        break;
    case ie::EltwiseLayer::eOperation::Prod:
        mvEltwise =
            _modelMcm.eltwise(mvInputs, "Multiply", mv::DType("Default"), initialQuantParams(), eltwiseLayer->name);
        break;
    default:
        VPU_THROW_EXCEPTION << "Eltwise operation" << eltwiseLayer->_operation << " is not supported";
    }

    bindOutput(mvEltwise, layer->outData[0]);
    _logger->debug(FINISH_PARSING_STR, mvEltwise->getName());
}

void FrontEndMcm::parseBias(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    mv::Data::TensorIterator mvBias;
    if (inputs.size() == 1) {
        logParsingStartHelper(_logger, layer, inputs);

        auto input = inputs[0];
        size_t dimC, stub;
        parseDims(input->desc(), stub, dimC, stub, stub);
        mv::Shape biasShape = {dimC};
        int biasesSize = dimC;
        auto biases = layer->blobs["biases"];

        auto weights = layer->blobs["weights"];
        auto biasData = packBlobToVector<double>(biases, biasesSize);

        auto mvBiasValues = _modelMcm.constant(biasData, biasShape, mv::DType("Float16"), mv::Order("W"));
        mvBias =
            _modelMcm.bias(input->getMcmNode(), mvBiasValues, mv::DType("Default"), initialQuantParams(), layer->name);
    } else if (inputs.size() == 2) {
        logParsingStartHelper(_logger, layer, inputs);

        auto input = inputs[0];
        auto input1 = inputs[1];
        mvBias = _modelMcm.bias(
            input->getMcmNode(), input1->getMcmNode(), mv::DType("Default"), initialQuantParams(), layer->name);
    } else {
        VPU_THROW_EXCEPTION << "Bias layer does not support " << inputs.size() << " inputs";
    }

    bindOutput(mvBias, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, mvBias->getName());
}

void FrontEndMcm::parseClamp(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    IE_ASSERT(inputs.size() == 1);

    auto clampLayer = std::dynamic_pointer_cast<ie::ClampLayer>(layer);
    IE_ASSERT(clampLayer != nullptr);

    logParsingStartHelper(_logger, layer, inputs);

    auto mvClampMin = _modelMcm.minimum(inputs[0]->getMcmNode(), clampLayer->max_value, mv::DType("Default"),
        initialQuantParams(), clampLayer->name + "clamp-min");
    auto mvClampMax = _modelMcm.maximum(
        mvClampMin, clampLayer->min_value, mv::DType("Default"), initialQuantParams(), clampLayer->name + "clamp-max");
    bindOutput(mvClampMax, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, mvClampMax->getName());
}

void FrontEndMcm::parseReshape(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    auto layerOutput = layer->outData[0];
    IE_ASSERT(layerOutput != nullptr);

    logParsingStartHelper(_logger, layer, inputs);

    for (size_t i = 1; i < inputs.size(); i++) {
        _modelMcm.removeOp(_modelMcm.getSourceOp(inputs[i]->getMcmNode()));
    }

    // Because mcmCompiler supports only "dense" layouts
    // for example NC should be represented as NCHW with dims NC11
    // Formation of a newShape, "dense" shape with 1, substituted in the places of non-existent measurements
    // TODO: Tests on parsing/compilation of different cases of reshape should be added: Jira: CVS-20409
    // McmCompiler accept only input in WHCN format
    mv::Shape newShape(getWHCN(layerOutput->getTensorDesc()).getDims());

    auto mvReshape =
        _modelMcm.reshape(inputs[0]->getMcmNode(), newShape, mv::DType("Default"), initialQuantParams(), layer->name);

    bindOutput(mvReshape, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, mvReshape->getName());
}

void FrontEndMcm::parseConcat(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    IE_ASSERT(!inputs.empty());

    auto concatLayer = std::dynamic_pointer_cast<ie::ConcatLayer>(layer);
    IE_ASSERT(concatLayer != nullptr);
    IE_ASSERT(concatLayer->_axis < inputs[0]->desc().getDims().size());

    logParsingStartHelper(_logger, layer, inputs);

    const auto ieLayout = ie::TensorDesc::getLayoutByDims(inputs[0]->desc().getDims());
    std::string mcmAxis = getDimLabel(concatLayer->_axis, ieLayout);

    std::vector<mv::Data::TensorIterator> concatInputs;
    for (const auto& input : inputs) {
        concatInputs.push_back(input->getMcmNode());
    }

    auto mvConcat =
        _modelMcm.concat(concatInputs, mcmAxis, mv::DType("Default"), initialQuantParams(), concatLayer->name);
    bindOutput(mvConcat, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, mvConcat->getName());
}

void FrontEndMcm::parseRegionYolo(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    IE_ASSERT(inputs.size() == 1);

    logParsingStartHelper(_logger, layer, inputs);

    auto coords = layer->GetParamAsUInt("coords");
    auto classes = layer->GetParamAsUInt("classes");
    auto do_softmax = layer->GetParamAsBool("do_softmax");
    auto num = layer->GetParamAsUInt("num");
    auto mask = layer->GetParamAsUInts("mask", {});

    auto region = _modelMcm.regionYolo(inputs[0]->getMcmNode(), coords, classes, do_softmax, num, mask,
        mv::DType("Default"), initialQuantParams(), layer->name);
    bindOutput(region, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, region->getName());
}

void FrontEndMcm::parseReorgYolo(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    IE_ASSERT(inputs.size() == 1);

    logParsingStartHelper(_logger, layer, inputs);

    auto stride = layer->GetParamAsUInt("stride");

    auto reorg =
        _modelMcm.reorgYolo(inputs[0]->getMcmNode(), stride, mv::DType("Default"), initialQuantParams(), layer->name);
    bindOutput(reorg, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, reorg->getName());
}

InferenceEngine::CNNLayerPtr getInputLayerSafe(const InferenceEngine::CNNLayerPtr& layer, const size_t index) {
    IE_ASSERT(index < layer->insData.size());
    auto inputData = layer->insData[index].lock();
    IE_ASSERT(inputData != nullptr);
    auto inputLayer = getCreatorLayer(inputData).lock();
    IE_ASSERT(inputLayer != nullptr);
    return inputLayer;
}

namespace {
bool isInteger(ie::Precision iePrecision) {
    static std::set<ie::Precision> integer_precision{
        ie::Precision::I8, ie::Precision::U8, ie::Precision::I32, ie::Precision::I64};
    return integer_precision.count(iePrecision);
}

}  // namespace

void FrontEndMcm::parseConst(const InferenceEngine::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    UNUSED(inputs);
    IE_ASSERT(layer->type == "Const");
    auto foundBlob = layer->blobs.begin();
    if (foundBlob == layer->blobs.end()) {
        VPU_THROW_EXCEPTION << "Const layer blob is not supportied";
    }
    const auto constBlob = foundBlob->second;
    auto blobPrecision = constBlob->getTensorDesc().getPrecision();
    auto mcmShape = sizeVectorToShape(layer->outData.front()->getDims());

    // IE add constant folding for PriorBox\PriorBox clustered
    // As a result we get 3d const instead of concat for DetectionOut layer
    // Current case unsupported on mcm side. WA expand dims (3d->4d)
    if (mcmShape.ndims() == 3) {
        mcmShape = mv::Shape::augment_major(mcmShape, 4);
    }

    if (isInteger(blobPrecision)) {
        std::vector<int64_t> constData = packBlobToVector<int64_t>(constBlob, constBlob->size());
        auto constMCM =
            _modelMcm.constantInt(constData, mcmShape, precisionToDType(constBlob->getTensorDesc().getPrecision()),
                mv::Order::getColMajorID(mcmShape.ndims()), initialQuantParams(), layer->name);
        bindOutput(constMCM, layer->outData[0]);
    } else {
        std::vector<double> constData = packBlobToVector<double>(constBlob, constBlob->size());
        auto constMCM =
            _modelMcm.constant(constData, mcmShape, precisionToDType(Precision(Precision::ePrecision::FP32)),
                // Initially  this parameter is: precisionToDType(constBlob->getTensorDesc().getPrecision()),
                // but as Work Around it is set to: precisionToDType(Precision(Precision::ePrecision::FP32)).
                // It is so just because mcmCompiler has not supported FP16 yet.
                // Do not forget to redo it when support for FP16 will be available in mcmCompiler.
                mv::Order::getColMajorID(mcmShape.ndims()), initialQuantParams(), layer->name);
        bindOutput(constMCM, layer->outData[0]);
    }
}

void FrontEndMcm::parseFakeQuantize(const InferenceEngine::CNNLayerPtr& layer, const vpu::McmNodeVector& inputs) {
    IE_ASSERT(layer->type == "FakeQuantize");

    const auto inputLowLayer = getInputLayerSafe(layer, 1);
    const auto inputHighLayer = getInputLayerSafe(layer, 2);
    const auto outputLowLayer = getInputLayerSafe(layer, 3);
    const auto outputHighLayer = getInputLayerSafe(layer, 4);

    const auto levels = layer->GetParamAsInt("levels");

    auto fakeQuantize = _modelMcm.fakeQuantize(inputs[0]->getMcmNode(), inputs[1]->getMcmNode(),
        inputs[2]->getMcmNode(), inputs[3]->getMcmNode(), inputs[4]->getMcmNode(), levels, layer->name);
    bindOutput(fakeQuantize, layer->outData[0]);
}

void FrontEndMcm::parseTopK(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    IE_ASSERT(inputs.size() == 2);
    logParsingStartHelper(_logger, layer, inputs);
    auto axis = layer->GetParamAsUInt("axis");
    auto mode = layer->GetParamAsString("mode");
    auto sort = layer->GetParamAsString("sort");
    int32_t k = inputs[1]->getMcmNode()->getIntData()[0];
    _modelMcm.removeOp(_modelMcm.getSourceOp(inputs[1]->getMcmNode()));

    auto topK = _modelMcm.topK(
        inputs[0]->getMcmNode(), sort, mode, k, axis, mv::DType("Default"), initialQuantParams(), layer->name);
    bindOutput(topK, layer->outData[0]);
    auto topKOp = _modelMcm.getSourceOp(topK);
    if (topKOp->outputSlots() > 1) bindOutput(topKOp->getOutputTensor(1), layer->outData[1]);
    _logger->debug(FINISH_PARSING_STR, topK->getName());
}

void FrontEndMcm::parseArgMax(const ie::CNNLayerPtr&, const McmNodeVector&) {
    VPU_THROW_EXCEPTION << "ArgMax layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseGRN(const ie::CNNLayerPtr&, const McmNodeVector&) {
    VPU_THROW_EXCEPTION << "GRN layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseMVN(const ie::CNNLayerPtr&, const McmNodeVector&) {
    VPU_THROW_EXCEPTION << "MVN layer is not supported by kmbPlugin";
}
void FrontEndMcm::parsePower(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    IE_ASSERT(inputs.size() == 1);
    auto powerLayer = std::dynamic_pointer_cast<ie::PowerLayer>(layer);
    IE_ASSERT(powerLayer != nullptr);

    if (powerLayer->power != 1) {
        VPU_THROW_EXCEPTION << "Layer " << powerLayer->name << " supports only power = 1";
    }

    auto input = inputs[0];

    size_t dimC, stub;
    parseDims(input->desc(), stub, dimC, stub, stub);

    double powerScale = powerLayer->scale;
    std::vector<double> scaleData;
    scaleData.resize(dimC, powerScale);

    ie::Blob::Ptr biases;
    if (powerLayer->offset != 0) {
        SizeVector dims({dimC});
        const TensorDesc biasTensor = TensorDesc(InferenceEngine::Precision::FP32, dims, ie::C);

        biases = make_blob_with_precision(biasTensor);
        biases->allocate();
        float* raw = biases->buffer().as<float*>();
        for (size_t i = 0; i < dimC; i++) {
            raw[i] = powerLayer->offset;
        }
    }

    parseScaleImpl(layer, inputs, scaleData, biases);
}

void FrontEndMcm::parseDetectionOutput(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    IE_ASSERT(inputs.size() == 3);

    int64_t num_classes = layer->GetParamAsInt("num_classes", 21);
    int64_t keep_top_k = layer->GetParamAsInt("keep_top_k", 200);
    double nms_threshold = layer->GetParamAsFloat("nms_threshold", 0.45);
    int64_t background_label_id = layer->GetParamAsInt("background_label_id", 0);
    int64_t top_k = layer->GetParamAsInt("top_k", 400);
    bool variance_encoded_in_target = layer->GetParamAsInt("variance_encoded_in_target", 0);
    std::string code_type = layer->GetParamAsString("code_type");
    bool share_location = layer->GetParamAsInt("share_location", 1);
    double confidence_threshold = layer->GetParamAsFloat("confidence_threshold", 0.01);
    bool clip_before_nms = layer->GetParamAsInt("clip_before_nms", 0);
    bool clip_after_nms = layer->GetParamAsInt("clip_after_nms", 0);
    int64_t decrease_label_id = 0;
    bool normalized = layer->GetParamAsInt("normalized", 1);
    int64_t input_height = layer->GetParamAsInt("input_height", 1);
    int64_t input_width = layer->GetParamAsInt("input_width", 1);
    double objectness_score = 0;

    mv::Data::TensorIterator mvDetectionOutput;
    std::vector<mv::Data::TensorIterator> detectionInputs;

    detectionInputs.push_back(inputs[0]->getMcmNode());
    detectionInputs.push_back(inputs[1]->getMcmNode());
    detectionInputs.push_back(inputs[2]->getMcmNode());

    mvDetectionOutput = _modelMcm.detectionOutput(detectionInputs, num_classes, keep_top_k, nms_threshold,
        background_label_id, top_k, variance_encoded_in_target, code_type, share_location, confidence_threshold,
        clip_before_nms, clip_after_nms, decrease_label_id, normalized, input_height, input_width, objectness_score,
        mv::DType("Default"), initialQuantParams(), layer->name);

    bindOutput(mvDetectionOutput, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, mvDetectionOutput->getName());
}

void FrontEndMcm::parseSigmoid(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    IE_ASSERT(inputs.size() == 1);

    IE_ASSERT(layer != nullptr);
    logParsingStartHelper(_logger, layer, inputs);

    auto inputQuantParams = inputs[0]->getMcmNode()->get<mv::QuantizationParams>("quantParams");
    auto mvSigmoid = _modelMcm.sigmoid(inputs[0]->getMcmNode(), mv::DType("Default"), inputQuantParams, layer->name);

    bindOutput(mvSigmoid, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, mvSigmoid->getName());
}

void FrontEndMcm::parseTanH(const ie::CNNLayerPtr&, const McmNodeVector&) {
    VPU_THROW_EXCEPTION << "TanH layer is not supported by kmbPlugin";
}

void FrontEndMcm::parsePReLU(const ie::CNNLayerPtr&, const McmNodeVector&) {
    VPU_THROW_EXCEPTION << "PReLU layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseBatchNorm(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "BatchNorm layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseDeconvolution(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    IE_ASSERT(!inputs.empty());
    auto input = inputs[0];

    auto deconvLayer = std::dynamic_pointer_cast<ie::DeconvolutionLayer>(layer);
    IE_ASSERT(deconvLayer != nullptr);

    logParsingStartHelper(_logger, layer, {input});
    int kernelSizeX = deconvLayer->_kernel_x;
    int kernelSizeY = deconvLayer->_kernel_y;

    int kernelStrideX = deconvLayer->_stride_x;
    int kernelStrideY = deconvLayer->_stride_y;

    auto paddings = getPaddings(*deconvLayer);
    int padLeft = paddings.begin.exist(ie::X_AXIS) ? paddings.begin[ie::X_AXIS] : 0;
    int padRight = paddings.end.exist(ie::X_AXIS) ? paddings.end[ie::X_AXIS] : padLeft;
    int padTop = paddings.begin.exist(ie::Y_AXIS) ? paddings.begin[ie::Y_AXIS] : 0;
    int padBottom = paddings.end.exist(ie::Y_AXIS) ? paddings.end[ie::Y_AXIS] : padTop;

    int dilationX = deconvLayer->_dilation_x;
    int dilationY = deconvLayer->_dilation_y;

    if (dilationX != dilationY) {
        VPU_THROW_EXCEPTION << "kmb Deconvolution supports only equal dilationX and dilationY";
    }
    size_t groupSize = deconvLayer->_group;

    auto layerOutput = layer->outData[0];
    IE_ASSERT(layerOutput != nullptr);
    auto outDesc = layerOutput->getTensorDesc();
    // TODO: implement cvtPaddingsFromCeilToFloorMode for deconv, existing func does not suit

    mv::DType convolutionDataType("Default");
    mv::Data::TensorIterator mvDeconv;
    mv::Data::TensorIterator mvDeconvOnly;
    size_t inputGroupSize, outputGroupSize, stub;
    parseDims(input->desc(), stub, inputGroupSize, stub, stub);
    parseDims(outDesc, stub, outputGroupSize, stub, stub);

    bool isDepthWiseConv = groupSize > 1 && groupSize == inputGroupSize && groupSize == outputGroupSize;

    if (isDepthWiseConv) {
        /* TODO: Need align API in mcmCompiler
           mcm expects (1,*,*,*) shape for depthwise weights, but Openvino has a (*,1,*,*) */
        auto weights = layer->blobs["weights"];
        auto weightsData = packBlobToVector<double>(weights, weights->size());

        mv::Shape mcmShape = {static_cast<uint64_t>(kernelSizeY), static_cast<uint64_t>(kernelSizeX), groupSize, 1lu};

        auto mvWeightsValues = _modelMcm.constant(
            weightsData, mcmShape, precisionToDType(Precision::FP32), mv::Order::getZMajorID(mcmShape.ndims()));
        // TODO: Initially  this parameter is: precisionToDType(constBlob->getTensorDesc().getPrecision()),
        // but as Work Around it is set to: precisionToDType(Precision(Precision::ePrecision::FP32)).
        // It is so just because mcmCompiler has not supported FP16 yet.
        // Do not forget to redo it when support for FP16 will be available in mcmCompiler.
        mvWeightsValues->set<bool>("is_depthwise_weights", true);

        mvDeconv = _modelMcm.deconv(input->getMcmNode(), mvWeightsValues,
            {static_cast<uint16_t>(kernelStrideX), static_cast<uint16_t>(kernelStrideY)},
            {static_cast<uint16_t>(padLeft), static_cast<uint16_t>(padRight), static_cast<uint16_t>(padTop),
                static_cast<uint16_t>(padBottom)},
            static_cast<unsigned>(dilationX), static_cast<unsigned>(groupSize), true, convolutionDataType,
            initialQuantParams(), deconvLayer->name);
    } else {
        VPU_THROW_EXCEPTION << "Non depthwise Deconvolution layer is not supported by kmbPlugin";
    }

    //  Need quantize bias, this logic provide by MCM team, need check
    if (inputs.size() == 3) {
        mvDeconvOnly = mvDeconv;
        auto mvBiases = inputs[2]->getMcmNode();
        mvDeconv = _modelMcm.bias(
            mvDeconvOnly, mvBiases, mv::DType("Default"), initialQuantParams(), deconvLayer->name + ":bias");
        _logger->debug("'%s' layer '%s': Bias part (%s) added to mcmModel", deconvLayer->type, deconvLayer->name,
            mvDeconv->getName());
    }
    bindOutput(mvDeconv, layerOutput);

    _logger->debug(FINISH_PARSING_STR, mvDeconv->getName());
}

void FrontEndMcm::parseCopy(const ie::CNNLayerPtr&, const McmNodeVector&) {
    VPU_THROW_EXCEPTION << "Copy layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseELU(const ie::CNNLayerPtr&, const McmNodeVector&) {
    VPU_THROW_EXCEPTION << "ELU layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseCrop(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    auto input = inputs[0];
    auto cropLayer = std::dynamic_pointer_cast<ie::CropLayer>(layer);
    IE_ASSERT(cropLayer != nullptr);
    logParsingStartHelper(_logger, layer, {input});

    mv::Data::TensorIterator mvSlice;

    if (layer->CheckParamPresence("axis") && layer->CheckParamPresence("offset") &&
        !(layer->CheckParamPresence("dim"))) {
        // Crop type 1
        VPU_THROW_EXCEPTION << "Crop (Type 1) layer is not supported by kmbPlugin";
    } else if (layer->CheckParamPresence("axis") && layer->CheckParamPresence("offset") &&
               layer->CheckParamPresence("dim")) {
        // Crop type 2
        auto axisParam = layer->GetParamAsInts("axis");      // axis is the number of a dimension to crop
        auto offsetParam = layer->GetParamAsInts("offset");  // offset is the starting point for crop in the input blob
        auto dimParam =
            layer->GetParamAsInts("dim");  // dim is the resulting size of the output blob for the specified axis

        IE_ASSERT(axisParam.size() == offsetParam.size());
        const auto& outDataDesc = layer->outData[0]->getTensorDesc();
        mv::Shape outShape(getWHCN(outDataDesc).getDims());
        auto ndims = outShape.ndims();

        mv::Shape mvOffsets(ndims);
        mv::Shape mvOutDims(ndims);

        // fill offsets and out dimensions size with conversion NCHW->WHCN
        for (int i = 0; i < static_cast<int>(ndims); ++i) {
            mvOffsets[ndims - 1 - axisParam[i]] = offsetParam[i];
            mvOutDims[ndims - 1 - axisParam[i]] = dimParam[i];
        }
        // _modelMcm.crop is single dimensional and _modelMcm.slice is multdimensional
        mvSlice = _modelMcm.slice(input->getMcmNode(), mvOffsets, outShape, initialQuantParams(), layer->name);

        bindOutput(mvSlice, layer->outData[0]);
    } else if (layer->CheckParamPresence("axis") && layer->CheckParamPresence("crop_begin") &&
               layer->CheckParamPresence("crop_end")) {
        // Crop type 3
        VPU_THROW_EXCEPTION << "Crop (Type 3) layer is not supported by kmbPlugin";
    } else {
        VPU_THROW_EXCEPTION << "Unrecognized Crop layer type";
    }
    _logger->debug(FINISH_PARSING_STR, mvSlice->getName());
}

void FrontEndMcm::parseTile(const ie::CNNLayerPtr&, const McmNodeVector&) {
    VPU_THROW_EXCEPTION << "Tile layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseNormalize(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    IE_ASSERT(inputs.size() == 1);
    logParsingStartHelper(_logger, layer, inputs);

    double eps = layer->GetParamAsFloat("eps");
    bool across_spatial = layer->GetParamAsBool("across_spatial");
    bool channel_shared = layer->GetParamAsBool("channel_shared");

    ie::Blob::Ptr weightsBlob = nullptr;
    weightsBlob = layer->blobs["weights"];
    IE_ASSERT(weightsBlob != nullptr);

    auto dims = inputs[0]->desc().getDims();
    auto weightsSize = weightsBlob->size();
    mv::Shape weightsShape = {1, dims[1], 1, 1};

    IE_ASSERT((dims[1] == weightsSize) || (channel_shared == 1 && weightsSize == 1));

    auto weightsData = packBlobToVector<double>(weightsBlob, weightsSize);
    if (channel_shared) {
        weightsData.assign(dims[1], weightsData[0]);
        channel_shared = false;
    }

    auto mvWeightsValues = _modelMcm.constant(
        weightsData, weightsShape, mv::DType(precisionToDType(Precision::ePrecision::FP32)), mv::Order::getZMajorID(4));

    mv::Data::TensorIterator mvNormalize;

    mvNormalize = _modelMcm.normalize(inputs[0]->getMcmNode(), mvWeightsValues, eps, across_spatial, channel_shared,
        mv::DType("Default"), initialQuantParams(), layer->name);

    bindOutput(mvNormalize, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, mvNormalize->getName());
}

void FrontEndMcm::parseCTCDecoder(const ie::CNNLayerPtr&, const McmNodeVector&) {
    VPU_THROW_EXCEPTION << "CTCDecoder layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseInterp(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    IE_ASSERT(inputs.size() == 1);
    logParsingStartHelper(_logger, layer, inputs);
    auto factor = layer->GetParamAsFloat("factor", 1.0);
    auto height = layer->GetParamAsUInt("height", 0);
    auto width = layer->GetParamAsUInt("width", 0);
    auto pad_begin = layer->GetParamAsUInt("pads_begin", 0);
    auto pad_end = layer->GetParamAsUInt("pads_end", 0);
    auto align_corners = layer->GetParamAsInt("align_corners", 0) == 1;

    auto mvInterp = _modelMcm.interp(inputs[0]->getMcmNode(), factor, pad_begin, pad_end, height, width, align_corners,
        mv::DType("Default"), initialQuantParams(), layer->name);

    bindOutput(mvInterp, layer->outData[0]);
    _logger->debug(FINISH_PARSING_STR, mvInterp->getName());
}

void FrontEndMcm::parseProposal(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    auto feat_stride = static_cast<size_t>(layer->GetParamAsInt("feat_stride"));
    auto base_size = static_cast<size_t>(layer->GetParamAsInt("base_size"));
    auto min_size = static_cast<size_t>(layer->GetParamAsInt("min_size"));
    auto pre_nms_topn = layer->GetParamAsInt("pre_nms_topn");
    auto post_nms_topn = layer->GetParamAsInt("post_nms_topn");
    auto nms_thresh = layer->GetParamAsFloat("nms_thresh");
    auto box_coordinate_scale = layer->GetParamAsFloat("box_coordinate_scale", 1.0);
    auto box_size_scale = layer->GetParamAsFloat("box_size_scale", 1.0);
    auto scale = layer->GetParamAsFloats("scale", {});
    auto ratio = layer->GetParamAsFloats("ratio", {});
    auto normalize = layer->GetParamAsBool("normalize", false);
    auto clip_before_nms = layer->GetParamAsBool("clip_before_nms", true);
    auto clip_after_nms = layer->GetParamAsBool("clip_after_nms", false);
    auto framework = layer->GetParamAsString("framework", "caffe");
    auto for_deformable = layer->GetParamAsBool("for_deformable", false);
    // NB: IR doens't contain this parameter
    float pre_nms_thresh = 0.0f;

    if (framework == "tensorflow" || framework == "caffe") {
        std::transform(framework.begin(), framework.end(), framework.begin(), ::toupper);
    } else {
        VPU_THROW_EXCEPTION << "Proposal layer doesn't support framework: " << framework;
    }

    std::vector<mv::Data::TensorIterator> proposal_ins;
    for (const auto& input : inputs) {
        proposal_ins.push_back(input->getMcmNode());
    }
    IE_ASSERT(proposal_ins.size() == 3u);

    auto proposal = _modelMcm.proposal(proposal_ins, scale, ratio, base_size, pre_nms_topn, post_nms_topn, nms_thresh,
        feat_stride, min_size, pre_nms_thresh, clip_before_nms, clip_after_nms, normalize, box_size_scale,
        box_coordinate_scale, framework, for_deformable, mv::DType("Default"));

    bindOutput(proposal, layer->outData[0]);
}

void FrontEndMcm::parseROIPooling(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    IE_ASSERT(2 == inputs.size());
    unsigned pooled_h = static_cast<unsigned>(layer->GetParamAsInt("pooled_h"));
    unsigned pooled_w = static_cast<unsigned>(layer->GetParamAsInt("pooled_w"));
    double spatial_scale = layer->GetParamAsFloat("spatial_scale", 0.0625f);
    std::string method = layer->GetParamAsString("method", "max");
    unsigned roi_pooling_method = (method == "bilinear") ? 1 : 0;

    std::vector<mv::Data::TensorIterator> roi_inputs;
    for (const auto& input : inputs) {
        roi_inputs.push_back(input->getMcmNode());
    }

    size_t n_roi, stub;
    parseDims(inputs.at(1)->desc(), n_roi, stub, stub, stub);
    unsigned num_rois = static_cast<unsigned int>(n_roi);

    auto roipool = _modelMcm.rOIPooling(roi_inputs, pooled_w, pooled_h, spatial_scale, roi_pooling_method, num_rois,
        mv::DType("Default"), initialQuantParams(), layer->name);

    bindOutput(roipool, layer->outData[0]);
}

void FrontEndMcm::parsePSROIPooling(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    auto output_dim = static_cast<size_t>(layer->GetParamAsInt("output_dim"));
    auto group_size = static_cast<size_t>(layer->GetParamAsInt("group_size"));
    auto spatial_scale = layer->GetParamAsFloat("spatial_scale");
    auto pooled_h = static_cast<size_t>(layer->GetParamAsInt("pooled_height", static_cast<int>(group_size)));
    auto pooled_w = static_cast<size_t>(layer->GetParamAsInt("pooled_width", static_cast<int>(group_size)));
    auto spatial_bins_x = static_cast<size_t>(layer->GetParamAsInt("spatial_bins_x", 1));
    auto spatial_bins_y = static_cast<size_t>(layer->GetParamAsInt("spatial_bins_y", 1));
    auto mode = layer->GetParamAsString("mode", "average");

    std::vector<mv::Data::TensorIterator> psroi_ins;
    for (const auto& input : inputs) {
        psroi_ins.push_back(input->getMcmNode());
    }

    auto psroi = _modelMcm.pSROIPooling(psroi_ins, output_dim, group_size, spatial_scale, pooled_h, pooled_w,
        spatial_bins_x, spatial_bins_y, mode, mv::DType("Default"), initialQuantParams(), layer->name);

    bindOutput(psroi, layer->outData[0]);
}

void FrontEndMcm::parseCustom(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    logParsingStartHelper(_logger, layer, inputs);
    IE_ASSERT(layer != nullptr);

    const auto customLayer = [&] {
        const auto customLayersForType = _customLayers.find(layer->type);
        IE_ASSERT(customLayersForType != _customLayers.end());
        const auto suitableLayers = getSuitableCustomLayers(customLayersForType->second, layer);
        IE_ASSERT(!suitableLayers.empty());
        // inputs there are always in HWC layout
        // this check should be moved to mcm side somehow
        return findMatchingCustomLayer(suitableLayers, inputs);
    }();

    auto parser = CustomLayerParser{layer, inputs};

    int stageIdx = 0;
    for (const auto& kernel : customLayer->kernels()) {
        const auto sortedKernelBindings = [&] {
            auto bindings = std::vector<CustomKernel::BindingParameter>{};
            bindings.reserve(kernel.arguments().size());

            for (const auto& arg : kernel.arguments()) {
                const auto& binding = kernel.bindings().find(arg.name);
                VPU_THROW_UNLESS(binding != kernel.bindings().end(),
                    "Failed to bind '%s' custom layer. "
                    "Can't find kernel argument '%s' in binding list.",
                    customLayer->layerName(), arg.name);
                bindings.push_back(binding->second);
            }

            return bindings;
        }();

        const auto stage = parser.parseKernelArguments(sortedKernelBindings);

        const auto kernelData = parser.resolveKernelArguments(kernel, stage.arguments);
        const auto stageOutputs = parser.resolveStageOutputs(kernel, *customLayer, stage.outputs);

        const auto layerName = layer->name + "_Custom:" + std::to_string(stageIdx);

        auto custom = _modelMcm.custom(stage.inputs, kernel.kernelBinary(), kernelData, stageOutputs,
            mv::DType{"Default"}, initialQuantParams(), layerName);

        const auto sourceOp = _modelMcm.getSourceOp(custom);
        const auto mcmOutputTensors = sourceOp->getOutputTensor();

        IE_ASSERT(stage.outputs.size() == mcmOutputTensors.size());

        for (size_t i = 0; i < stage.outputs.size(); i++) {
            const auto& output = stage.outputs[i];
            if (output.isBuffer) {
                parser.addBuffer(output.portIndex, mcmOutputTensors[i]);
            } else {
                bindOutput(mcmOutputTensors[i], layer->outData[output.portIndex]);
            }
        }

        _logger->debug(FINISH_PARSING_STR, custom->getName() + "_stage#" + std::to_string(stageIdx++));
    }
}

void FrontEndMcm::parseMTCNN(const ie::CNNLayerPtr&, const McmNodeVector&) {
    VPU_THROW_EXCEPTION << "MTCNN layer is not supported by kmbPlugin";
}

void FrontEndMcm::parsePad(const ie::CNNLayerPtr&, const McmNodeVector&) {
    VPU_THROW_EXCEPTION << "Pad layer is not supported by kmbPlugin";
}

const static std::map<std::string, std::string> interpolationMap = {
    {"caffe.ResampleParameter.NEAREST", "NEAREST"},
    {"caffe.ResampleParameter.CUBIC", "BICUBIC"},
    {"caffe.ResampleParameter.LINEAR", "BILINEAR"},
};

void FrontEndMcm::parseResample(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    logParsingStartHelper(_logger, layer, inputs);

    auto antialias = false;
    auto interpolation = layer->GetParamAsString("type", "caffe.ResampleParameter.NEAREST");

    auto layerOutput = layer->outData[0];
    IE_ASSERT(layerOutput != nullptr);
    mv::Shape output_shape(getWHCN(layerOutput->getTensorDesc()).getDims());

    auto resample_result = _modelMcm.resample(inputs[0]->getMcmNode(), interpolationMap.at(interpolation), antialias,
        output_shape, mv::DType("Default"), initialQuantParams(), layer->name);

    bindOutput(resample_result, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, resample_result->getName());
}

void FrontEndMcm::parseLSTMCell(const ie::CNNLayerPtr&, const McmNodeVector&) {
    VPU_THROW_EXCEPTION << "LSTMCell layer is not supported by kmbPlugin";
}

void FrontEndMcm::parsePriorBox(const ie::CNNLayerPtr& layer, const McmNodeVector&) {
    if (layer->insData.size() != 2 || layer->outData.empty())
        THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

    if (layer->insData[0].lock()->getTensorDesc().getDims().size() != 4 ||
        layer->insData[1].lock()->getTensorDesc().getDims().size() != 4)
        THROW_IE_EXCEPTION << "PriorBox supports only 4D blobs!";
    auto& dataMemPtr = layer->insData[0];
    auto& imageMemPtr = layer->insData[1];
    auto& dstMemPtr = layer->outData[0];
    SizeVector data_dims = dataMemPtr.lock()->getTensorDesc().getDims();
    SizeVector image_dims = imageMemPtr.lock()->getTensorDesc().getDims();
    SizeVector out_dims = dstMemPtr->getTensorDesc().getDims();

    float offset = layer->GetParamAsFloat("offset");
    float step = layer->GetParamAsFloat("step", 0.f);
    std::vector<float> min_sizes = layer->GetParamAsFloats("min_size", {});
    std::vector<float> max_sizes = layer->GetParamAsFloats("max_size", {});
    bool flip = layer->GetParamAsBool("flip", false);
    bool clip = layer->GetParamAsBool("clip", false);
    bool scale_all_sizes = layer->GetParamAsBool("scale_all_sizes", true);

    std::vector<float> fixed_sizes = layer->GetParamAsFloats("fixed_size", {});
    std::vector<float> fixed_ratios = layer->GetParamAsFloats("fixed_ratio", {});
    std::vector<float> densitys = layer->GetParamAsFloats("density", {});

    const std::vector<float> src_aspect_ratios = layer->GetParamAsFloats("aspect_ratio", {});
    const std::vector<float> src_variance = layer->GetParamAsFloats("variance", {});

    KmbPlugin::utils::priorBoxParam param(offset, step, min_sizes, max_sizes, flip, clip, scale_all_sizes, fixed_sizes,
        fixed_ratios, densitys, src_aspect_ratios, src_variance, data_dims, image_dims, out_dims);

    auto boxes = KmbPlugin::utils::computePriorbox(param);
    auto priorbox = _modelMcm.constant(boxes, {boxes.size() / 2, 2, 1, 1}, mv::DType("Float64"), mv::Order("NHWC"),
        initialQuantParams(), layer->name + "_const");

    bindOutput(priorbox, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, priorbox->getName());
}

void FrontEndMcm::parsePriorBoxClustered(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    UNUSED(inputs);
    if (layer->insData.size() != 2 || layer->outData.empty()) {
        THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";
    }

    if (layer->insData[0].lock()->getTensorDesc().getDims().size() != 4 ||
        layer->insData[1].lock()->getTensorDesc().getDims().size() != 4) {
        THROW_IE_EXCEPTION << "PriorBoxClustered supports only 4D blobs!";
    }

    if (layer->outData.size() != 1) {
        THROW_IE_EXCEPTION << "PriorBoxClustered must have only one output";
    }

    const int clip = layer->GetParamAsInt("clip");
    const int img_h = layer->GetParamAsInt("img_h", 0);
    const int img_w = layer->GetParamAsInt("img_w", 0);
    std::vector<float> widths = layer->GetParamAsFloats("width", {});
    std::vector<float> heights = layer->GetParamAsFloats("height", {});
    const float step = layer->GetParamAsFloat("step", 0);
    const float offset = layer->GetParamAsFloat("offset");

    float step_w = layer->GetParamAsFloat("step_w", step);
    float step_h = layer->GetParamAsFloat("step_h", step);

    int img_width = layer->insData[1].lock()->getTensorDesc().getDims()[3];
    int img_height = layer->insData[1].lock()->getTensorDesc().getDims()[2];
    img_width = img_w == 0 ? img_width : img_w;
    img_height = img_h == 0 ? img_height : img_h;

    int layer_width = layer->insData[0].lock()->getTensorDesc().getDims()[3];
    int layer_height = layer->insData[0].lock()->getTensorDesc().getDims()[2];

    IE_ASSERT(widths.size() == heights.size());
    int num_priors = widths.size();

    std::vector<float> variance = layer->GetParamAsFloats("variance", {});
    if (variance.empty()) {
        variance.push_back(0.1f);
    }

    if (step_w == 0.f && step_h == 0.f) {
        if (step == 0.f) {
            step_w = static_cast<float>(img_width) / layer_width;
            step_h = static_cast<float>(img_height) / layer_height;
        } else {
            step_w = step;
            step_h = step;
        }
    }

    const auto& dims = layer->outData.front()->getDims();

    IE_ASSERT(dims.size() == 3);
    int size = dims[0] * dims[1] * dims[2];

    KmbPlugin::utils::priorBoxClusteredParam param{offset, clip, step_w, step_h, layer_width, layer_height, img_width,
        img_height, num_priors, std::move(widths), std::move(heights), std::move(variance), size};

    auto boxes = computePriorboxClustered(param);

    auto priorboxClustered =
        _modelMcm.constant(boxes, {boxes.size() / 2, 2, 1, 1}, mv::DType("Float64"), mv::Order("NHWC"));

    bindOutput(priorboxClustered, layer->outData[0]);
}

void FrontEndMcm::parseSplit(const ie::CNNLayerPtr&, const McmNodeVector&) {
    VPU_THROW_EXCEPTION << "Split layer is not supported by kmbPlugin";
}

std::vector<CustomLayer::Ptr> FrontEndMcm::getSuitableCustomLayers(
    const std::vector<CustomLayer::Ptr>& customLayers, const ie::CNNLayerPtr& cnnLayer) {
    _logger->debug("Check for suitable custom implementation for layer %s:%s", cnnLayer->name, cnnLayer->type);

    const auto isSuitableLayer = [&](const CustomLayer::Ptr& customLayer) {
        _logger->debug("Check custom layer : %v", customLayer->layerName());

        if (!customLayer->meetsWhereRestrictions(cnnLayer->params)) {
            _logger->debug("Not suitable: 'Where' restrictions are not met");
            return false;
        }

        for (const auto& kernel : customLayer->kernels()) {
            const auto& gws = kernel.globalGridSizeRules();
            const auto& lws = kernel.localGridSizeRules();

            const auto validSizeRule = [&](const std::string& rule) {
                return CustomLayer::isLegalSizeRule(rule, cnnLayer->params);
            };

            const auto validGridSizes =
                std::all_of(begin(gws), end(gws), validSizeRule) && std::all_of(begin(lws), end(lws), validSizeRule);

            const auto workGroupDims = 3;
            VPU_THROW_UNLESS(lws.size() <= workGroupDims,
                "Failed to parse '%s' custom layer binding list. Local work group size count "
                "is greater than 3.",
                customLayer->layerName());
            VPU_THROW_UNLESS(gws.size() <= workGroupDims,
                "Failed to parse '%s' custom layer binding list. Global work group size count "
                "is greater than 3.",
                customLayer->layerName());

            if (!validGridSizes) {
                _logger->debug("Not suitable: Work group grid sizes are not valid");
                return false;
            }
        }

        return true;
    };

    auto suitableCustomLayers = SmallVector<CustomLayer::Ptr>{};

    std::copy_if(begin(customLayers), end(customLayers), back_inserter(suitableCustomLayers), isSuitableLayer);

    if (suitableCustomLayers.empty()) {
        _logger->debug("Suitable custom layer is not found");
        return {};
    }

    _logger->debug("Found %d suitable custom layers", suitableCustomLayers.size());
    return suitableCustomLayers;
}

CustomLayer::Ptr FrontEndMcm::findMatchingCustomLayer(
    const std::vector<CustomLayer::Ptr>& customLayers, const McmNodeVector& inputs) {
    const auto inputsLayoutMatch = [&](const SmallVector<mv::Order>& cnnEdges, const std::map<int, Layout>& clEdges) {
        for (const auto clEdge : clEdges) {
            size_t port = clEdge.first;
            VPU_THROW_UNLESS(
                port < cnnEdges.size(), "Can't bind custom layer edge with port '%s' to CNNNetwork layer", port);

            const auto clFormat = clEdge.second;
            const auto cnnFormat = cnnEdges[port];
            if (cnnFormat != layoutToOrder(clFormat) && clFormat != Layout::ANY) {
                return false;
            }
        }
        return true;
    };

    const auto mcmInputs = [&] {
        auto mcmInputs = SmallVector<mv::Order>{};
        mcmInputs.reserve(inputs.size());
        for (const auto& input : inputs) {
            const auto layout = input->getMcmNode()->getOrder();
            mcmInputs.push_back(layout);
        }
        return mcmInputs;
    }();

    for (const auto& customLayer : customLayers) {
        const auto clInputs = customLayer->inputs();

        if (inputsLayoutMatch(mcmInputs, clInputs)) {
            _logger->debug("Found suitable '%s' custom layer", customLayer->layerName());
            return customLayer;
        }
    }

    const auto firstGoodLayer = customLayers.front();
    _logger->debug("Found suitable custom layer '%s'. Input layouts "
                   "do not match up with what CNNNetwork expected",
        firstGoodLayer->layerName());
    return firstGoodLayer;
}

}  // namespace vpu
#endif
