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

#include <graph_transformer.h>
#include <precision_utils.h>

#include <algorithm>
#include <blob_factory.hpp>
#include <frontend_mcm.hpp>
#include <graph_tools.hpp>
#include <ie_profiling.hpp>
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

#ifndef UNUSED
#define UNUSED(var) (void)var
#endif

#ifdef ENABLE_MCM_COMPILER

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
                {"FakeQuantize",       &FrontEndMcm::parseFakeQuantize},
                {"Const",              &FrontEndMcm::parseConst},
        };

mv::DType convert_data_type(const ie::Precision& iePrecision) {
    mv::DType mvType;
    switch (iePrecision) {
    case ie::Precision::UNSPECIFIED:
        mvType = mv::DType("Default");
        break;
    case ie::Precision::I8:
        mvType = mv::DType("Int8");
        break;
    case ie::Precision::U8:
        mvType = mv::DType("UInt8");
        break;
    case ie::Precision::I32:
        mvType = mv::DType("Int32");
        break;
    case ie::Precision::I64:
        mvType = mv::DType("Int64");
        break;
    case ie::Precision::FP16:
        mvType = mv::DType("Float16");
        break;
    case ie::Precision::FP32:
        mvType = mv::DType("Float32");
        break;
    default:
        VPU_THROW_EXCEPTION << "Data type handling is not implemented" << iePrecision.name();
    }
    return mvType;
}

mv::Order convert_layout(const ie::Layout& ieLayout) {
    std::ostringstream layoutToOrder;
    layoutToOrder << ieLayout;
    return mv::Order(layoutToOrder.str());
}

// clang-format on

}  // namespace

void FrontEndMcm::buildInitialModel(ie::ICNNNetwork& network) {
    runCommonPasses(network);
    for (const auto& layer : _parsedNetwork.orderedLayers) {
        IE_ASSERT(layer != nullptr);
        _logger->debug("Try to parse layer %s", layer->name);

        auto it = g_mcm_parsers.find(layer->type);
        if (it == g_mcm_parsers.end()) {
            VPU_THROW_EXCEPTION << "Cannot convert layer \"" << layer->name << "\" due to unsupported layer type \""
                                << layer->type << "\"";
        }

        auto parser = it->second;
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
    IE_PROFILING_AUTO_SCOPE(parseNetworkDFS);

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
        for (const auto& consumer : inputData->getInputTo()) {
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

std::vector<CNNLayerPtr> getInputsFQ(const CNNLayer& layer) {
    std::vector<CNNLayerPtr> result;

    auto inputs = CNNNetworkHelper::getParents(layer);
    for (auto& input : inputs) {
        if ((input->type == "FakeQuantize") && (CNNNetworkHelper::getParent(*input)->type != "Const")) {
            result.push_back(input);
        } else {
            auto parentInputs = getInputsFQ(*input);
            result.insert(result.end(), parentInputs.begin(), parentInputs.end());
        }
    }

    return result;
}

void FrontEndMcm::alignEltwiseScales(ie::CNNNetwork& network) {
    for (auto& layer : network) {
        if (layer->type == "Eltwise") {
            auto inputs = getInputsFQ(*layer);
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

void FrontEndMcm::alignConcatScales(ie::CNNNetwork& network) {
    for (auto& layer : network) {
        if (layer->type == "Concat") {
            auto inputs = getInputsFQ(*layer);
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
    auto parentLayer = inputData->getCreatorLayer().lock();

    //  Check that FQ on weights
    return parentLayer->type == "Const" ? true : false;
}
}  // namespace

void FrontEndMcm::alignZeroPointsOnWeights(ie::CNNNetwork& network) {
    for (auto& layer : network) {
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

    if (_config.eltwiseScalesAlignment()) {
        alignEltwiseScales(cnnNet);
    }
    if (_config.concatScalesAlignment()) {
        alignConcatScales(cnnNet);
    }
    if (_config.zeroPointsOnWeightsAlignment()) {
        alignZeroPointsOnWeights(cnnNet);
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

std::unordered_map<int, char> DIM_NAMES({{3, 'W'}, {2, 'H'}, {1, 'C'}, {0, 'N'}});

constexpr char FINISH_PARSING_STR[] = "Parsed to mcmModel as '%s";

void logParsingStartHelper(Logger::Ptr logger, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    logger->debug("Start parsing '%s' layer: '%s'", layer->type, layer->name);

    if (inputs.empty()) {
        logger->debug("Layer has no input");
    } else {
        for (size_t i = 0; i < inputs.size(); ++i)
            logger->debug("Layer input %d: '%s'", i, inputs[0]->getMcmNode()->getName());
    }
}

double inf = std::numeric_limits<double>::infinity();
mv::QuantizationParams initialQuantParams = {{0}, {1}, {-inf}, {inf}};

bool isInputPrecisionSupported(const ie::Precision& inputPrecision) {
    const std::set<ie::Precision> supportedInPrecisions = {ie::Precision::U8};
    return supportedInPrecisions.find(inputPrecision) != supportedInPrecisions.end();
}

bool isInputLayoutSupported(const ie::Layout& inputLayout) {
    const std::set<ie::Layout> supportedInLayouts = {ie::Layout::NHWC};
    return supportedInLayouts.find(inputLayout) != supportedInLayouts.end();
}

bool isOutputPrecisionSupported(const ie::Precision& outputPrecision, bool allowFP32Output) {
    std::set<ie::Precision> supportedOutPrecisions = {ie::Precision::U8, ie::Precision::FP16};
    if (allowFP32Output) {
        supportedOutPrecisions.insert(ie::Precision::FP32);
    }
    return supportedOutPrecisions.find(outputPrecision) != supportedOutPrecisions.end();
}

bool isOutputLayoutSupported(const ie::Layout& outputLayout, bool allowNCOutput) {
    std::set<ie::Layout> supportedOutLayouts = {ie::Layout::NHWC};
    if (allowNCOutput) {
        supportedOutLayouts.insert(ie::Layout::NC);
    }
    return supportedOutLayouts.find(outputLayout) != supportedOutLayouts.end();
}

void FrontEndMcm::parseInputData() {
    _logger->debug("Try to parse network input");

    if (_parsedNetwork.networkInputs.size() != 1) {
        THROW_IE_EXCEPTION << "Only single input is supported currently";
    }

    for (const auto& inputInfo : _parsedNetwork.networkInputs) {
        auto netInput = inputInfo.second;
        IE_ASSERT(netInput != nullptr);

        auto ieData = netInput->getInputData();
        IE_ASSERT(ieData != nullptr);

        const auto& dataDesc = ieData->getTensorDesc();
        mv::Shape inputShape(getWHCN(dataDesc).getDims());

        auto inputLayerPtr = ieData->getCreatorLayer().lock();

        const InferenceEngine::Layout inputLayout = ieData->getTensorDesc().getLayout();
        if (!isInputLayoutSupported(inputLayout)) {
            VPU_THROW_EXCEPTION << "Input layout is not supported: " << ieData->getTensorDesc().getLayout();
        }

        const InferenceEngine::Precision inputPrecision = ieData->getTensorDesc().getPrecision();
        if (!isInputPrecisionSupported(inputPrecision)) {
            VPU_THROW_EXCEPTION << "Input data type is not supported: " << ieData->getTensorDesc().getPrecision();
        }

        auto mvInput = _modelMcm.input(inputShape, convert_data_type(inputPrecision), convert_layout(inputLayout),
            initialQuantParams, netInput->name());
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
        if (!isOutputPrecisionSupported(outputPrecision, _config.allowFP32Output())) {
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
        // NC outputs are not supported by MCM, but the output can be casted to NC via VPU_COMPILER_ALLOW_NC_OUTPUT
        if (!isOutputLayoutSupported(outputLayout, _config.allowNCOutput())) {
            VPU_THROW_EXCEPTION << "Output layout is not supported: " << outputLayout;
        }

        auto mvOutput = _modelMcm.output(lastLayerOut->getMcmNode(), outputType, {{}, {}, {}, {}});
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
        auto newWeightsShape = {
            static_cast<std::size_t>(kernelSizeX), static_cast<std::size_t>(kernelSizeY), inputGroupSize, 1lu};

        constWeightTensor->setShape(newWeightsShape);
        mvWeights->setShape(newWeightsShape);

        mvConv = _modelMcm.depthwiseConv(input->getMcmNode(), mvWeights,
            {static_cast<uint16_t>(kernelStrideX), static_cast<uint16_t>(kernelStrideY)},
            {static_cast<uint16_t>(padLeft), static_cast<uint16_t>(padRight), static_cast<uint16_t>(padTop),
                static_cast<uint16_t>(padBottom)},
            static_cast<unsigned>(dilationX), convolutionDataType, initialQuantParams, convLayer->name);
    } else {
        mvConv = _modelMcm.conv(input->getMcmNode(), mvWeights,
            {static_cast<uint16_t>(kernelStrideX), static_cast<uint16_t>(kernelStrideY)},
            {static_cast<uint16_t>(padLeft), static_cast<uint16_t>(padRight), static_cast<uint16_t>(padTop),
                static_cast<uint16_t>(padBottom)},
            static_cast<unsigned>(dilationX), static_cast<unsigned>(groupSize), convolutionDataType, initialQuantParams,
            convLayer->name);
    }

    //  Need quantize bias, this logic provide by MCM team, need check
    if (inputs.size() == 3) {
        mvConvOnly = mvConv;
        auto mvBiases = inputs[2]->getMcmNode();
        mvConv =
            _modelMcm.bias(mvConvOnly, mvBiases, mv::DType("Default"), initialQuantParams, convLayer->name + ":bias");
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
            poolLayer->_exclude_pad, mv::DType("Default"), initialQuantParams, poolLayer->name);
    } else {
        mvPooling = _modelMcm.maxPool(inputs[0]->getMcmNode(),
            {static_cast<uint16_t>(kernelSizeX), static_cast<uint16_t>(kernelSizeY)},
            {static_cast<uint16_t>(kernelStrideX), static_cast<uint16_t>(kernelStrideY)},
            {static_cast<uint16_t>(padLeft), static_cast<uint16_t>(padRight), static_cast<uint16_t>(padTop),
                static_cast<uint16_t>(padBottom)},
            poolLayer->_exclude_pad, mv::DType("Default"), initialQuantParams, poolLayer->name);
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
        _modelMcm.fullyConnected(input->getMcmNode(), mvWeights, layerDataType, initialQuantParams, FClayer->name);

    if (inputs.size() == 3) {
        auto mvBiases = inputs[2]->getMcmNode();
        mvFullyConnected = _modelMcm.bias(
            mvFullyConnected, mvBiases, mv::DType("Default"), initialQuantParams, FClayer->name + ":bias");
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
        mvRelu = _modelMcm.relu(inputs[0]->getMcmNode(), mv::DType("Default"), initialQuantParams, reluLayer->name);
    } else {
        mvRelu = _modelMcm.leakyRelu(
            inputs[0]->getMcmNode(), negativeSlope, mv::DType("Default"), initialQuantParams, reluLayer->name);
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

    std::string mcmAxis;
    mcmAxis = mcmAxis + DIM_NAMES[softMaxLayer->axis];
    auto mvSoftmax = _modelMcm.softmax(
        inputs[0]->getMcmNode(), mcmAxis, mv::DType("Default"), initialQuantParams, softMaxLayer->name);

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
        initialQuantParams, normLayer->name);

    bindOutput(mvLRN, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, mvLRN->getName());
}

void FrontEndMcm::parseScaleImpl(
    const ie::CNNLayerPtr& layer, const McmNodeVector& inputs, std::vector<double>& weights, ie::Blob::Ptr biases) {
    logParsingStartHelper(_logger, layer, inputs);

    auto input = inputs[0];

    std::vector<double> quantizeScale;

    mv::Shape weightsShape = {weights.size()};
    auto mvWeights = _modelMcm.constant(
        weights, weightsShape, mv::DType("Float32"), mv::Order::getColMajorID(1), initialQuantParams);

    auto scale = _modelMcm.scale(input->getMcmNode(), mvWeights, mv::DType("Default"), initialQuantParams, layer->name);
    auto scaleShift = scale;

    std::vector<double> biasData;
    if (biases != nullptr) {
        biasData = packBlobToVector<double>(biases, biases->size());
        mv::Shape shiftShape {biases->size()};
        auto shiftData = _modelMcm.constant(
            biasData, shiftShape, mv::DType("Float32"), mv::Order::getColMajorID(1), initialQuantParams);
        // TODO: return logging
        scaleShift = _modelMcm.bias(scale, shiftData, mv::DType("Default"), initialQuantParams, layer->name + ":bias");
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

    //  4d NCHW inputs are supported
    for (size_t i = 0; i < ieOrder.size(); i++) {
        newOrder += DIM_NAMES[ieOrder[ieOrder.size() - 1 - i]];
    }

    auto mvPerm = _modelMcm.permute(
        inputs[0]->getMcmNode(), mv::Order(newOrder), mv::DType("Default"), initialQuantParams, layer->name);

    // Workaround to avoid parsing stage crash:
    // 'ArgumentError: attribute identifer quantParams - Undefined identifier'
    // [Track number: D#2284, D#2237]
    mvPerm->set<mv::QuantizationParams>("quantParams", initialQuantParams);

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
                            << "Eltwise Sub operations with with more than 2 operands is not supported by kmbPlugin";
    }

    for (size_t i = 0; i < eltwiseLayer->coeff.size(); ++i) {
        if (std::abs(eltwiseLayer->coeff[i]) != 1.0f) {
            VPU_THROW_EXCEPTION << eltwiseLayer->name
                                << " Eltwise Sum/Sub operations with such coefficients is not supported by kmbPlugin";
        }
    }

    switch (eltwiseLayer->_operation) {
    case ie::EltwiseLayer::eOperation::Sub:
        mvEltwise =
            _modelMcm.eltwise(mvInputs, "Subtract", mv::DType("Default"), initialQuantParams, eltwiseLayer->name);
        break;
    case ie::EltwiseLayer::eOperation::Sum:
        mvEltwise = _modelMcm.eltwise(mvInputs, "Add", mv::DType("Default"), initialQuantParams, eltwiseLayer->name);
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
            _modelMcm.bias(input->getMcmNode(), mvBiasValues, mv::DType("Default"), initialQuantParams, layer->name);
    } else if (inputs.size() == 2) {
        logParsingStartHelper(_logger, layer, inputs);

        auto input = inputs[0];
        auto input1 = inputs[1];
        mvBias = _modelMcm.bias(
            input->getMcmNode(), input1->getMcmNode(), mv::DType("Default"), initialQuantParams, layer->name);
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
        initialQuantParams, clampLayer->name + "clamp-min");
    auto mvClampMax = _modelMcm.maximum(
        mvClampMin, clampLayer->min_value, mv::DType("Default"), initialQuantParams, clampLayer->name + "clamp-max");
    bindOutput(mvClampMax, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, mvClampMax->getName());
}

void FrontEndMcm::parseReshape(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    auto layerOutput = layer->outData[0];
    IE_ASSERT(layerOutput != nullptr);
    for (size_t i = 1; i < inputs.size(); i++) {
        _modelMcm.removeOp(_modelMcm.getSourceOp(inputs[i]->getMcmNode()));
    }
    logParsingStartHelper(_logger, layer, inputs);

    // Because mcmCompiler supports only "dense" layouts
    // for example NC should be represented as NCHW with dims NC11
    // Formation of a newShape, "dense" shape with 1, substituted in the places of non-existent measurements
    // TODO: Tests on parsing/compilation of different cases of reshape should be added: Jira: CVS-20409
    // McmCompiler accept only input in WHCN format
    mv::Shape newShape(getWHCN(layerOutput->getTensorDesc()).getDims());
    auto mvReshape =
        _modelMcm.reshape(inputs[0]->getMcmNode(), newShape, mv::DType("Default"), initialQuantParams, layer->name);

    bindOutput(mvReshape, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, mvReshape->getName());
}

void FrontEndMcm::parseConcat(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    IE_ASSERT(!inputs.empty());

    auto concatLayer = std::dynamic_pointer_cast<ie::ConcatLayer>(layer);
    IE_ASSERT(concatLayer != nullptr);
    IE_ASSERT(concatLayer->_axis < inputs[0]->desc().getDims().size());

    logParsingStartHelper(_logger, layer, inputs);

    std::string mcmAxis;
    mcmAxis = mcmAxis + DIM_NAMES[concatLayer->_axis];
    std::vector<mv::Data::TensorIterator> concatInputs;

    for (const auto& input : inputs) {
        concatInputs.push_back(input->getMcmNode());
    }

    auto mvConcat =
        _modelMcm.concat(concatInputs, mcmAxis, mv::DType("Default"), initialQuantParams, concatLayer->name);
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

    auto region = _modelMcm.regionYolo(inputs[0]->getMcmNode(), coords, classes, do_softmax, num, {},
        mv::DType("Default"), initialQuantParams, layer->name);
    bindOutput(region, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, region->getName());
}

void FrontEndMcm::parseReorgYolo(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    IE_ASSERT(inputs.size() == 1);

    logParsingStartHelper(_logger, layer, inputs);

    auto stride = layer->GetParamAsUInt("stride");

    auto reorg =
        _modelMcm.reorgYolo(inputs[0]->getMcmNode(), stride, mv::DType("Default"), initialQuantParams, layer->name);
    bindOutput(reorg, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, reorg->getName());
}

InferenceEngine::CNNLayerPtr getInputLayerSafe(const InferenceEngine::CNNLayerPtr& layer, const size_t index) {
    IE_ASSERT(index < layer->insData.size());
    auto inputData = layer->insData[index].lock();
    IE_ASSERT(inputData != nullptr);
    auto inputLayer = inputData->getCreatorLayer().lock();
    IE_ASSERT(inputLayer != nullptr);
    return inputLayer;
}

namespace {
bool isInteger(ie::Precision iePrecision) {
    static std::set<ie::Precision> integer_precision {
        ie::Precision::I8, ie::Precision::U8, ie::Precision::I32, ie::Precision::I64};
    return integer_precision.count(iePrecision);
}

mv::Shape calculateMcmShape(const SizeVector dims) {
    if (dims.size() == 0) {
        return mv::Shape({1});
    } else {
        auto newShapes = dims;
        std::reverse(newShapes.begin(), newShapes.end());
        return mv::Shape(newShapes);
    }
}
}  // namespace

void FrontEndMcm::parseConst(const InferenceEngine::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    IE_ASSERT(layer->type == "Const");
    const auto constBlob = layer->blobs.begin()->second;
    auto blobPrecision = constBlob->getTensorDesc().getPrecision();
    auto mcmShape = calculateMcmShape(layer->outData.front()->getDims());
    if (isInteger(blobPrecision)) {
        std::vector<int64_t> constData = packBlobToVector<int64_t>(constBlob, constBlob->size());
        auto constMCM =
            _modelMcm.constantInt(constData, mcmShape, convert_data_type(constBlob->getTensorDesc().getPrecision()),
                mv::Order::getColMajorID(mcmShape.ndims()), initialQuantParams, layer->name);
        bindOutput(constMCM, layer->outData[0]);
    } else {
        std::vector<double> constData = packBlobToVector<double>(constBlob, constBlob->size());
        auto constMCM =
            _modelMcm.constant(constData, mcmShape, convert_data_type(Precision(Precision::ePrecision::FP32)),
                // Initially  this parameter is: convert_data_type(constBlob->getTensorDesc().getPrecision()),
                // but as Work Around it is set to: convert_data_type(Precision(Precision::ePrecision::FP32)).
                // It is so just because mcmCompiler has not supported FP16 yet.
                // Do not forget to redo it when support for FP16 will be available in mcmCompiler.
                mv::Order::getColMajorID(mcmShape.ndims()), initialQuantParams, layer->name);
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
        inputs[2]->getMcmNode(), inputs[3]->getMcmNode(), inputs[4]->getMcmNode(), levels);
    bindOutput(fakeQuantize, layer->outData[0]);
}

void FrontEndMcm::parseArgMax(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "ArgMax layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseGRN(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "GRN layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseMVN(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
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
    bool clip_before_nms = 0;
    bool clip_after_nms = 0;
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
        mv::DType("Default"), initialQuantParams, layer->name);

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

void FrontEndMcm::parseTanH(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "TanH layer is not supported by kmbPlugin";
}

void FrontEndMcm::parsePReLU(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "PReLU layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseBatchNorm(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "PReLU layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseDeconvolution(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    UNUSED(inputs);
    // TODO: Leyer can be with bias
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "Deconvolution layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseCopy(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "Copy layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseELU(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "ELU layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseCrop(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "Crop layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseTile(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
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

    auto weightsPrecision = weightsBlob->getTensorDesc().getPrecision();
    auto weightsData = packBlobToVector<double>(weightsBlob, weightsSize);
    if (channel_shared) {
        weightsData.assign(dims[1], weightsData[0]);
        channel_shared = false;
    }

    auto mvWeightsValues = _modelMcm.constant(
        weightsData, weightsShape, mv::DType(convert_data_type(weightsPrecision)), mv::Order::getZMajorID(4));

    mv::Data::TensorIterator mvNormalize;

    mvNormalize = _modelMcm.normalize(inputs[0]->getMcmNode(), mvWeightsValues, eps, across_spatial, channel_shared,
        mv::DType("Default"), initialQuantParams, layer->name);

    bindOutput(mvNormalize, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, mvNormalize->getName());
}

void FrontEndMcm::parseCTCDecoder(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "CTCDecoder layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseInterp(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "Interp layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseProposal(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "Proposal layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseROIPooling(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "ROIPooling layer is not supported by kmbPlugin";
}

void FrontEndMcm::parsePSROIPooling(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "PSROIPooling layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseCustom(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "Custom layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseMTCNN(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "MTCNN layer is not supported by kmbPlugin";
}

void FrontEndMcm::parsePad(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "Pad layer is not supported by kmbPlugin";
}

const static std::map<std::string, std::string> interpolationMap = {
    {"caffe.ResampleParameter.NEAREST", "NEAREST"},
    {"caffe.ResampleParameter.CUBIC", "BICUBIC"},
    {"caffe.ResampleParameter.LINEAR", "BILINEAR"},
};

void FrontEndMcm::parseResample(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    logParsingStartHelper(_logger, layer, inputs);

    auto antialias = layer->GetParamAsBool("antialias", 0);
    auto factor = layer->GetParamAsFloat("factor", 2.0);
    auto height = layer->GetParamAsUInt("height", 0);
    auto width = layer->GetParamAsUInt("width", 0);
    auto interpolation = layer->GetParamAsString("type", "caffe.ResampleParameter.NEAREST");

    auto layerOutput = layer->outData[0];
    IE_ASSERT(layerOutput != nullptr);
    mv::Shape output_shape(getWHCN(layerOutput->getTensorDesc()).getDims());

    auto resample_result = _modelMcm.resample(inputs[0]->getMcmNode(), interpolationMap.at(interpolation), antialias,
        output_shape, mv::DType("Default"), initialQuantParams, layer->name);

    bindOutput(resample_result, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, resample_result->getName());
}

void FrontEndMcm::parseLSTMCell(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "LSTMCell layer is not supported by kmbPlugin";
}

void FrontEndMcm::parsePriorBox(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    UNUSED(inputs);

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

    ParseLayersHelpers::priorBoxParam param(offset, step, min_sizes, max_sizes, flip, clip, scale_all_sizes,
        fixed_sizes, fixed_ratios, densitys, src_aspect_ratios, src_variance, data_dims, image_dims, out_dims);

    auto boxes = ParseLayersHelpers::computePriorbox(param);
    auto priorbox = _modelMcm.constant(boxes, {boxes.size() / 2, 2, 1, 1}, mv::DType("Float64"), mv::Order("NHWC"),
        initialQuantParams, layer->name + "_const");

    bindOutput(priorbox, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, priorbox->getName());
}

void FrontEndMcm::parsePriorBoxClustered(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "PriorBoxClustered layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseSplit(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "Split layer is not supported by kmbPlugin";
}

}  // namespace vpu
#endif
