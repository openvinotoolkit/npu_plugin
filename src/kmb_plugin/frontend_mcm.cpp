//
// Copyright 2018 Intel Corporation.
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

#include <frontend_mcm.hpp>

#include <atomic>
#include <memory>
#include <string>
#include <tuple>
#include <set>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <graph_tools.hpp>
#include <ie_profiling.hpp>

#include <utils/dims_parser.hpp>

#ifdef ENABLE_MCM_COMPILER
namespace vpu {

namespace KmbPlugin {

namespace {

typedef void (FrontEndMcm::*parser_t)(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs);

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
};

std::atomic<int> g_counter(0);

}  // namespace

mv::DType convert_data_type(ie::Precision iePrecision) {
    mv::DType mvType;
    switch (iePrecision) {
    case ie::Precision::I8:
        mvType = mv::DType("Int8");
        break;
    case ie::Precision::U8:
        mvType = mv::DType("UInt8");
        break;
    case ie::Precision::I32:
        mvType = mv::DType("Int32");
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

void FrontEndMcm::buildInitialModel(const ie::ICNNNetwork& network) {
    runCommonPasses(network);

    for (const auto& layer : _parsedNetwork.orderedLayers) {
        IE_ASSERT(layer != nullptr);

        _logger->debug("try to parse layer %s", layer->name);

        McmNodeVector inputs;
        getInputData(layer, inputs);
        auto it = g_mcm_parsers.find(layer->type);
        if (it == g_mcm_parsers.end()) {
            VPU_THROW_EXCEPTION
                    << "Cannot convert layer \""
                    << layer->name
                    << "\" due to unsupported layer type \""
                    << layer->type
                    << "\"";
        }

        auto parser = it->second;
        IE_ASSERT(parser != nullptr);

        (this->*parser)(_modelMcm, layer, inputs);
    }

    parseOutputData();
}

std::set<std::string> FrontEndMcm::checkSupportedLayers(const ie::ICNNNetwork& network) {
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

                (this->*parser)(_modelMcm, layer, inputs);

                layerNames.insert(layer->name);
            } catch (const ie::details::InferenceEngineException&) {
                // TODO: continue instead?
                break;
            }
        }
    }

    return layerNames;
}

ie::CNNNetwork FrontEndMcm::detectNetworkBatch(
        const ie::ICNNNetwork& origNetwork) {
    IE_PROFILING_AUTO_SCOPE(detectNetworkBatch);

    // Keep original network.
    IE_SUPPRESS_DEPRECATED_START
    return ie::CNNNetwork(const_cast<ie::ICNNNetwork*>(&origNetwork));
    IE_SUPPRESS_DEPRECATED_END
}

void FrontEndMcm::parseInputData() {
    IE_PROFILING_AUTO_SCOPE(parseInputData);

    _logger->debug("Try to parse network input");

    //
    // Parse network inputs
    //

    IE_ASSERT(_parsedNetwork.networkInputs.size() == 1);

    for (const auto& inputInfo : _parsedNetwork.networkInputs) {
        auto netInput = inputInfo.second;
        IE_ASSERT(netInput != nullptr);

        auto ieData = netInput->getInputData();
        IE_ASSERT(ieData != nullptr);

        const auto& dataDesc = ieData->getTensorDesc();
        mv::Shape inputShape(getWHCN(dataDesc).getDims());
        auto mvInput = _modelMcm.input(inputShape, convert_data_type(dataDesc.getPrecision()), mv::Order("NCHW"), {{}, {}, {}, {}}, netInput->name());
        auto input = std::make_shared<McmNodeObject>(mvInput, dataDesc);
        _nodes.push_back(input);
        bindData(input, ieData);
        _logger->debug("Network input '%s'(orig: '%s') parsed to mcmModel", mvInput->getName(), netInput->name());
    }

#if 0  // TODO: rewrite logic to mcm compiler if this part is needed
    //
    // Parse constant data
    //

        auto kmbData = model->addInputData(ieData->getName(), kmbDesc);
        bindData(kmbData, ieData);

        auto kmbData = model->addConstData(
            ieData->getName(),
            kmbDesc,
            ieBlobContent(ieBlob));

        // User might ask to return the output from Const layer.
        if (auto kmbOutData = getVpuData(ieData)) {
            IE_ASSERT(kmbOutData->usage() == DataUsage::Output);

            _stageBuilder->addCopyStage(
                model,
                formatString("%s@return-const", kmbData->name()),
                nullptr,
                kmbData,
                kmbOutData);
        }

        bindData(kmbData, ieData);
#endif
}

void FrontEndMcm::parseOutputData() {
    IE_PROFILING_AUTO_SCOPE(parseOutputData);

    _logger->debug("Try to parse network output");

    //
    // Parse network outputs
    //

    for (const auto& outputInfo : _parsedNetwork.networkOutputs) {
        auto ieData = outputInfo.second;

        IE_ASSERT(ieData != nullptr);

        auto lastLayerOut = getMcmData(ieData);
        IE_ASSERT(lastLayerOut != nullptr);
        auto name = lastLayerOut->getMcmNode()->getName();

        auto mvOutput = _modelMcm.output(lastLayerOut->getMcmNode());
        _output = std::make_shared<McmNodeObject>(mvOutput, lastLayerOut->desc());
        _nodes.push_back(_output);
    }
}

void FrontEndMcm::parseNetworkDFS(const ie::CNNNetwork& network, ParsedNetwork& parsedNetwork) {
    IE_PROFILING_AUTO_SCOPE(parseNetworkDFS);

    ie::details::CaselessEq<std::string> cmp;

    //
    // Collect all network input data.
    //

    auto reshapedNetwork = detectNetworkBatch(network);

    parsedNetwork.networkInputs = reshapedNetwork.getInputsInfo();
    parsedNetwork.networkOutputs = reshapedNetwork.getOutputsInfo();

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

    for (const auto& layer : ie::CNNNetGetAllInputLayers(network)) {
        IE_ASSERT(layer != nullptr);

        if (!cmp(layer->type, "Const"))
            continue;

        if (layer->outData.size() != 1) {
            VPU_THROW_EXCEPTION
                    << "Const layer " << layer->name
                    << " has unsupported number of outputs "
                    << layer->outData.size();
        }

        if (layer->blobs.size() != 1) {
            VPU_THROW_EXCEPTION
                    << "Const layer " << layer->name
                    << " has unsupported number of blobs "
                    << layer->blobs.size();
        }

        auto constData = layer->outData[0];
        IE_ASSERT(constData != nullptr);

        auto constBlob = layer->blobs.begin()->second;
        IE_ASSERT(constBlob != nullptr);

        parsedNetwork.constDatas[constData] = constBlob;

        allInputDatas.insert(constData);
    }

    //
    // Collect initial layers.
    //

    std::unordered_set<ie::CNNLayerPtr> visitedInitialLayers;
    SmallVector<ie::CNNLayerPtr> initialLayers;

    for (const auto& inputData : allInputDatas) {
        for (const auto& consumer : inputData->getInputTo()) {
            auto initialLayer = consumer.second;
            IE_ASSERT(initialLayer != nullptr);

            if (visitedInitialLayers.count(initialLayer) > 0)
                continue;

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

    std::sort(initialLayers.begin(), initialLayers.end(),
              [](const ie::CNNLayerPtr& left, const ie::CNNLayerPtr& right) {
                  ie::details::CaselessLess<std::string> cmp;
                  return cmp(left->name, right->name);
              });

    InferenceEngine::CNNNetForestDFS(initialLayers, [&parsedNetwork](const ie::CNNLayerPtr& layer) {
        parsedNetwork.orderedLayers.emplace_back(layer);
    }, false);

    std::reverse(parsedNetwork.orderedLayers.begin(), parsedNetwork.orderedLayers.end());
}

void FrontEndMcm::runCommonPasses(const ie::ICNNNetwork& network) {
    auto reshapedNetwork = detectNetworkBatch(network);
    parseNetworkDFS(reshapedNetwork, _parsedNetwork);
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

void FrontEndMcm::getInputData(
        const ie::CNNLayerPtr& layer,
        McmNodeVector& inputs) {
    IE_ASSERT(layer != nullptr);

    inputs.resize(layer->insData.size());
    for (size_t i = 0; i < layer->insData.size(); ++i) {
        auto layerInput = layer->insData[i].lock();
        IE_ASSERT(layerInput != nullptr);

        inputs[i] = getMcmData(layerInput);
        IE_ASSERT(inputs[i] != nullptr);
    }
}

}  // namespace KmbPlugin

}  // namespace vpu
#endif
