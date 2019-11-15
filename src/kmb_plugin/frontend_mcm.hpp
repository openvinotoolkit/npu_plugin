//
// Copyright 2019 Intel Corporation.
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

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <set>

#include <cpp/ie_cnn_network.h>
#include <details/caseless.hpp>

#include <vpu/utils/logger.hpp>
#include <vpu/utils/enums.hpp>
#include <vpu/utils/attributes_map.hpp>

#include <kmb_config.h>

#ifdef ENABLE_MCM_COMPILER
// suppress 'error: ‘inf’ defined but not used'
// static double inf = std::numeric_limits<double>::infinity();
// TODO remove when it is fixed in mcmCompiler
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#include <include/mcm/op_model.hpp>
#pragma GCC diagnostic pop
#include "kmb_base.hpp"

#include <graph_tools.hpp>

namespace vpu {

namespace KmbPlugin {

mv::DType convert_data_type(ie::Precision iePrecision);

class McmNodeObject final :
        public EnableHandle,
        public EnableCustomAttributes {
public:
    explicit McmNodeObject(mv::Data::TensorIterator node, InferenceEngine::TensorDesc desc) : _desc(desc), _mcmNode(node) {}
    KMB_MODEL_ATTRIBUTE(InferenceEngine::TensorDesc, desc, InferenceEngine::TensorDesc())
    KMB_MODEL_ATTRIBUTE(ie::DataPtr, origData, nullptr)

    mv::Data::TensorIterator& getMcmNode() { return _mcmNode; }
    void setOrigData(const ie::DataPtr& origData) { _origData = origData; }

private:
    mv::Data::TensorIterator _mcmNode;
};

KMB_DEFINE_MODEL_TYPES(McmNode, Object)

namespace ie = InferenceEngine;

class FrontEndMcm final : public std::enable_shared_from_this<FrontEndMcm> {
//
// Public API
//

public:
    using Ptr = std::shared_ptr<FrontEndMcm>;

    explicit FrontEndMcm(mv::OpModel& modelMcm, const KmbConfig& config)
             : _modelMcm(modelMcm)
             , _logger(std::make_shared<Logger>("FrontEndMcm", config.logLevel(), consoleOutput())) { }
    void buildInitialModel(ie::ICNNNetwork& network);

    std::set<std::string> checkSupportedLayers(ie::ICNNNetwork& network);

    McmNodePtr output() { return _output; }

//
// Passes
//

private:
    void runCommonPasses(ie::ICNNNetwork& network);

    void parseInputData();
    void parseOutputData();

//
// IR Parsers
//

public:
    void parseConvolution(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parsePooling(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseFullyConnected(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseReLU(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseSoftMax(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseGRN(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseMVN(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseNorm(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parsePower(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseScale(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parsePermute(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseDetectionOutput(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseEltwise(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseSigmoid(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseTanH(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parsePReLU(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseBatchNorm(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseDeconvolution(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseCopy(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseELU(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseCrop(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseTile(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseNormalize(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseRegionYolo(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseReorgYolo(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseBias(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseCTCDecoder(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseInterp(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseClamp(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseProposal(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseROIPooling(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parsePSROIPooling(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseCustom(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseMTCNN(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseLSTMCell(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parsePad(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseResample(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseArgMax(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parsePriorBox(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parsePriorBoxClustered(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseReshape(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseConcat(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseSplit(const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);

//
// Utility
//

private:
    McmNode getMcmData(const ie::DataPtr& ieData);
    void bindData(const McmNode& data, const ie::DataPtr& ieData);
    void bindOutput(mv::Data::TensorIterator node, ie::DataPtr& layerOutput);
    void getInputData(
            const ie::CNNLayerPtr& layer,
            McmNodeVector& inputs);

    struct ParsedNetwork {
        ie::InputsDataMap networkInputs;
        ie::OutputsDataMap networkOutputs;
        std::unordered_map<ie::DataPtr, ie::Blob::Ptr> constDatas;
        std::vector<ie::CNNLayerPtr> orderedLayers;
    };
    void parseNetworkDFS(const ie::ICNNNetwork& network, ParsedNetwork& parsedNetwork);

//
// Internal state
//

private:
    mv::OpModel& _modelMcm;
    McmNodePtrList _nodes;
    McmNodePtr _output;
    Logger::Ptr _logger;

    std::unordered_map<ie::DataPtr, McmNode> _ieToMcmMap;

    ParsedNetwork _parsedNetwork;
};
}  // namespace KmbPlugin
}  // namespace vpu
#endif
