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

#include <vpu/frontend/stage_builder.hpp>
#include <vpu/custom_layer.hpp>
#include <vpu/utils/enums.hpp>
#include <vpu/frontend/parse_network.hpp>

#ifdef ENABLE_MCM_COMPILER
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include <graph_tools.hpp>

namespace vpu {

namespace KmbPlugin {

class McmNodeObject final :
        public EnableHandleFromThis<McmNodeObject>,
        public EnableCustomAttributes {
public:
    explicit McmNodeObject(mv::Data::TensorIterator node, DataDesc desc) : _desc(desc), _mcmNode(node) {}
    VPU_MODEL_ATTRIBUTE(DataDesc, desc, DataDesc())
    VPU_MODEL_ATTRIBUTE(ie::DataPtr, origData, nullptr)
    VPU_MODEL_ATTRIBUTE(DataContent::Ptr, content, nullptr)

    mv::Data::TensorIterator& getMcmNode() { return _mcmNode; }
    void setOrigData(const ie::DataPtr& origData) { _origData = origData; }

private:
    mv::Data::TensorIterator _mcmNode;
};

VPU_DEFINE_MODEL_TYPES(McmNode, Object)

namespace ie = InferenceEngine;

class FrontEndMcm final : public std::enable_shared_from_this<FrontEndMcm> {
//
// Public API
//

public:
    using Ptr = std::shared_ptr<FrontEndMcm>;

    explicit FrontEndMcm(mv::OpModel& modelMcm) : _modelMcm(modelMcm) {}

    void buildInitialModel(const ie::ICNNNetwork& network);

    std::set<std::string> checkSupportedLayers(const ie::ICNNNetwork& network);

    const std::vector<ie::CNNLayerPtr>& allLayers() const { return _ieNetworkParser.orderedLayers; }
    McmNodePtr output() { return _output; }
    mv::OpModel& getOpModel() {
        return _modelMcm;
    }

//
// Passes
//

private:
    void runCommonPasses(
            const ie::ICNNNetwork& network,
            LayersOrder order);

    ie::CNNNetwork detectNetworkBatch(
            const ie::ICNNNetwork& network);

    void checkNetwork(const ie::CNNNetwork& network);

    void parseInputData();
    void parseOutputData();
    void addDataTypeConvertStages(const mv::OpModel& modelMcm);
    void addPreProcessStages(const mv::OpModel& modelMcm);

//
// IR Parsers
//

public:
    //
    // Layers, that might be both SW and HW
    //

    void parseConvolution(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parsePooling(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseFullyConnected(const mv::OpModel& modelMcml, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);

    //
    // SW only layers
    //

    void parseReLU(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseSoftMax(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseGRN(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseMVN(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseNorm(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parsePower(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseScale(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parsePermute(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseDetectionOutput(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseEltwise(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseSigmoid(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseTanH(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parsePReLU(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseBatchNorm(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseDeconvolution(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseCopy(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseELU(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseCrop(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseTile(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseNormalize(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseRegionYolo(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseReorgYolo(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseBias(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseCTCDecoder(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseInterp(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseClamp(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseProposal(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseROIPooling(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parsePSROIPooling(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseCustom(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseMTCNN(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseLSTMCell(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parsePad(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseResample(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseArgMax(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);

    //
    // Special layers
    //

    void parsePriorBox(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parsePriorBoxClustered(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseReshape(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseConcat(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);
    void parseSplit(const mv::OpModel& modelMcm, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs);

//
// Utility
//

private:
    McmNode getMcmData(const ie::DataPtr& ieData);
    void bindData(const McmNode& data, const ie::DataPtr& ieData);

    void getInputData(
            const ie::CNNLayerPtr& layer,
            McmNodeVector& inputs);

//
// Internal state
//

private:
    mv::OpModel& _modelMcm;
    McmNodePtrList _nodes;
    McmNodePtr _output;

    std::unordered_set<ie::DataPtr> _unbatchedOutputs;
    std::unordered_map<ie::DataPtr, McmNode> _ieToMcmMap;

    ie::details::caseless_map<std::string, CustomLayer::Ptr> _customLayers;
    vpu::IeNetworkParser _ieNetworkParser;
};

}  // namespace KmbPlugin

}  // namespace vpu
#endif
