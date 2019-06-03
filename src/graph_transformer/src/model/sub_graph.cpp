//
// Copyright (C) 2019 Intel Corporation.
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

#include <vpu/model/sub_graph.hpp>

#include <cctype>
#include <memory>
#include <string>
#include <set>
#include <exception>
#include <algorithm>

#include <details/caseless.hpp>

#include <vpu/compile_env.hpp>
#include <vpu/model/model.hpp>

namespace vpu {

//
// Constructor(s) / Destructor
//

SubGraph::~SubGraph() {
    if (_model.expired()) {
        return;
    }

    try {
        for (const auto& stage : _stagePtrs) {
            _model->disconnectStageDatas(stage);
        }
        _stagePtrs.clear();

        _model->cleanUpDatas();
    } catch (const std::exception& exc) {
        std::cerr << "Error in SubGraph destructor : " << exc.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown error in SubGraph destructor" << std::endl;
    }
}

//
// Data nodes
//

Data SubGraph::addConstData(
        const std::string& name,
        const DataDesc& desc,
        const DataContent::Ptr& content) {
    return _model->addConstData(name, desc, content);
}

Data SubGraph::addNewData(
        const std::string& name,
        const DataDesc& desc) {
    return _model->addNewData(name, desc);
}

Data SubGraph::addFakeData() {
    return _model->addFakeData();
}

Data SubGraph::duplicateData(
        const Data& origData,
        const std::string& postfix,
        const DataDesc& newDesc,
        const DataContent::Ptr& newContent) {
    return _model->duplicateData(origData, postfix, newDesc, newContent);
}

//
// Stage nodes
//

Stage SubGraph::addNewStageImpl(
    const std::string& name,
    StageType type,
    const ie::CNNLayerPtr& origLayer,
    const DataVector& inputs,
    const DataVector& outputs,
    const FuncRef<StagePtr()>& creator) {
    //
    // Check that Stage has inputs and outputs.
    //

    IE_ASSERT(!inputs.empty());
    IE_ASSERT(!outputs.empty());

    //
    // Check that Data objects belong to the same Model.
    //

    for (const auto& input : inputs) {
        IE_ASSERT(input->_model == _model);
    }
    for (const auto& output : outputs) {
        IE_ASSERT(output->_model == _model);
    }

    //
    // Check that there are no loops.
    //

    // TODO: more advanced check.
    for (const auto& output : outputs) {
        for (const auto& input : inputs) {
            IE_ASSERT(input != output);
        }
    }

    auto stage = creator();

    stage->_name = name;
    stage->_type = type;
    stage->_origLayer = origLayer;
    stage->_model = _model->handle_from_this();
    stage->_subGraph = handle_from_this();

    _cache.inputStages.emplace(stage);
    _cache.outputStages.emplace(stage);

    for (const auto& input : inputs) {
        addStageInput(stage, input);
    }
    for (const auto& output : outputs) {
        addStageOutput(stage, output);
    }

    stage->_ptrPosInSubGraph = _stagePtrs.emplace(_stagePtrs.end(), stage);

    _cache.resetStageOrder = true;

    return stage;
}

Stage SubGraph::duplicateStage(
        const std::string& name,
        const Stage& origStage,
        const DataVector& inputs,
        const DataVector& outputs) {
    //
    // Check that the new Stage has inputs and outputs.
    //

    IE_ASSERT(!inputs.empty());
    IE_ASSERT(!outputs.empty());

    //
    // Check that the objects belong to the same Model.
    //

    IE_ASSERT(origStage->_subGraph.get() == this);

    for (const auto& input : inputs) {
        IE_ASSERT(input->_model == _model);
    }

    for (const auto& output : outputs) {
        IE_ASSERT(output->_model == _model);
    }

    //
    // Check that there are no loops.
    //

    // TODO: more advanced check.
    for (const auto& output : outputs) {
        for (const auto& input : inputs) {
            IE_ASSERT(input != output);
        }
    }

    //
    // Create new Stage.
    //

    auto newStage = origStage->cloneImpl();

    newStage->_name = name;
    newStage->_type = origStage->_type;
    newStage->_origLayer = origStage->_origLayer;
    newStage->_model = _model->handle_from_this();
    newStage->_subGraph = handle_from_this();

    _cache.inputStages.emplace(newStage);
    _cache.outputStages.emplace(newStage);

    for (const auto& input : inputs) {
        addStageInput(newStage, input);
    }
    for (const auto& output : outputs) {
        addStageOutput(newStage, output);
    }
    for (const auto& tempBufferEdge : origStage->_tempBufferEdges) {
        addTempBuffer(newStage, tempBufferEdge->tempBuffer()->desc());
    }

    newStage->_ptrPosInSubGraph = _stagePtrs.emplace(_stagePtrs.end(), newStage);

    _cache.resetStageOrder = true;

    return newStage;
}

//
// Stage <-> Data edges
//

StageInput SubGraph::addStageInput(
        const Stage& stage,
        const Data& data) {
    return _model->addStageInput(stage, data);
}

StageOutput SubGraph::addStageOutput(
        const Stage& stage,
        const Data& data) {
    return _model->addStageOutput(stage, data);
}

StageTempBuffer SubGraph::addTempBuffer(
        const Stage& stage,
        const DataDesc& desc) {
    return _model->addTempBuffer(stage, desc);
}

void SubGraph::replaceStageInput(
        const StageInput& edge,
        const Data& newInput) {
    _model->replaceStageInput(edge, newInput);
}

void SubGraph::replaceStageOutput(
        const StageOutput& edge,
        const Data& newOutput) {
    _model->replaceStageOutput(edge, newOutput);
}

//
// Data <-> Data edges
//

DataEdgeBuilder SubGraph::connectDatas() {
    return _model->connectDatas();
}

//
// Stage <-> Stage edges
//

InjectedStageEdgeBuilder SubGraph::injectStage() {
    return _model->injectStage();
}

void SubGraph::revertInjection(const InjectedStage& edge) {
    _model->revertInjection(edge);
}

//
// Stage nodes removal
//

void SubGraph::disconnectStageDatas(const Stage& stage) {
    _model->disconnectStageDatas(stage);
}

void SubGraph::removeStage(const Stage& stage) {
    IE_ASSERT(stage->_subGraph.get() == this);

    disconnectStageDatas(stage);

    _cache.inputStages.erase(stage);
    _cache.outputStages.erase(stage);

    _cache.resetStageOrder = true;

    IE_ASSERT(stage->_ptrPosInSubGraph != _stagePtrs.end());
    _stagePtrs.erase(stage->_ptrPosInSubGraph);
}

//
// Stage order
//

void SubGraph::buildStageOrder(StageOrder order) const {
    VPU_PROFILE(buildStageOrder);

    if ((!_cache.resetStageOrder) && (order == _cache.stageOrder)) {
        IE_ASSERT(_cache.orderedStages.size() == _stagePtrs.size());
        return;
    }

    _cache.orderedStages.clear();
    _cache.resetStageOrder = false;
    _cache.stageOrder = order;

    if (_stagePtrs.empty()) {
        return;
    }

    //
    // Run recursive DFS algorithm
    //

    IE_ASSERT(!_cache.inputStages.empty());

    StageMap<bool> visitedMap;
    if (order == StageOrder::DFS) {
        for (auto revIt = _cache.inputStages.rbegin(); revIt != _cache.inputStages.rend(); ++revIt) {
            runDFS(*revIt, visitedMap);
        }
    } else if (order == StageOrder::BFS) {
        StageList queue(&StageNode::_posInSubGraphBfsTempList);
        for (const auto& stage : _cache.inputStages) {
            queue.push_back(stage);
            visitedMap[stage] = true;
        }
        runBFS(queue, visitedMap);
    } else {
        VPU_THROW_EXCEPTION << "Unsupported order " << order;
    }

    IE_ASSERT(_cache.orderedStages.size() == _stagePtrs.size());

    int stageInd = 0;
    for (const auto& stage : _cache.orderedStages) {
        stage->_index = stageInd;
        ++stageInd;
    }
}

void SubGraph::runDFS(
        const Stage& stage,
        StageMap<bool>& visitedMap) const {
    IE_ASSERT(stage->_parentStageEdge == nullptr);

    visitedMap[stage] = false;

    for (const auto& nextStage : stage->_cache.nextStages) {
        IE_ASSERT(nextStage.second > 0);

        // Traverse only stages from current sub-graph
        if (nextStage.first->_subGraph.get() != this) {
            continue;
        }

        auto it = visitedMap.find(nextStage.first);

        if (it != visitedMap.end()) {
            auto visited = it->second;

            if (!visited) {
                VPU_THROW_EXCEPTION << "Graph has cycle";
            }

            continue;
        }

        runDFS(nextStage.first, visitedMap);
    }

    visitedMap[stage] = true;

    _cache.orderedStages.push_front(stage);
}

void SubGraph::runBFS(
        StageList& queue,
        StageMap<bool>& visitedMap) const {
    while (!queue.empty()) {
        auto curStage = queue.front();
        queue.pop_front();

        _cache.orderedStages.push_back(curStage);

        for (const auto& nextStage : curStage->_cache.nextStages) {
            // Traverse only stages from current sub-graph
            if (nextStage.first->_subGraph.get() != this) {
                continue;
            }

            auto it = visitedMap.find(nextStage.first);

            if (it != visitedMap.end()) {
                auto visited = it->second;

                if (!visited) {
                    VPU_THROW_EXCEPTION << "Graph has cycle";
                }

                continue;
            }

            queue.push_back(nextStage.first);
            visitedMap[nextStage.first] = true;
        }
    }
}

}  // namespace vpu
