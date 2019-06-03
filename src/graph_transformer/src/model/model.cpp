//
// Copyright (C) 2018-2019 Intel Corporation.
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

#include <vpu/model/model.hpp>

#include <cctype>
#include <memory>
#include <string>
#include <set>
#include <exception>
#include <algorithm>
#include <vector>
#include <utility>

#include <details/caseless.hpp>
#include <vpu/utils/auto_scope.hpp>

namespace vpu {

//
// Resources
//

void printTo(std::ostream& os, const Resources& res) {
    os << "[" << std::endl;

    os << "numCMXSlices=" << res.numCMXSlices << std::endl;
    os << "numSHAVEs=" << res.numSHAVEs << std::endl;
    os << "cmxLimit=" << res.cmxLimit << std::endl;

    os << "]";
}

void printTo(DotLabel& lbl, const Resources& res) {
    DotLabel subLbl(lbl);
    subLbl.appendPair("numCMXSlices", res.numCMXSlices);
    subLbl.appendPair("numSHAVEs", res.numSHAVEs);
    subLbl.appendPair("cmxLimit", res.cmxLimit);
}

//
// Model : Data nodes
//

Data Model::createData(
        const std::string& name,
        DataUsage usage,
        const DataDesc& desc) {
    std::shared_ptr<DataNode> data(new DataNode);

    data->_name = name;
    data->_usage = usage;
    data->_desc = desc;
    data->_model = handle_from_this();

    data->_ptrPosInModel = _dataPtrs.emplace(_dataPtrs.end(), data);
    _cache.dataHandles.push_back(data);

    return data;
}

Data Model::addConstData(
        const std::string& name,
        const DataDesc& desc,
        const DataContent::Ptr& content) {
    _allocator.setNeedToAllocNonIntermData();

    IE_ASSERT(content != nullptr);
    content->_desc = desc;

    auto data = createData(name, DataUsage::Const, desc);
    data->_content = content;

    return data;
}

Data Model::duplicateData(
        const Data& origData,
        const std::string& postfix,
        const DataDesc& newDesc,
        const DataContent::Ptr& newContent) {
    //
    // Check that the objects belong to the same Model.
    //

    IE_ASSERT(origData->_model.get() == this);

    //
    // Duplicate Data node.
    //

    auto newDataUsage = origData->usage();

    // Duplicates for Input & Output can be only Intermediate
    if (newDataUsage == DataUsage::Input ||
        newDataUsage == DataUsage::Output) {
        newDataUsage = DataUsage::Intermediate;
    }

    auto newData = createData(origData->name() + postfix, newDataUsage, newDesc.numDims() != 0 ? newDesc : origData->desc());

    newData->attrs().copyFrom(origData->attrs());

    if (newDataUsage == DataUsage::Const) {
        newData->_content = newContent != nullptr ? newContent : origData->content();
        if (newContent != nullptr) {
            newContent->_desc = newData->_desc;
        }
    }

    return newData;
}

//
// Model : Stage <-> Data edges
//

namespace {

template <class StageOrderedMap>
inline bool anyHasSameSubGraph(const StageOrderedMap& stages, const SubGraphHandle& subGraph) {
    return std::any_of(stages.begin(), stages.end(), [&subGraph](const typename StageOrderedMap::value_type& p) {
        return p.first->subGraph() == subGraph;
    });
}

}  // namespace

void Model::resetStageOrderCache(const Stage& stage) {
    if (!stage->_subGraph.expired()) {
        stage->_subGraph->_cache.resetStageOrder = true;
    }
}

void Model::updateStageOrderCache(const Stage& producer, const Stage& consumer) {
    auto val1 = ++producer->_cache.nextStages[consumer];
    if (val1 == 1) {
        updateOutputStagesCache(producer);
        resetStageOrderCache(producer);
    }

    auto val2 = ++consumer->_cache.prevStages[producer];
    if (val2 == 1) {
        updateInputStagesCache(consumer);
        resetStageOrderCache(consumer);
    }
}

void Model::eraseFromStageOrderCache(const Stage& producer, const Stage& consumer) {
    auto it1 = producer->_cache.nextStages.find(consumer);
    IE_ASSERT(it1 != producer->_cache.nextStages.end());

    --it1->second;
    if (it1->second <= 0) {
        producer->_cache.nextStages.erase(it1);

        updateOutputStagesCache(producer);
        resetStageOrderCache(producer);
    }

    auto it2 = consumer->_cache.prevStages.find(producer);
    IE_ASSERT(it2 != consumer->_cache.prevStages.end());

    --it2->second;
    if (it2->second <= 0) {
        consumer->_cache.prevStages.erase(it2);

        updateInputStagesCache(consumer);
        resetStageOrderCache(consumer);
    }
}

void Model::updateInputStagesCache(const Stage& stage) {
    if (!stage->_subGraph.expired()) {
        if (anyHasSameSubGraph(stage->_cache.prevStages, stage->_subGraph)) {
            stage->_subGraph->_cache.inputStages.erase(stage);
        } else {
            stage->_subGraph->_cache.inputStages.emplace(stage);
        }
    }
}

void Model::updateOutputStagesCache(const Stage& stage) {
    if (!stage->_subGraph.expired()) {
        if (anyHasSameSubGraph(stage->_cache.nextStages, stage->_subGraph)) {
            stage->_subGraph->_cache.outputStages.erase(stage);
        } else {
            stage->_subGraph->_cache.outputStages.emplace(stage);
        }
    }
}

void Model::eraseFromInputOutputStagesCache(const Stage& stage) {
    if (!stage->_subGraph.expired()) {
        stage->_subGraph->_cache.inputStages.erase(stage);
        stage->_subGraph->_cache.outputStages.erase(stage);
    }
}

StageInput Model::addStageInput(
        const Stage& stage,
        const Data& data) {
    //
    // Check that the objects belong to the same Model.
    //

    IE_ASSERT(stage->_model.get() == this);
    IE_ASSERT(data->_model.get() == this);

    // TODO: check for loops in the graph.

    //
    // Input data can't be Temp.
    //

    IE_ASSERT(data->_usage != DataUsage::Temp);

    //
    // Create new Edge.
    //

    std::shared_ptr<StageInputEdge> edge(new StageInputEdge);

    edge->_consumer = stage;
    edge->_input = data;
    edge->_portInd = checked_cast<int>(stage->_inputEdges.size());
    edge->_model = handle_from_this();

    edge->_ptrPosInModel = _inEdgePtrs.emplace(_inEdgePtrs.end(), edge);
    data->_consumerEdges.push_back(edge);
    stage->_inputEdges.emplace_back(edge);

    //
    // Update Cache
    //

    if (data->_producerEdge != nullptr) {
        IE_ASSERT(stage->_parentStageEdge == nullptr);
        IE_ASSERT(data->_producerEdge->_producer->_parentStageEdge == nullptr);

        updateStageOrderCache(data->_producerEdge->_producer, stage);
    }

    return edge;
}

StageOutput Model::addStageOutput(
        const Stage& stage,
        const Data& data) {
    //
    // Check that the objects belong to the same Model.
    //

    IE_ASSERT(stage->_model.get() == this);
    IE_ASSERT(data->_model.get() == this);

    //
    // Check that the `data` is free.
    //

    IE_ASSERT(data->_producerEdge == nullptr);

    if (data->_parentDataEdge != nullptr) {
        IE_ASSERT(data->_parentDataEdge->_order != SharedDataOrder::ParentWritesToChild);
    }

    for (const auto& childDataEdge : data->_childDataEdges) {
        IE_ASSERT(childDataEdge->_order != SharedDataOrder::ChildWritesToParent);
    }

    //
    // Output data can be Output, Intermediate, or Fake only.
    //

    IE_ASSERT(data->_usage == DataUsage::Output || data->_usage == DataUsage::Intermediate || data->_usage == DataUsage::Fake);

    // TODO: check for loops in the graph.

    std::shared_ptr<StageOutputEdge> edge(new StageOutputEdge);

    edge->_producer = stage;
    edge->_output = data;
    edge->_portInd = checked_cast<int>(stage->_outputEdges.size());
    edge->_model = handle_from_this();

    edge->_ptrPosInModel = _outEdgePtrs.emplace(_outEdgePtrs.end(), edge);
    stage->_outputEdges.emplace_back(edge);
    data->_producerEdge = edge;

    //
    // Update Cache
    //

    for (const auto& consumerEdge : data->_consumerEdges) {
        IE_ASSERT(stage->_parentStageEdge == nullptr);
        IE_ASSERT(consumerEdge->_consumer->_parentStageEdge == nullptr);

        updateStageOrderCache(stage, consumerEdge->_consumer);
    }

    return edge;
}

StageTempBuffer Model::addTempBuffer(
        const Stage& stage,
        const DataDesc& desc) {
    //
    // Check that objects belong to the same Model.
    //

    IE_ASSERT(stage->_model.get() == this);

    //
    // Create new Data.
    //

    auto data = createData(
        formatString("%s@temp@%d", stage->name(), stage->_tempBufferEdges.size() + 1),
        DataUsage::Temp,
        desc);

    //
    // Create new Edge.
    //

    std::shared_ptr<StageTempBufferEdge> edge(new StageTempBufferEdge);

    edge->_stage = stage;
    edge->_tempBuffer = data;
    edge->_portInd = checked_cast<int>(stage->_tempBufferEdges.size());
    edge->_model = handle_from_this();

    edge->_ptrPosInModel = _tempBufferEdgePtrs.emplace(_tempBufferEdgePtrs.end(), edge);
    stage->_tempBufferEdges.emplace_back(edge);
    data->_tempBufferEdge = edge;

    return edge;
}

void Model::replaceStageInput(
        const StageInput& edge,
        const Data& newInput) {
    //
    // Check that objects belong to the same Model.
    //

    IE_ASSERT(edge->_model.get() == this);
    IE_ASSERT(newInput->_model.get() == this);

    //
    // Check that there are no loops.
    //

    // TODO: more advanced check.
    for (const auto& output : edge->consumer()->outputs()) {
        IE_ASSERT(newInput != output);
    }

    //
    // Input data can't be Temp.
    //

    IE_ASSERT(newInput->_usage != DataUsage::Temp);

    //
    // Can't replace Edge from injected Stage.
    //

    IE_ASSERT(edge->_parentEdge == nullptr);
    IE_ASSERT(edge->_childEdge == nullptr);

    //
    // Remove Edge from previous input.
    //

    edge->_input->_consumerEdges.erase(edge);

    //
    // Clear previous Cache
    //

    if (edge->_input->_producerEdge != nullptr) {
        eraseFromStageOrderCache(edge->_input->_producerEdge->_producer, edge->_consumer);
    }

    //
    // Set new input.
    //

    edge->_input = newInput;
    newInput->_consumerEdges.push_back(edge);

    //
    // Update current Cache
    //

    if (newInput->_producerEdge != nullptr) {
        IE_ASSERT(edge->_consumer->_parentStageEdge == nullptr);
        IE_ASSERT(newInput->_producerEdge->_producer->_parentStageEdge == nullptr);

        updateStageOrderCache(newInput->_producerEdge->_producer, edge->_consumer);
    }
}

void Model::replaceStageOutput(
        const StageOutput& edge,
        const Data& newOutput) {
    //
    // Check that objects belong to the same Model.
    //

    IE_ASSERT(edge->_model.get() == this);
    IE_ASSERT(newOutput->_model.get() == this);

    //
    // Check that there are no loops.
    //

    // TODO: more advanced check.
    for (const auto& input : edge->producer()->inputs()) {
        IE_ASSERT(newOutput != input);
    }

    //
    // Check that `data` is free.
    //

    IE_ASSERT(newOutput->_producerEdge == nullptr);

    if (newOutput->_parentDataEdge != nullptr) {
        IE_ASSERT(newOutput->_parentDataEdge->_order != SharedDataOrder::ParentWritesToChild);
    }

    for (const auto& childDataEdge : newOutput->_childDataEdges) {
        IE_ASSERT(childDataEdge->_order != SharedDataOrder::ChildWritesToParent);
    }

    //
    // Output data can be Output/Intermediate/Fake.
    //

    IE_ASSERT(newOutput->_usage == DataUsage::Output ||
              newOutput->_usage == DataUsage::Intermediate ||
              newOutput->_usage == DataUsage::Fake);

    //
    // Can't replace Edge from injected Stage.
    //

    IE_ASSERT(edge->_parentEdge == nullptr);
    IE_ASSERT(edge->_childEdge == nullptr);

    //
    // Remove Edge from previous output.
    //

    edge->_output->_producerEdge = nullptr;

    //
    // Clear previous Cache
    //

    for (const auto& consumerEdge : edge->_output->_consumerEdges) {
        eraseFromStageOrderCache(edge->_producer, consumerEdge->_consumer);
    }

    //
    // Set new output.
    //

    edge->_output = newOutput;
    newOutput->_producerEdge = edge;

    //
    // Update current Cache
    //

    for (const auto& consumerEdge : newOutput->_consumerEdges) {
        IE_ASSERT(edge->_producer->_parentStageEdge == nullptr);
        IE_ASSERT(consumerEdge->_consumer->_parentStageEdge == nullptr);

        updateStageOrderCache(edge->_producer, consumerEdge->_consumer);
    }
}

//
// Model : Data <-> Data edges
//

DataEdgeBuilder::~DataEdgeBuilder() {
    //
    // Check that `done` was called.
    //

    if (_model != nullptr) {
        std::terminate();
    }
}

DataEdgeBuilder& DataEdgeBuilder::parent(const Data& parent) {
    //
    // Check that `done` was not called.
    //

    IE_ASSERT(_model != nullptr);

    //
    // Check that `parent` was not called.
    //

    IE_ASSERT(_parent == nullptr);

    //
    // Check that objects belong to the same Model.
    //

    IE_ASSERT(parent->_model == _model);

    _parent = parent;

    return *this;
}

DataEdgeBuilder& DataEdgeBuilder::child(const Data& child) {
    //
    // Check that `done` was not called.
    //

    IE_ASSERT(_model != nullptr);

    //
    // Check that `child` was not called.
    //

    IE_ASSERT(_child == nullptr);

    //
    // Check that objects belong to the same Model.
    //

    IE_ASSERT(child->_model == _model);

    _child = child;

    return *this;
}

DataEdgeBuilder& DataEdgeBuilder::mode(SharedDataMode mode) {
    //
    // Check that `done` was not called.
    //

    IE_ASSERT(_model != nullptr);

    //
    // Check that `mode` was not called.
    //

    IE_ASSERT(!_modeSet);

    _mode = mode;
    _modeSet = true;

    return *this;
}

DataEdgeBuilder& DataEdgeBuilder::order(SharedDataOrder order) {
    //
    // Check that `done` was not called.
    //

    IE_ASSERT(_model != nullptr);

    //
    // Check that `order` was not called.
    //

    IE_ASSERT(!_orderSet);

    _order = order;
    _orderSet = true;

    return *this;
}

DataEdgeBuilder& DataEdgeBuilder::offset(const DimValues& offset) {
    //
    // Check that `done` was not called.
    //

    IE_ASSERT(_model != nullptr);

    //
    // Check that `offset` was not called.
    //

    IE_ASSERT(!_offsetSet);

    _offset = offset;
    _offsetSet = true;

    return *this;
}

SharedAllocation DataEdgeBuilder::done() {
    //
    // Check that `done` was not called.
    //

    IE_ASSERT(_model != nullptr);

    //
    // Check that all fields were set.
    //

    IE_ASSERT(_parent != nullptr);
    IE_ASSERT(_child != nullptr);
    IE_ASSERT(_modeSet);
    IE_ASSERT(_orderSet);

    AutoScope autoNullModel([&] {
        _model = nullptr;
    });

    //
    // Call the actual implementation.
    //

    auto edge = _model->connectDatasImpl(
        _parent, _child,
        _mode, _order,
        _offset);

    //
    // Reset internal state.
    //

    _model = nullptr;

    return edge;
}

SharedAllocation Model::connectDatasImpl(
        const Data& parent,
        const Data& child,
        SharedDataMode mode,
        SharedDataOrder order,
        const DimValues& offset) {
    //
    // Get producer and consumer data.
    //

    Data producer, consumer;
    if (order == SharedDataOrder::ChildWritesToParent) {
        producer = child;
        consumer = parent;
    } else if (order == SharedDataOrder::ParentWritesToChild) {
        producer = parent;
        consumer = child;
    } else {
        VPU_THROW_EXCEPTION << "Invalid data order " << order;
    }

    //
    // Child must be Intermediate.
    //

    VPU_THROW_UNLESS(child->_usage == DataUsage::Intermediate);

    //
    // Parent can't be Temp or Fake.
    //

    VPU_THROW_UNLESS(parent->_usage != DataUsage::Temp && parent->_usage != DataUsage::Fake);

    //
    // Consumer must be accesible from the producer.
    //

    Stage connectionStage;

    for (const auto& consumerEdge : producer->_consumerEdges) {
        for (const auto& outEdge : consumerEdge->_consumer->_outputEdges) {
            if (outEdge->_output == consumer) {
                connectionStage = consumerEdge->_consumer;
                break;
            }
        }

        if (connectionStage != nullptr) {
            break;
        }
    }

    IE_ASSERT(connectionStage != nullptr);

    //
    // Connection stage must be special.
    //

    VPU_THROW_UNLESS(connectionStage->category() == StageCategory::Special);

    //
    // Special checks for each mode.
    //

    if (mode == SharedDataMode::ROI) {
        //
        // Check connection stage type and that parent has the largest buffer.
        //

        if (connectionStage->_type == StageType::Concat ||
            connectionStage->_type == StageType::Expand) {
            IE_ASSERT(producer == child);
            IE_ASSERT(consumer == parent);
        } else if (connectionStage->_type == StageType::Split ||
                   connectionStage->_type == StageType::Shrink) {
            IE_ASSERT(producer == parent);
            IE_ASSERT(consumer == child);
        } else {
            VPU_THROW_EXCEPTION
                    << "Stage type " << connectionStage->_type
                    << " can't be used for ROI data connection";
        }

        //
        // Parent and child must have the same order.
        //

        VPU_THROW_UNLESS(parent->desc().dimsOrder() == child->desc().dimsOrder());

        //
        // Offset must be valid.
        //

        for (const auto& p : offset) {
            IE_ASSERT(parent->desc().dimsOrder().hasDim(p.first));

            IE_ASSERT(child->desc().dim(p.first) + p.second <= parent->desc().dim(p.first));
        }

        //
        // Check strides requirements
        //

        IE_ASSERT(checkStrides(child->desc(), parent->strides(), child->_requiredStrides));
        child->resetRequiredStrides();
    } else if (mode == SharedDataMode::Reshape) {
        //
        // Check connection stage type.
        //

        IE_ASSERT(connectionStage->_type == StageType::Reshape);

        //
        // Parent and child must have the same data type.
        //

        IE_ASSERT(parent->desc().type() == child->desc().type());

        //
        // Parent and child must have the same number of elements.
        //

        IE_ASSERT(parent->desc().totalDimSize() == child->desc().totalDimSize());

        //
        // Parent and child must be compact.
        //

        // TODO: can we weaken this restriction?
        IE_ASSERT(parent->checkStrides(StridesRequirement::compact()));
        IE_ASSERT(child->checkStrides(StridesRequirement::compact()));
    } else {
        VPU_THROW_EXCEPTION << "Invalid shared data mode " << mode;
    }

    //
    // Remove previous edge if any.
    //

    auto prevEdge = child->_parentDataEdge;

    if (prevEdge != nullptr) {
        prevEdge->_parent->_childDataEdges.erase(prevEdge);
    }

    //
    // Create new Edge.
    //

    std::shared_ptr<SharedAllocationEdge> edge(new SharedAllocationEdge);

    edge->_parent = parent;
    edge->_child = child;
    edge->_connection = connectionStage;
    edge->_mode = mode;
    edge->_order = order;
    edge->_model = handle_from_this();

    if (mode == SharedDataMode::ROI) {
        edge->attrs().set<DimValues>("offset", offset);
    }

    edge->_ptrPosInModel = _dataEdgePtrs.emplace(_dataEdgePtrs.end(), edge);
    parent->_childDataEdges.push_back(edge);

    child->_parentDataEdge = edge;

    //
    // Deallocate previous edge if any.
    //

    if (prevEdge != nullptr) {
        IE_ASSERT(prevEdge->_ptrPosInModel != _dataEdgePtrs.end());
        _dataEdgePtrs.erase(prevEdge->_ptrPosInModel);
    }

    //
    // Notify allocator
    //

    _allocator.setNeedToAllocNonIntermData();

    return edge;
}

//
// Model : Stage <-> Stage edges
//

InjectedStageEdgeBuilder::~InjectedStageEdgeBuilder() {
    //
    // Check that `done` was called.
    //

    if (_model != nullptr) {
        std::terminate();
    }
}

InjectedStageEdgeBuilder& InjectedStageEdgeBuilder::parentHW(const Stage& parent) {
    //
    // Check that `done` was not called.
    //

    IE_ASSERT(_model != nullptr);

    //
    // Check that `parentHW` was not called.
    //

    IE_ASSERT(_parent == nullptr);

    //
    // Check that objects belong to the same Model.
    //

    IE_ASSERT(parent->_model == _model);

    //
    // Check that `parent` is HW.
    //

    IE_ASSERT(parent->category() == StageCategory::HW);

    _parent = parent;

    return *this;
}

InjectedStageEdgeBuilder& InjectedStageEdgeBuilder::childSW(const Stage& child) {
    //
    // Check that `done` was not called.
    //

    IE_ASSERT(_model != nullptr);

    //
    // Check that `childSW` was not called.
    //

    IE_ASSERT(_child == nullptr);

    //
    // Check that objects belong to the same Model.
    //

    IE_ASSERT(child->_model == _model);

    //
    // Check that `parent` is HW.
    //

    IE_ASSERT(child->category() == StageCategory::DMA || child->category() == StageCategory::SHAVE);

    _child = child;

    return *this;
}

InjectedStage InjectedStageEdgeBuilder::done() {
    //
    // Check that `done` was not called.
    //

    IE_ASSERT(_model != nullptr);

    //
    // Check that all fields were set.
    //

    IE_ASSERT(_parent != nullptr);
    IE_ASSERT(_child != nullptr);

    //
    // Call actual implementation.
    //

    auto edge = _model->injectStageImpl(_parent, _child);

    //
    // Reset the internal state.
    //

    _model = nullptr;

    return edge;
}

InjectedStage Model::injectStageImpl(
        const Stage& parent,
        const Stage& child) {
    //
    // Check the parent and child was not already injected.
    //

    IE_ASSERT(parent->_parentStageEdge == nullptr);

    IE_ASSERT(child->_parentStageEdge == nullptr);
    IE_ASSERT(child->_injectedStageEdges.empty());

    //
    // Create new Edge.
    //

    std::shared_ptr<InjectedStageEdge> edge(new InjectedStageEdge);

    edge->_parent = parent;
    edge->_child = child.lock();
    edge->_portInd = checked_cast<int>(parent->_injectedStageEdges.size());
    edge->_model = handle_from_this();

    edge->_ptrPosInModel = _stageEdgePtrs.emplace(_stageEdgePtrs.end(), edge);
    parent->_injectedStageEdges.push_back(edge);

    child->_parentStageEdge = edge;

    //
    // Redirect child inputs to parent.
    //

    for (const auto& childInEdge : child->_inputEdges) {
        if (childInEdge->_input->_producerEdge != nullptr) {
            eraseFromStageOrderCache(childInEdge->_input->_producerEdge->_producer, childInEdge->_consumer);
        }

        childInEdge->_input->_consumerEdges.erase(childInEdge);

        auto parentInEdge = addStageInput(parent, childInEdge->_input);

        childInEdge->_parentEdge = parentInEdge;
        parentInEdge->_childEdge = childInEdge;
    }

    //
    // Redirect child outputs to parent.
    //

    for (const auto& childOutEdge : child->_outputEdges) {
        for (const auto& consumerEdge : childOutEdge->_output->_consumerEdges) {
            eraseFromStageOrderCache(childOutEdge->_producer, consumerEdge->_consumer);
        }

        childOutEdge->_output->_producerEdge = nullptr;

        auto parentOutEdge = addStageOutput(parent, childOutEdge->_output);

        childOutEdge->_parentEdge = parentOutEdge;
        parentOutEdge->_childEdge = childOutEdge;
    }

    //
    // Redirect child temp buffers to parent.
    //

    for (const auto& childEdge : child->_tempBufferEdges) {
        childEdge->_tempBuffer->_tempBufferEdge = nullptr;

        std::shared_ptr<StageTempBufferEdge> parentEdge(new StageTempBufferEdge);

        parentEdge->_stage = parent;
        parentEdge->_tempBuffer = childEdge->_tempBuffer;
        parentEdge->_portInd = checked_cast<int>(parent->_tempBufferEdges.size());
        parentEdge->_model = handle_from_this();

        parentEdge->_ptrPosInModel = _tempBufferEdgePtrs.emplace(_tempBufferEdgePtrs.end(), parentEdge);

        parent->_tempBufferEdges.emplace_back(parentEdge);
        childEdge->_tempBuffer->_tempBufferEdge = parentEdge;

        childEdge->_parentEdge = parentEdge;
        parentEdge->_childEdge = childEdge;
    }

    //
    // Move child Stage from the Model to parent Stage.
    //

    IE_ASSERT(child->_ptrPosInSubGraph != child->_subGraph->_stagePtrs.end());
    child->_subGraph->_stagePtrs.erase(child->_ptrPosInSubGraph);
    child->_ptrPosInSubGraph = child->_subGraph->_stagePtrs.end();

    //
    // Update Cache
    //

    updateInputStagesCache(parent);
    updateOutputStagesCache(parent);

    eraseFromInputOutputStagesCache(child);

    resetStageOrderCache(parent);
    resetStageOrderCache(child);

    return edge;
}

void Model::revertInjection(const InjectedStage& edge) {
    //
    // Check that objects belong to the same Model.
    //

    IE_ASSERT(edge->_model.get() == this);

    auto parentStage = edge->_parent;
    auto childStage = edge->_child;

    IE_ASSERT(parentStage->_model.get() == this);
    IE_ASSERT(childStage->_model.get() == this);
    IE_ASSERT(childStage->_parentStageEdge == edge);

    //
    // Move child Stage from parent Stage to the Model.
    //

    childStage->_ptrPosInSubGraph = childStage->_subGraph->_stagePtrs.emplace(childStage->_subGraph->_stagePtrs.end(), childStage);

    //
    // Remove InjectedStage Edge from parent and child Stage.
    //

    parentStage->_injectedStageEdges.erase(edge);
    childStage->_parentStageEdge = nullptr;

    //
    // Remove Injected Input Edges from parent Stage.
    //

    int startInd = -1;
    int endInd = -1;

    for (const auto& inEdge : parentStage->_inputEdges) {
        if (inEdge->_childEdge == nullptr) {
            IE_ASSERT(startInd < 0);
            continue;
        }

        if (startInd >= 0 && endInd >= 0) {
            IE_ASSERT(inEdge->_childEdge->_consumer != childStage);
        }

        if (inEdge->_childEdge->_consumer != childStage) {
            if (startInd >= 0 && endInd < 0) {
                endInd = inEdge->_portInd;
            }
            continue;
        }

        if (startInd < 0) {
            startInd = inEdge->_portInd;
        }
        if (inEdge->_portInd == checked_cast<int>(parentStage->_inputEdges.size()) - 1) {
            endInd = inEdge->_portInd + 1;
        }

        if (inEdge->_input->_producerEdge != nullptr) {
            eraseFromStageOrderCache(inEdge->_input->_producerEdge->_producer, inEdge->_consumer);
        }

        if (inEdge->_childEdge->_input->_producerEdge != nullptr) {
            IE_ASSERT(inEdge->_childEdge->_consumer->_parentStageEdge == nullptr);
            IE_ASSERT(inEdge->_childEdge->_input->_producerEdge->_producer->_parentStageEdge == nullptr);

            updateStageOrderCache(inEdge->_childEdge->_input->_producerEdge->_producer, inEdge->_childEdge->_consumer);
        }

        inEdge->_childEdge->_parentEdge = nullptr;
        inEdge->_input->_consumerEdges.erase(inEdge);
        inEdge->_input->_consumerEdges.push_back(inEdge->_childEdge);

        IE_ASSERT(inEdge->_ptrPosInModel != _inEdgePtrs.end());
        _inEdgePtrs.erase(inEdge->_ptrPosInModel);
    }

    IE_ASSERT(startInd >= 0 && endInd > startInd && static_cast<size_t>(startInd) <= parentStage->_inputEdges.size());
    parentStage->_inputEdges.erase(
        parentStage->_inputEdges.begin() + startInd,
        parentStage->_inputEdges.begin() + endInd);

    for (size_t i = 0; i < parentStage->_inputEdges.size(); ++i) {
        parentStage->_inputEdges[i]->_portInd = checked_cast<int>(i);
    }

    //
    // Remove Injected Output Edges from parent Stage.
    //

    startInd = -1;
    endInd = -1;

    for (const auto& outEdge : parentStage->_outputEdges) {
        if (outEdge->_childEdge == nullptr) {
            IE_ASSERT(startInd < 0);
            continue;
        }

        if (startInd >= 0 && endInd >= 0) {
            IE_ASSERT(outEdge->_childEdge->_producer != childStage);
        }

        if (outEdge->_childEdge->_producer != childStage) {
            if (startInd >= 0 && endInd < 0) {
                endInd = outEdge->_portInd;
            }
            continue;
        }

        if (startInd < 0) {
            startInd = outEdge->_portInd;
        }
        if (outEdge->_portInd == checked_cast<int>(parentStage->_outputEdges.size()) - 1) {
            endInd = outEdge->_portInd + 1;
        }

        for (const auto& consumerEdge : outEdge->_output->_consumerEdges) {
            eraseFromStageOrderCache(outEdge->_producer, consumerEdge->_consumer);
        }

        for (const auto& consumerEdge : outEdge->_childEdge->_output->_consumerEdges) {
            IE_ASSERT(outEdge->_childEdge->_producer->_parentStageEdge == nullptr);
            IE_ASSERT(consumerEdge->_consumer->_parentStageEdge == nullptr);

            updateStageOrderCache(outEdge->_childEdge->_producer, consumerEdge->_consumer);
        }

        outEdge->_childEdge->_parentEdge = nullptr;
        outEdge->_output->_producerEdge = outEdge->_childEdge;

        IE_ASSERT(outEdge->_ptrPosInModel != _outEdgePtrs.end());
        _outEdgePtrs.erase(outEdge->_ptrPosInModel);
    }

    IE_ASSERT(startInd >= 0 && endInd > startInd && static_cast<size_t>(startInd) <= parentStage->_outputEdges.size());
    parentStage->_outputEdges.erase(
        parentStage->_outputEdges.begin() + startInd,
        parentStage->_outputEdges.begin() + endInd);

    for (size_t i = 0; i < parentStage->_outputEdges.size(); ++i) {
        parentStage->_outputEdges[i]->_portInd = checked_cast<int>(i);
    }

    //
    // Remove Injected Temp Buffer Edges from parent Stage.
    //

    startInd = -1;
    endInd = -1;

    for (const auto& tempBufferEdge : parentStage->_tempBufferEdges) {
        if (tempBufferEdge->_childEdge == nullptr) {
            IE_ASSERT(startInd < 0);
            continue;
        }

        if (startInd >= 0 && endInd >= 0) {
            IE_ASSERT(tempBufferEdge->_childEdge->_stage != childStage);
        }

        if (tempBufferEdge->_childEdge->_stage != childStage) {
            if (startInd >= 0 && endInd < 0) {
                endInd = tempBufferEdge->_portInd;
            }
            continue;
        }

        if (startInd < 0) {
            startInd = tempBufferEdge->_portInd;
        }
        if (tempBufferEdge->_portInd == checked_cast<int>(parentStage->_tempBufferEdges.size()) - 1) {
            endInd = tempBufferEdge->_portInd + 1;
        }

        tempBufferEdge->_childEdge->_parentEdge = nullptr;
        tempBufferEdge->_tempBuffer->_tempBufferEdge = tempBufferEdge->_childEdge;

        IE_ASSERT(tempBufferEdge->_ptrPosInModel != _tempBufferEdgePtrs.end());
        _tempBufferEdgePtrs.erase(tempBufferEdge->_ptrPosInModel);
    }

    if (startInd >= 0) {
        IE_ASSERT(endInd > startInd && static_cast<size_t>(startInd) <= parentStage->_tempBufferEdges.size());
        parentStage->_tempBufferEdges.erase(
            parentStage->_tempBufferEdges.begin() + startInd,
            parentStage->_tempBufferEdges.begin() + endInd);

        for (size_t i = 0; i < parentStage->_tempBufferEdges.size(); ++i) {
            parentStage->_tempBufferEdges[i]->_portInd = checked_cast<int>(i);
        }
    }

    //
    // Remove the InjectedStage Edge from the Model.
    //

    IE_ASSERT(edge->_ptrPosInModel != _stageEdgePtrs.end());
    _stageEdgePtrs.erase(edge->_ptrPosInModel);

    //
    // Update Cache
    //

    updateInputStagesCache(parentStage);
    updateOutputStagesCache(parentStage);

    updateInputStagesCache(childStage);
    updateOutputStagesCache(childStage);

    resetStageOrderCache(parentStage);
    resetStageOrderCache(childStage);
}

//
// Model : Stage nodes removal
//

void Model::disconnectStageDatas(const Stage& stage) {
    //
    // Check that objects belong to the same Model.
    //

    IE_ASSERT(stage->_model.get() == this);

    //
    // Disconnect input datas.
    //

    for (const auto& inEdge : stage->_inputEdges) {
        if (inEdge->_input->_producerEdge != nullptr) {
            eraseFromStageOrderCache(inEdge->_input->_producerEdge->_producer, inEdge->_consumer);
        }

        inEdge->_input->_consumerEdges.erase(inEdge);

        IE_ASSERT(inEdge->_ptrPosInModel != _inEdgePtrs.end());
        _inEdgePtrs.erase(inEdge->_ptrPosInModel);
    }

    stage->_inputEdges.clear();

    //
    // Disconnect output datas.
    //

    for (const auto& outEdge : stage->_outputEdges) {
        for (const auto& consumerEdge : outEdge->_output->_consumerEdges) {
            eraseFromStageOrderCache(outEdge->_producer, consumerEdge->_consumer);
        }

        outEdge->_output->_producerEdge = nullptr;

        IE_ASSERT(outEdge->_ptrPosInModel != _outEdgePtrs.end());
        _outEdgePtrs.erase(outEdge->_ptrPosInModel);
    }

    stage->_outputEdges.clear();

    //
    // Disconnect temp datas.
    //

    for (const auto& tempBufferEdge : stage->_tempBufferEdges) {
        tempBufferEdge->_tempBuffer->_tempBufferEdge = nullptr;

        IE_ASSERT(tempBufferEdge->_ptrPosInModel != _tempBufferEdgePtrs.end());
        _tempBufferEdgePtrs.erase(tempBufferEdge->_ptrPosInModel);
    }

    stage->_tempBufferEdges.clear();

    //
    // Notify allocator
    //

    _allocator.setNeedToAllocNonIntermData();

    //
    // Update Cache
    //

    eraseFromInputOutputStagesCache(stage);
    resetStageOrderCache(stage);
}

//
// Model : Data nodes removal
//

void Model::cleanUpDatas() {
    bool needAllocatorPreprocess = false;

    for (const auto& data : datas()) {
        if (data->_usage == DataUsage::Input) {
            IE_ASSERT(!data->_consumerEdges.empty());
            IE_ASSERT(data->_parentDataEdge == nullptr);
        } else if (data->_usage == DataUsage::Output) {
            IE_ASSERT(data->_producerEdge != nullptr);
            IE_ASSERT(data->_parentDataEdge == nullptr);
        } else if (data->_usage == DataUsage::Temp) {
            if (data->_tempBufferEdge == nullptr) {
                _cache.dataHandles.erase(data);

                IE_ASSERT(data->_ptrPosInModel != _dataPtrs.end());
                _dataPtrs.erase(data->_ptrPosInModel);
            }
        } else {
            if (data->_consumerEdges.empty() && data->_producerEdge == nullptr) {
                if (data->usage() != DataUsage::Intermediate) {
                    needAllocatorPreprocess = true;
                }

                _cache.dataHandles.erase(data);

                IE_ASSERT(data->_ptrPosInModel != _dataPtrs.end());
                _dataPtrs.erase(data->_ptrPosInModel);
            }
        }
    }

    if (needAllocatorPreprocess) {
        _allocator.setNeedToAllocNonIntermData();
    }
}

void Model::removeUnusedData(const Data& data) {
    IE_ASSERT(data->numConsumers() == 0);

    if (data->usage() != DataUsage::Intermediate &&
        data->usage() != DataUsage::Temp) {
        _allocator.setNeedToAllocNonIntermData();
    }

    _cache.dataHandles.erase(data);

    IE_ASSERT(data->_ptrPosInModel != _dataPtrs.end());
    _dataPtrs.erase(data->_ptrPosInModel);
}

//
// Model : Sub-graphs accessors
//

void Model::initSingleSubGraph() const {
    if (_subGraphs.size() == 1) {
        return;
    }

    IE_ASSERT(_subGraphs.empty());

    _subGraphs.emplace_back(new SubGraph(handle_from_this(), 0));
    _cache.subGraphHandles.push_back(_subGraphs.back());
}

void Model::splitOntoSubGraphs() {
    IE_ASSERT(_subGraphs.size() == 1);

    auto numSubGraphs = attrs().getOrDefault<int>("numSubGraphs", 1);
    if (numSubGraphs == 1) {
        return;
    }

    auto& oldSubGraph = _subGraphs.front();

    std::vector<SubGraphPtr> newSubGraphs(checked_cast<size_t>(numSubGraphs));
    for (size_t i = 0; i < newSubGraphs.size(); ++i) {
        newSubGraphs[i].reset(new SubGraph(handle_from_this(), i));
    }

    for (const auto& stage : oldSubGraph->_stagePtrs) {
        auto stageSubGraphInd = stage->attrs().get<int>("subGraphInd");
        IE_ASSERT(stageSubGraphInd < numSubGraphs);

        auto& newSubGraph = newSubGraphs[checked_cast<size_t>(stageSubGraphInd)];

        stage->_subGraph = newSubGraph->handle_from_this();
        stage->_ptrPosInSubGraph = newSubGraph->_stagePtrs.emplace(newSubGraph->_stagePtrs.end(), stage);
    }
    oldSubGraph->_stagePtrs.clear();

    for (const auto& newSubGraph : newSubGraphs) {
        for (const auto& stage : newSubGraph->_stagePtrs) {
            updateInputStagesCache(stage);
            updateOutputStagesCache(stage);
        }
    }

    _subGraphs = std::move(newSubGraphs);

    _cache.subGraphHandles.clear();
    for (const auto& subGraph : _subGraphs) {
        _cache.subGraphHandles.push_back(subGraph);
    }
}

void Model::mergeSubGraphs() {
    if (_subGraphs.size() <= 1) {
        return;
    }

    SubGraphPtr newSubGraph(new SubGraph(handle_from_this(), 0));

    for (const auto& oldSubGraph : _subGraphs) {
        for (const auto& stage : oldSubGraph->_stagePtrs) {
            stage->_subGraph = newSubGraph->handle_from_this();
            stage->_ptrPosInSubGraph = newSubGraph->_stagePtrs.emplace(newSubGraph->_stagePtrs.end(), stage);
        }

        oldSubGraph->_stagePtrs.clear();
    }

    for (const auto& stage : newSubGraph->_stagePtrs) {
        updateInputStagesCache(stage);
        updateOutputStagesCache(stage);
    }

    _subGraphs.clear();
    _subGraphs.emplace_back(std::move(newSubGraph));

    _cache.subGraphHandles.clear();
    _cache.subGraphHandles.push_back(_subGraphs.back());
}

}  // namespace vpu
