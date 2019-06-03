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
#include <set>
#include <utility>

#include <vpu/model/base.hpp>
#include <vpu/model/edges.hpp>
#include <vpu/model/data.hpp>
#include <vpu/model/stage.hpp>
#include <vpu/utils/enums.hpp>
#include <vpu/utils/range.hpp>
#include <vpu/utils/containers.hpp>
#include <vpu/utils/func_ref.hpp>
#include <vpu/utils/numeric.hpp>

namespace vpu {

class SubGraph final : public EnableHandleFromThis<SubGraph> {
    //
    // Attributes
    //

    VPU_MODEL_ATTRIBUTE(ModelHandle, model, nullptr)
    VPU_MODEL_ATTRIBUTE(int, index, 0)

public:
    using Ptr = SubGraphPtr;

    //
    // Constructor(s) / Destructor
    //

    ~SubGraph();

    SubGraph(const SubGraph&) = delete;
    SubGraph& operator=(const SubGraph&) = delete;

    inline SubGraph(SubGraph&&) = delete;
    inline SubGraph& operator=(SubGraph&&) = delete;

    //
    // Data nodes
    //

    Data addConstData(
            const std::string& name,
            const DataDesc& desc,
            const DataContent::Ptr& content);

    Data addNewData(
            const std::string& name,
            const DataDesc& desc);

    Data addFakeData();

    Data duplicateData(
            const Data& origData,
            const std::string& postfix,
            const DataDesc& newDesc = DataDesc(),
            const DataContent::Ptr& newContent = nullptr);

    //
    // Stage nodes
    //

    template <class StageImpl>
    Stage addNewStage(
            const std::string& name,
            StageType type,
            const ie::CNNLayerPtr& origLayer,
            const DataVector& inputs,
            const DataVector& outputs);

    Stage duplicateStage(
            const std::string& name,
            const Stage& origStage,
            const DataVector& inputs,
            const DataVector& outputs);

    //
    // Stage <-> Data edges
    //

    StageInput addStageInput(
            const Stage& stage,
            const Data& data);

    StageOutput addStageOutput(
            const Stage& stage,
            const Data& data);

    StageTempBuffer addTempBuffer(
            const Stage& stage,
            const DataDesc& desc);

    void replaceStageInput(
            const StageInput& edge,
            const Data& newInput);

    void replaceStageOutput(
            const StageOutput& edge,
            const Data& newOutput);

    //
    // Data <-> Data edges
    //

    DataEdgeBuilder connectDatas();

    //
    // Stage <-> Stage edges
    //

    InjectedStageEdgeBuilder injectStage();

    void revertInjection(const InjectedStage& edge);

    //
    // Stage nodes removal
    //

    void disconnectStageDatas(const Stage& stage);

    void removeStage(const Stage& stage);

    //
    // Stage order
    //

    void buildStageOrder(StageOrder order = StageOrder::DFS) const;

    //
    // Stage nodes accessors
    //

    inline int numStages() const { return checked_cast<int>(_stagePtrs.size()); }

    inline auto getStages(StageOrder order = StageOrder::DFS) const -> decltype(contRange(std::declval<StageList>())) {
        buildStageOrder(order);
        return contRange(_cache.orderedStages);
    }

    inline auto inputStages() const -> decltype(contRange(std::declval<StageNode::OrderedSet>())) {
        return contRange(_cache.inputStages);
    }

    inline auto outputStages() const -> decltype(contRange(std::declval<StageNode::OrderedSet>())) {
        return contRange(_cache.outputStages);
    }

private:
    struct Cache final {
        StageNode::OrderedSet inputStages;
        StageNode::OrderedSet outputStages;

        StageList orderedStages;
        bool resetStageOrder = true;
        StageOrder stageOrder = StageOrder::DFS;

        inline Cache() : orderedStages(&StageNode::_posInSubGraphCache) {}
    };

private:
    inline SubGraph(ModelHandle model, int index) :
            _model(model), _index(index), _posInModelCache(this) {
    }

    Stage addNewStageImpl(
            const std::string& name,
            StageType type,
            const ie::CNNLayerPtr& origLayer,
            const DataVector& inputs,
            const DataVector& outputs,
            const FuncRef<StagePtr()>& creator);

    void runDFS(
            const Stage& stage,
            StageMap<bool>& visitedMap) const;

    void runBFS(
            StageList& queue,
            StageMap<bool>& visitedMap) const;

private:
    StagePtrList _stagePtrs;

    mutable Cache _cache;

    IntrusivePtrListNode<SubGraph> _posInModelCache;

    friend class Model;
};

template <class StageImpl>
inline Stage SubGraph::addNewStage(
        const std::string& name,
        StageType type,
        const ie::CNNLayerPtr& origLayer,
        const DataVector& inputs,
        const DataVector& outputs) {
    return addNewStageImpl(
        name,
        type,
        origLayer,
        inputs,
        outputs,
        []() { return std::make_shared<StageImpl>(); });
}

}  // namespace vpu
