//
// Copyright 2018-2019 Intel Corporation.
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
#include <utility>

#include <ie_icnn_network.hpp>

#include <vpu/model/sub_graph.hpp>
#include <vpu/utils/enums.hpp>
#include <vpu/utils/io.hpp>
#include <vpu/utils/dot_io.hpp>
#include <vpu/utils/numeric.hpp>
#include <vpu/allocator.hpp>

namespace vpu {

namespace ie = InferenceEngine;

//
// Resources
//

// TODO: get rid of `cmxLimit`.

struct Resources final {
    int numCMXSlices = 0;
    int numSHAVEs = 0;
    int cmxLimit = 0;
};

void printTo(std::ostream& os, const Resources& res);
void printTo(DotLabel& lbl, const Resources& res);

//
// Model
//

class Model final :
        public EnableHandleFromThis<Model>,
        public EnableCustomAttributes {
    //
    // Main attributes
    //

    VPU_MODEL_ATTRIBUTE(std::string, name, std::string())

    VPU_MODEL_ATTRIBUTE(int, batchSize, 1)

    VPU_MODEL_ATTRIBUTE(ie::NetworkStatsMap, nodesStats, {})

public:
    using Ptr = ModelPtr;

    //
    // Constructor(s) / Destructor
    //

    inline explicit Model(const std::string& name) : _name(name) {}

    inline ~Model() = default;

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    inline Model(Model&&) = delete;
    inline Model& operator=(Model&&) = delete;

    //
    // Main attributes setters
    //

    inline void setBatchSize(int batchSize) {
        VPU_THROW_UNLESS(batchSize >= 1);
        _batchSize = batchSize;
        _allocator.setBatchSize(batchSize);
    }

    inline void setNodesStats(const ie::NetworkStatsMap& stats) {
        _nodesStats = stats;
    }

    //
    // Data nodes
    //

    inline Data addInputData(
            const std::string& name,
            const DataDesc& desc) {
        _allocator.setNeedToAllocNonIntermData();
        return createData(name, DataUsage::Input, desc);
    }

    inline Data addOutputData(
            const std::string& name,
            const DataDesc& desc) {
        _allocator.setNeedToAllocNonIntermData();
        return createData(name, DataUsage::Output, desc);
    }

    Data addConstData(
            const std::string& name,
            const DataDesc& desc,
            const DataContent::Ptr& content);

    inline Data addNewData(
            const std::string& name,
            const DataDesc& desc) {
        return createData(name, DataUsage::Intermediate, desc);
    }

    inline Data addFakeData() {
        return createData("<fake>", DataUsage::Fake, DataDesc({1}));
    }

    Data duplicateData(
            const Data& origData,
            const std::string& postfix,
            const DataDesc& newDesc = DataDesc(),
            const DataContent::Ptr& newContent = nullptr);

    //
    // Stage nodes
    //

    template <class StageImpl>
    inline Stage addNewStage(
            const std::string& name,
            StageType type,
            const ie::CNNLayerPtr& origLayer,
            const DataVector& inputs,
            const DataVector& outputs) {
        initSingleSubGraph();
        return _subGraphs[0]->addNewStage<StageImpl>(name, type, origLayer, inputs, outputs);
    }

    inline Stage duplicateStage(
            const std::string& name,
            const Stage& origStage,
            const DataVector& inputs,
            const DataVector& outputs) {
        IE_ASSERT(_subGraphs.size() == 1);
        return _subGraphs[0]->duplicateStage(name, origStage, inputs, outputs);
    }

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

    inline DataEdgeBuilder connectDatas() { return DataEdgeBuilder(handle_from_this()); }

    //
    // Stage <-> Stage edges
    //

    inline InjectedStageEdgeBuilder injectStage() { return InjectedStageEdgeBuilder(handle_from_this()); }

    void revertInjection(const InjectedStage& edge);

    //
    // Stage nodes removal
    //

    void disconnectStageDatas(const Stage& stage);

    inline void removeStage(const Stage& stage) {
        IE_ASSERT(_subGraphs.size() == 1);
        _subGraphs[0]->removeStage(stage);
    }

    //
    // Data nodes removal
    //

    void cleanUpDatas();

    void removeUnusedData(const Data& data);

    //
    // Stage order
    //

    inline void buildStageOrder(StageOrder order = StageOrder::DFS) const {
        initSingleSubGraph();
        _subGraphs[0]->buildStageOrder(order);
    }

    //
    // Stage nodes accessors
    //

    inline int numStages() const {
        initSingleSubGraph();
        return _subGraphs[0]->numStages();
    }

    inline auto getStages(StageOrder order = StageOrder::DFS) const -> decltype(contRange(std::declval<StageList>())) {
        initSingleSubGraph();
        return _subGraphs[0]->getStages(order);
    }

    inline auto inputStages() const -> decltype(contRange(std::declval<StageNode::OrderedSet>())) {
        initSingleSubGraph();
        return _subGraphs[0]->inputStages();
    }

    inline auto outputStages() const -> decltype(contRange(std::declval<StageNode::OrderedSet>())) {
        initSingleSubGraph();
        return _subGraphs[0]->outputStages();
    }

    //
    // Data nodes accessors
    //

    inline int numDatas() const { return checked_cast<int>(_dataPtrs.size()); }

    inline auto datas() const -> decltype(contRange(std::declval<DataList>())) {
        return contRange(_cache.dataHandles);
    }

    //
    // Sub-graphs accessors
    //

    void splitOntoSubGraphs();
    void mergeSubGraphs();

    inline int numSubGraphs() const { return checked_cast<int>(_subGraphs.size()); }

    inline auto subGraphs() const -> decltype(contRange(std::declval<IntrusivePtrList<SubGraph>>())) {
        return contRange(_cache.subGraphHandles);
    }

    inline SubGraphHandle subGraph(int ind) const {
        IE_ASSERT(ind >= 0 && static_cast<size_t>(ind) < _subGraphs.size());
        return _subGraphs[static_cast<size_t>(ind)];
    }

    //
    // Allocator
    //

    inline Allocator& getAllocator() { return _allocator; }

private:
    struct Cache final {
        DataList dataHandles;
        IntrusivePtrList<SubGraph> subGraphHandles;

        inline Cache() : dataHandles(&DataNode::_posInModelCache), subGraphHandles(&SubGraph::_posInModelCache) {}
    };

private:
    void initSingleSubGraph() const;

    static void resetStageOrderCache(const Stage& stage);
    static void updateStageOrderCache(const Stage& producer, const Stage& consumer);
    static void eraseFromStageOrderCache(const Stage& producer, const Stage& consumer);

    static void updateInputStagesCache(const Stage& stage);
    static void updateOutputStagesCache(const Stage& stage);
    static void eraseFromInputOutputStagesCache(const Stage& stage);

    Data createData(
            const std::string& name,
            DataUsage usage,
            const DataDesc& desc);

    SharedAllocation connectDatasImpl(
            const Data& parent,
            const Data& child,
            SharedDataMode mode,
            SharedDataOrder order,
            const DimValues& offset);

    InjectedStage injectStageImpl(
            const Stage& parent,
            const Stage& child);

private:
    DataPtrList _dataPtrs;

    StageInputPtrList _inEdgePtrs;
    StageOutputPtrList _outEdgePtrs;
    StageTempBufferPtrList _tempBufferEdgePtrs;
    SharedAllocationPtrList _dataEdgePtrs;
    InjectedStagePtrList _stageEdgePtrs;

    mutable std::vector<SubGraphPtr> _subGraphs;

    Allocator _allocator;

    mutable Cache _cache;

    friend class SubGraph;
    friend class DataEdgeBuilder;
    friend class InjectedStageEdgeBuilder;
};

//
// runAllocator
//

AllocationResult runAllocator(
        const Model::Ptr& model,
        bool onlyCheckCMX = false);

}  // namespace vpu
