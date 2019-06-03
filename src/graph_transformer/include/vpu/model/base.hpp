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

#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <list>
#include <queue>
#include <stack>

#include <vpu/model/data_desc.hpp>
#include <vpu/utils/extra.hpp>
#include <vpu/utils/enums.hpp>
#include <vpu/utils/handle.hpp>
#include <vpu/utils/attributes_map.hpp>
#include <vpu/utils/range.hpp>
#include <vpu/utils/containers.hpp>

namespace vpu {

//
// VPU_DEFINE_MODEL_TYPES
//

#define VPU_DEFINE_MODEL_TYPES(type, postfix)                                                       \
    using type = Handle<VPU_COMBINE(type, postfix)>;                                                \
    \
    using VPU_COMBINE(type, Vector) = SmallVector<type>;                                            \
    \
    using VPU_COMBINE(type, List) = IntrusivePtrList<VPU_COMBINE(type, postfix)>;                   \
    \
    using VPU_COMBINE(type, Set) = std::unordered_set<type, HandleHash>;                            \
    \
    template <typename Val>                                                                         \
    using VPU_COMBINE(type, Map) = std::unordered_map<type, Val, HandleHash>;                       \
    \
    using VPU_COMBINE(type, Ptr) = std::shared_ptr<VPU_COMBINE(type, postfix)>;                     \
    \
    using VPU_COMBINE(type, PtrList) = std::list<VPU_COMBINE(type, Ptr)>;

//
// VPU_MODEL_ATTRIBUTE
//

#define VPU_MODEL_ATTRIBUTE(type, name, defVal)                                 \
    protected:                                                                  \
        type VPU_COMBINE(_, name) = defVal;                                     \
    public:                                                                     \
        inline const type& name() const {                                       \
            return VPU_COMBINE(_, name);                                        \
        }

#define VPU_MODEL_ATTRIBUTE_RANGE(type, name)                                   \
    protected:                                                                  \
        type VPU_COMBINE(_, name);                                              \
    public:                                                                     \
        inline auto name() const -> decltype(contRange(VPU_COMBINE(_, name))) { \
            return contRange(VPU_COMBINE(_, name));                             \
        }

//
// Forward declaration
//

class GraphTransformerImpl;

class Model;
using ModelPtr = std::shared_ptr<Model>;
using ModelHandle = Handle<Model>;

class SubGraph;
using SubGraphPtr = std::shared_ptr<SubGraph>;
using SubGraphHandle = Handle<SubGraph>;

class DataNode;
VPU_DEFINE_MODEL_TYPES(Data, Node)

class StageNode;
VPU_DEFINE_MODEL_TYPES(Stage, Node)

class StageInputEdge;
VPU_DEFINE_MODEL_TYPES(StageInput, Edge)

class StageOutputEdge;
VPU_DEFINE_MODEL_TYPES(StageOutput, Edge)

class StageTempBufferEdge;
VPU_DEFINE_MODEL_TYPES(StageTempBuffer, Edge)

class SharedAllocationEdge;
VPU_DEFINE_MODEL_TYPES(SharedAllocation, Edge)

class InjectedStageEdge;
VPU_DEFINE_MODEL_TYPES(InjectedStage, Edge)

//
// StageOrder
//

VPU_DECLARE_ENUM(StageOrder,
    DFS,
    BFS
);

//
// SharedAllocation
//

VPU_DECLARE_ENUM(SharedDataMode,
    ROI,
    Reshape)

VPU_DECLARE_ENUM(SharedDataOrder,
    ParentWritesToChild,
    ChildWritesToParent)

//
// Edge builders
//

class DataEdgeBuilder final {
public:
    inline DataEdgeBuilder(DataEdgeBuilder&&) = default;

    DataEdgeBuilder(const DataEdgeBuilder&) = delete;
    DataEdgeBuilder& operator=(const DataEdgeBuilder&) = delete;
    DataEdgeBuilder& operator=(DataEdgeBuilder&&) = delete;

    ~DataEdgeBuilder();

    DataEdgeBuilder& parent(const Data& parent);
    DataEdgeBuilder& child(const Data& child);

    DataEdgeBuilder& mode(SharedDataMode mode);
    DataEdgeBuilder& order(SharedDataOrder order);

    DataEdgeBuilder& offset(const DimValues& offset);

    SharedAllocation done();

private:
    inline explicit DataEdgeBuilder(ModelHandle model) : _model(model) {}

private:
    ModelHandle _model;

    Data _parent;
    Data _child;

    SharedDataMode _mode = SharedDataMode::ROI;
    bool _modeSet = false;

    SharedDataOrder _order = SharedDataOrder::ParentWritesToChild;
    bool _orderSet = false;

    DimValues _offset;
    bool _offsetSet = false;

    friend class Model;
};

class InjectedStageEdgeBuilder final {
public:
    inline InjectedStageEdgeBuilder(InjectedStageEdgeBuilder&&) = default;

    InjectedStageEdgeBuilder(const InjectedStageEdgeBuilder&) = delete;
    InjectedStageEdgeBuilder& operator=(const InjectedStageEdgeBuilder&) = delete;
    InjectedStageEdgeBuilder& operator=(InjectedStageEdgeBuilder&&) = delete;

    ~InjectedStageEdgeBuilder();

    InjectedStageEdgeBuilder& parentHW(const Stage& parent);
    InjectedStageEdgeBuilder& childSW(const Stage& child);

    InjectedStage done();

private:
    inline explicit InjectedStageEdgeBuilder(ModelHandle model) : _model(model) {}

private:
    ModelHandle _model;

    Stage _parent;
    Stage _child;

    friend class Model;
};

}  // namespace vpu
