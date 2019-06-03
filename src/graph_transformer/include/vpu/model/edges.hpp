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

#include <vpu/model/base.hpp>

namespace vpu {

//
// StageInputEdge
//

//
// Data -> Stage edge.
//

class StageInputEdge final :
        public EnableHandleFromThis<StageInputEdge>,
        public EnableCustomAttributes {
    VPU_MODEL_ATTRIBUTE(Data, input, nullptr)
    VPU_MODEL_ATTRIBUTE(Stage, consumer, nullptr)
    VPU_MODEL_ATTRIBUTE(int, portInd, -1)
    VPU_MODEL_ATTRIBUTE(StageInput, parentEdge, nullptr)
    VPU_MODEL_ATTRIBUTE(StageInput, childEdge, nullptr)
    VPU_MODEL_ATTRIBUTE(ModelHandle, model, nullptr)

private:
    StageInputEdge() : _posInData(this) {}

private:
    StageInputPtrList::iterator _ptrPosInModel;
    IntrusivePtrListNode<StageInputEdge> _posInData;

    friend class Model;
    friend class SubGraph;
    friend class DataNode;
};

//
// StageOutputEdge
//

//
// Stage -> Data edge.
//

class StageOutputEdge final :
        public EnableHandleFromThis<StageOutputEdge>,
        public EnableCustomAttributes {
    VPU_MODEL_ATTRIBUTE(Stage, producer, nullptr)
    VPU_MODEL_ATTRIBUTE(Data, output, nullptr)
    VPU_MODEL_ATTRIBUTE(int, portInd, -1)
    VPU_MODEL_ATTRIBUTE(StageOutput, parentEdge, nullptr)
    VPU_MODEL_ATTRIBUTE(StageOutput, childEdge, nullptr)
    VPU_MODEL_ATTRIBUTE(ModelHandle, model, nullptr)

private:
    StageOutputPtrList::iterator _ptrPosInModel;

    friend class Model;
    friend class SubGraph;
};

//
// StageTempBufferEdge
//

class StageTempBufferEdge final :
        public EnableHandleFromThis<StageTempBufferEdge>,
        public EnableCustomAttributes {
    VPU_MODEL_ATTRIBUTE(Stage, stage, nullptr)
    VPU_MODEL_ATTRIBUTE(Data, tempBuffer, nullptr)
    VPU_MODEL_ATTRIBUTE(int, portInd, -1)
    VPU_MODEL_ATTRIBUTE(StageTempBuffer, parentEdge, nullptr)
    VPU_MODEL_ATTRIBUTE(StageTempBuffer, childEdge, nullptr)
    VPU_MODEL_ATTRIBUTE(ModelHandle, model, nullptr)

private:
    StageTempBufferPtrList::iterator _ptrPosInModel;

    friend class Model;
    friend class SubGraph;
};

//
// SharedAllocationEdge
//

//
// Data <-> Data edges - used to share memory buffer between Data objects.
// Parent Data object owns the memory, while child reuses it.
//
// SharedDataMode defines the relationship between the Data objects:
//    * ROI : child is a sub-tensor of parent.
//      They have the same layout and strides, but child has smaller dimensions.
//    * Reshape : used for Reshape operation.
//      Child shares the same memory buffer, but has completely different layout.
//
// SharedDataOrder defined the Data flow order between parent and child.
//    * ParentWritesToChild :
//      (Producer) -> [Parent] -> [Child] -> (Consumer)
//    * ChildWritesToParent :
//      (Producer) -> [Child] -> [Parent] -> (Consumer)
//

class SharedAllocationEdge final :
        public EnableHandleFromThis<SharedAllocationEdge>,
        public EnableCustomAttributes {
    VPU_MODEL_ATTRIBUTE(Data, parent, nullptr)
    VPU_MODEL_ATTRIBUTE(Data, child, nullptr)
    VPU_MODEL_ATTRIBUTE(Stage, connection, nullptr)
    VPU_MODEL_ATTRIBUTE(SharedDataMode, mode, SharedDataMode::ROI)
    VPU_MODEL_ATTRIBUTE(SharedDataOrder, order, SharedDataOrder::ParentWritesToChild)
    VPU_MODEL_ATTRIBUTE(ModelHandle, model, nullptr)

private:
    SharedAllocationEdge() : _posInData(this) {}

private:
    SharedAllocationPtrList::iterator _ptrPosInModel;
    IntrusivePtrListNode<SharedAllocationEdge> _posInData;

    friend class Model;
    friend class SubGraph;
    friend class DataNode;
};

//
// InjectedStageEdge
//

//
// Stage <-> Stage edges - used to inject SW operations into HW
//

class InjectedStageEdge final :
        public EnableHandleFromThis<InjectedStageEdge>,
        public EnableCustomAttributes {
    VPU_MODEL_ATTRIBUTE(Stage, parent, nullptr)
    VPU_MODEL_ATTRIBUTE(StagePtr, child, nullptr)
    VPU_MODEL_ATTRIBUTE(int, portInd, -1)
    VPU_MODEL_ATTRIBUTE(ModelHandle, model, nullptr)

private:
    InjectedStageEdge() : _posInStage(this) {}

private:
    InjectedStagePtrList::iterator _ptrPosInModel;
    IntrusivePtrListNode<InjectedStageEdge> _posInStage;

    friend class Model;
    friend class SubGraph;
    friend class StageNode;
};

}  // namespace vpu
