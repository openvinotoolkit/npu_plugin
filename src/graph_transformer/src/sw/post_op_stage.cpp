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

#include <vpu/sw/post_op_stage.hpp>

#include <memory>

#include <vpu/model/edges.hpp>
#include <vpu/model/data.hpp>

namespace vpu {

void PostOpStage::propagateDataOrderImpl() const {
    IE_ASSERT(!_inputEdges.empty());
    IE_ASSERT(_outputEdges.size() == 1);

    // Non-zero-port inputs are constant (scales/biases).
    for (const auto& inEdge : _inputEdges) {
        if (inEdge->portInd() > 0) {
            IE_ASSERT(inEdge->input()->usage() == DataUsage::Const);
        }
    }

    auto input = _inputEdges[0]->input();

    auto inDimsOrder = input->desc().dimsOrder();

    // TODO: support HCW on firmware side
    if (inDimsOrder.dimInd(Dim::C) == 1) {
        inDimsOrder = inDimsOrder.createMovedDim(Dim::C, 2);  // CHW
        _orderInfo.setInput(_inputEdges[0], inDimsOrder);
    }

    _orderInfo.setOutput(_outputEdges[0], inDimsOrder);
}

void PostOpStage::getDataStridesRequirementsImpl() const {
    IE_ASSERT(!_inputEdges.empty());
    IE_ASSERT(_outputEdges.size() == 1);

    // Non-zero-port inputs are constant (scales/biases).
    for (const auto& inEdge : _inputEdges) {
        if (inEdge->portInd() > 0) {
            IE_ASSERT(inEdge->input()->usage() == DataUsage::Const);
        }
    }

    auto input = _inputEdges[0]->input();

    StridesRequirement reqs;

    // Current PostOp implementation requires Compact major stride.
    reqs.add(2, DimStride::Compact);

    if (input->desc().dim(Dim::N, 1) > 1) {
        // To merge batch into previous dimension.
        reqs.add(input->desc().dimsOrder().dimInd(Dim::N), DimStride::Compact);
    }

    _stridesInfo.setInput(_inputEdges[0], reqs);
    _stridesInfo.setOutput(_outputEdges[0], reqs);
}

void PostOpStage::finalizeDataLayoutImpl() {
}

void PostOpStage::getBatchSupportInfoImpl() const {
    IE_ASSERT(!_inputEdges.empty());
    IE_ASSERT(_outputEdges.size() == 1);

    // Non-zero-port inputs are constant (scales/biases).
    for (const auto& inEdge : _inputEdges) {
        if (inEdge->portInd() > 0) {
            IE_ASSERT(inEdge->input()->usage() == DataUsage::Const);
        }
    }

    auto mainDesc = _inputEdges[0]->input()->desc();

    // PostOp will support batch by merging it with previous dimension.
    for (const auto& inEdge : _inputEdges) {
        auto input = inEdge->input();

        if (inEdge->portInd() == 0)
            continue;

        if (input->desc().dimsOrder().dimInd(Dim::C) == input->desc().numDims() - 2) {
            IE_ASSERT(input->desc().totalDimSize() == input->desc().dim(Dim::C));
            _batchInfo.setInput(inEdge, BatchSupport::ReplicateConstContent);
        }
    }
}

StageSHAVEsRequirements PostOpStage::getSHAVEsRequirementsImpl() const {
    // TODO: more SHAVEs leads to hang on public MTCNN network with U8 input
    return StageSHAVEsRequirements::TwoOrOne;
}

void PostOpStage::finalCheckImpl() const {
}

void PostOpStage::serializeDataImpl(BlobSerializer& serializer) const {
    IE_ASSERT(!_inputEdges.empty());
    IE_ASSERT(_outputEdges.size() == 1);
    IE_ASSERT(_tempBufferEdges.empty());

    auto input = _inputEdges[0]->input();
    auto output = _outputEdges[0]->output();

    if (input->desc().dimsOrder() == DimsOrder::NC) {
        input->serializeOldBuffer(
            handle_from_this(),
            serializer,
            DimsOrder::HWC,
            {
                {Dim::W, {Dim::N}},
                {Dim::C, {Dim::C}}
            });

        output->serializeOldBuffer(
            handle_from_this(),
            serializer,
            DimsOrder::HWC,
            {
                {Dim::W, {Dim::N}},
                {Dim::C, {Dim::C}}
            });
    } else if (input->desc().dim(Dim::N, 1) > 1) {
        auto perm = input->desc().dimsOrder().toPermutation();
        IE_ASSERT(perm.size() == 4);

        input->serializeOldBuffer(
            handle_from_this(),
            serializer,
            DimsOrder::HWC,
            {
                {Dim::H, {perm[2], perm[3]}},
                {Dim::W, {perm[1]}},
                {Dim::C, {perm[0]}}
            });

        output->serializeOldBuffer(
            handle_from_this(),
            serializer,
            DimsOrder::HWC,
            {
                {Dim::H, {perm[2], perm[3]}},
                {Dim::W, {perm[1]}},
                {Dim::C, {perm[0]}}
            });
    } else {
        input->serializeOldBuffer(handle_from_this(), serializer);

        output->serializeOldBuffer(handle_from_this(), serializer);
    }

    for (int i = 1; i < _inputEdges.size(); ++i) {
        _inputEdges[i]->input()->serializeOldBuffer(handle_from_this(), serializer);
    }
}

}  // namespace vpu
