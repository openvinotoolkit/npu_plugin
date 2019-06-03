//
// Copyright 2017-2018 Intel Corporation.
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

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <utility>
#include <unordered_set>
#include <memory>
#include <set>

namespace vpu {

namespace {

template <class Cont1, class Cont2>
SmallVector<typename Cont1::value_type, MAX_DIMS_64> permuteArray(const Cont1& src, const Cont2& permutation) {
    SmallVector<typename Cont1::value_type, MAX_DIMS_64> out(permutation.size());

    for (int i = 0; i < out.size(); i++) {
        auto newInd = static_cast<int>(permutation[i]);

        IE_ASSERT(newInd >= 0);
        IE_ASSERT(newInd < src.size());

        out[i] = src[newInd];
    }

    return out;
}

class PermuteStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<PermuteStage>(*this);
    }

    void propagateScaleFactorsImpl(
            const SmallVector<float>& inputScales,
            ScalePropagationStep step) override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        if (step == ScalePropagationStep::Propagate) {
            _scaleInfo.setOutput(_outputEdges[0], inputScales[0]);
        } else {
            // Copy can only propagate scaling.
            _scaleInfo.setInput(_inputEdges[0], 1.0f);
            _scaleInfo.setOutput(_outputEdges[0], 1.0f);
        }
    }

    void propagateDataOrderImpl() const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();

        _orderInfo.setOutput(_outputEdges[0], input->desc().dimsOrder());
    }

    void getDataStridesRequirementsImpl() const override {
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl() const override {
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::CanBeLimited;
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 1);

        auto input = _inputEdges[0]->input();

        const auto& order = attrs().get<SmallVector<int, MAX_DIMS_64>>("order");

        auto perm = input->desc().dimsOrder().toPermutation();
        auto ind = input->desc().dimsOrder().toIndices();

        auto dimPerm = permuteArray(order, perm);
        auto memoryOrderPerm = permuteArray(ind.toVector(-1), dimPerm);

        int i = 0;
        for (i = 0; i < memoryOrderPerm.size(); i++) {
            serializer.append(static_cast<uint32_t>(memoryOrderPerm[i]));
        }
        for (; i < MAX_DIMS_32; i++) {
            serializer.append(static_cast<uint32_t>(-1));
        }
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);
        IE_ASSERT(_tempBufferEdges.empty());

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        input->serializeNewBuffer(serializer);
        output->serializeNewBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parsePermute(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto ieOrder = layer->GetParamAsInts("order");

    auto maxIeOrder = *std::max_element(ieOrder.begin(), ieOrder.end());

    SmallVector<int, MAX_DIMS_64> vpuOrder(MAX_DIMS_64, -1);
    for (size_t i = 0; i < ieOrder.size(); i++) {
        vpuOrder[i] = maxIeOrder - ieOrder[ieOrder.size() - 1 - i];
    }

    auto input = inputs[0];
    auto output = outputs[0];

    auto stage = model->addNewStage<PermuteStage>(
        layer->name,
        StageType::Permute,
        layer,
        inputs,
        outputs);

    stage->attrs().set<SmallVector<int, MAX_DIMS_64>>("order", vpuOrder);
}

}  // namespace vpu
