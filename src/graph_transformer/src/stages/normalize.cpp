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
#include <map>
#include <unordered_set>
#include <memory>
#include <set>

namespace vpu {

namespace {

class NormalizeStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<NormalizeStage>(*this);
    }

    void propagateDataOrderImpl() const override {
        IE_ASSERT(_inputEdges.size() == 2);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();

        _orderInfo.setOutput(_outputEdges[0], input->desc().dimsOrder());
    }

    void getDataStridesRequirementsImpl() const override {
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl() const override {
        IE_ASSERT(_inputEdges.size() == 2);
        IE_ASSERT(_outputEdges.size() == 1);

        _batchInfo.setInput(_inputEdges[0], BatchSupport::Split);
        _batchInfo.setOutput(_outputEdges[0], BatchSupport::Split);
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto acrossSpatial = attrs().get<bool>("acrossSpatial");
        auto channelShared = attrs().get<bool>("channelShared");
        auto eps = attrs().get<float>("eps");

        serializer.append(static_cast<int32_t>(acrossSpatial));
        serializer.append(static_cast<int32_t>(channelShared));
        serializer.append(static_cast<float>(eps));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 2);
        IE_ASSERT(_outputEdges.size() == 1);
        IE_ASSERT(_tempBufferEdges.empty());

        auto input = _inputEdges[0]->input();
        auto scales = _inputEdges[1]->input();
        auto output = _outputEdges[0]->output();

        input->serializeOldBuffer(handle_from_this(), serializer);
        output->serializeOldBuffer(handle_from_this(), serializer);
        scales->serializeOldBuffer(handle_from_this(), serializer);
    }
};

}  // namespace

void FrontEnd::parseNormalize(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto acrossSpatial = layer->GetParamAsInt("across_spatial", 0);
    auto channelShared = layer->GetParamAsInt("channel_shared", 0);
    float eps = layer->GetParamAsFloat("eps", 0.0f);

    auto weightsIt = layer->blobs.find("weights");
    if (weightsIt == layer->blobs.end()) {
        VPU_THROW_EXCEPTION << "Missing weights for " << layer->name << " layer";
    }

    auto weightsBlob = weightsIt->second;
    IE_ASSERT(weightsBlob != nullptr);

    auto output = outputs[0];

    auto scales = model->addConstData(
        layer->name + "@scales",
        DataDesc({weightsBlob->size()}),
        ieBlobContent(weightsBlob));

    auto stage = model->addNewStage<NormalizeStage>(
        layer->name,
        StageType::Normalize,
        layer,
        {inputs[0], scales},
        outputs);

    stage->attrs().set<bool>("acrossSpatial", acrossSpatial);
    stage->attrs().set<bool>("channelShared", channelShared);
    stage->attrs().set<float>("eps", eps);
}

}  // namespace vpu
