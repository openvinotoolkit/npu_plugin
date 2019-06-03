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
#include <unordered_set>
#include <memory>
#include <set>

namespace vpu {

namespace {

class RegionYoloStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<RegionYoloStage>(*this);
    }

    void propagateDataOrderImpl() const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto output = _outputEdges[0]->output();

        if (!attrs().get<bool>("doSoftMax")) {
            _orderInfo.setOutput(_outputEdges[0], output->desc().dimsOrder().createMovedDim(Dim::C, 2));  // CHW
        }
    }

    void getDataStridesRequirementsImpl() const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        if (attrs().get<bool>("doSoftMax")) {
            // Major dimension must be compact.
            _stridesInfo.setInput(_inputEdges[0], StridesRequirement().add(2, DimStride::Compact));
        }
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl() const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        _batchInfo.setInput(_inputEdges[0], BatchSupport::Split);
        _batchInfo.setOutput(_outputEdges[0], BatchSupport::Split);
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto classes = attrs().get<int>("classes");
        auto coords = attrs().get<int>("coords");
        auto num = attrs().get<int>("num");
        auto maskSize = attrs().get<int>("maskSize");
        auto doSoftMax = attrs().get<bool>("doSoftMax");

        serializer.append(static_cast<int32_t>(classes));
        serializer.append(static_cast<int32_t>(coords));
        serializer.append(static_cast<int32_t>(num));
        serializer.append(static_cast<int32_t>(maskSize));
        serializer.append(static_cast<int32_t>(doSoftMax));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);
        IE_ASSERT(_tempBufferEdges.empty());

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        input->serializeOldBuffer(handle_from_this(), serializer);
        output->serializeOldBuffer(handle_from_this(), serializer);
    }
};

}  // namespace

void FrontEnd::parseRegionYolo(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto mask = layer->GetParamAsInts("mask", {});

    auto stage = model->addNewStage<RegionYoloStage>(
        layer->name,
        StageType::RegionYolo,
        layer,
        inputs,
        outputs);

    stage->attrs().set<int>("classes", layer->GetParamAsInt("classes", 20));
    stage->attrs().set<int>("coords", layer->GetParamAsInt("coords", 4));
    stage->attrs().set<int>("num", layer->GetParamAsInt("num", 5));
    stage->attrs().set<int>("maskSize", static_cast<int>(mask.size()));
    stage->attrs().set<bool>("doSoftMax", layer->GetParamAsInt("do_softmax", 1));
}

}  // namespace vpu
