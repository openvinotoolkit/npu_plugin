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

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <unordered_set>
#include <memory>
#include <set>
#include <string>

namespace vpu {

VPU_DECLARE_ENUM(ResampleType,
    Nearest  = 0,
    Bilinear = 1
)

namespace {

class ResampleStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<ResampleStage>(*this);
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
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        _batchInfo.setInput(_inputEdges[0], BatchSupport::Split);
        _batchInfo.setOutput(_outputEdges[0], BatchSupport::Split);
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto antialias = attrs().get<bool>("antialias");
        auto factor = attrs().get<float>("factor");
        auto sampleType = attrs().get<ResampleType>("type");

        serializer.append(static_cast<int32_t>(antialias));
        serializer.append(static_cast<float>(factor));
        serializer.append(static_cast<uint32_t>(sampleType));
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

void FrontEnd::parseResample(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    ie::details::CaselessEq<std::string> cmp;

    auto stage = model->addNewStage<ResampleStage>(
        layer->name,
        StageType::Resample,
        layer,
        inputs,
        outputs);

    stage->attrs().set<bool>("antialias", layer->GetParamAsInt("antialias", 0));
    stage->attrs().set<float>("factor", layer->GetParamAsInt("factor", -1.0f));

    auto method = layer->GetParamAsString("type", "caffe.ResampleParameter.NEAREST");
    if (cmp(method, "caffe.ResampleParameter.NEAREST")) {
        stage->attrs().set<ResampleType>("type", ResampleType::Nearest);
    } else {
        stage->attrs().set<ResampleType>("type", ResampleType::Bilinear);
    }
}

}  // namespace vpu
