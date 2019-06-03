//
// INTEL CONFIDENTIAL
// Copyright (C) 2018-2019 Intel Corporation.
//
// The source code contained or described herein and all documents
// related to the source code ("Material") are owned by Intel Corporation
// or its suppliers or licensors. Title to the Material remains with
// Intel Corporation or its suppliers and licensors. The Material may
// contain trade secrets and proprietary and confidential information
// of Intel Corporation and its suppliers and licensors, and is protected
// by worldwide copyright and trade secret laws and treaty provisions.
// No part of the Material may be used, copied, reproduced, modified,
// published, uploaded, posted, transmitted, distributed, or disclosed
// in any way without Intel's prior express written permission.
//
// No license under any patent, copyright, trade secret or other
// intellectual property right is granted to or conferred upon you by
// disclosure or delivery of the Materials, either expressly, by implication,
// inducement, estoppel or otherwise. Any license under such intellectual
// property rights must be express and approved by Intel in writing.
//
// Include any supplier copyright notices as supplier requires Intel to use.
//
// Include supplier trademarks or logos as supplier requires Intel to use,
// preceded by an asterisk. An asterisked footnote can be added as follows:
// *Third Party trademarks are the property of their respective owners.
//
// Unless otherwise agreed by Intel in writing, you may not remove or alter
// this notice or any other notice embedded in Materials by Intel or Intel's
// suppliers or licensors in any way.
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <unordered_set>
#include <memory>
#include <set>

namespace vpu {

namespace {

class PSROIPoolingStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<PSROIPoolingStage>(*this);
    }

    void propagateDataOrderImpl() const override {
        IE_ASSERT(_inputEdges.size() == 2);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input0 = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        _orderInfo.setInput(_inputEdges[0], input0->desc().dimsOrder().createMovedDim(Dim::C, 2));
        _orderInfo.setOutput(_outputEdges[0], output->desc().dimsOrder().createMovedDim(Dim::C, 2));
    }

    void getDataStridesRequirementsImpl() const override {
        IE_ASSERT(_inputEdges.size() == 2);
        IE_ASSERT(_outputEdges.size() == 1);

        _stridesInfo.setInput(_inputEdges[0], StridesRequirement::compact());
        _stridesInfo.setInput(_inputEdges[1], StridesRequirement::compact());
        _stridesInfo.setOutput(_outputEdges[0], StridesRequirement::compact());
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl() const override {
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto group_size = attrs().get<int>("group_size");
        auto output_dim = attrs().get<int>("output_dim");
        auto spatial_scale = attrs().get<float>("spatial_scale");

        serializer.append(static_cast<uint32_t>(group_size));
        serializer.append(static_cast<uint32_t>(output_dim));
        serializer.append(static_cast<float>(spatial_scale));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 2);
        IE_ASSERT(_outputEdges.size() == 1);
        IE_ASSERT(_tempBufferEdges.empty());

        auto input0 = _inputEdges[0]->input();
        auto input1 = _inputEdges[1]->input();
        auto output = _outputEdges[0]->output();

        input0->serializeNewBuffer(serializer);
        output->serializeNewBuffer(serializer);
        input1->serializeNewBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parsePSROIPooling(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 2);
    IE_ASSERT(outputs.size() == 1);

    auto stage = model->addNewStage<PSROIPoolingStage>(
        layer->name,
        StageType::PSROIPooling,
        layer,
        inputs,
        outputs);

    stage->attrs().set<int>("group_size", layer->GetParamAsInt("group_size", 7));
    stage->attrs().set<int>("output_dim", layer->GetParamAsInt("output_dim", 21));
    stage->attrs().set<float>("spatial_scale", layer->GetParamAsFloat("spatial_scale", 0.0625f));
}

}  // namespace vpu
