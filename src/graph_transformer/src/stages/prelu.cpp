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
#include <memory>

#include <vpu/sw/post_op_stage.hpp>

namespace vpu {

namespace {

class PReluStage final : public PostOpStage {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<PReluStage>(*this);
    }

    void serializeParamsImpl(BlobSerializer&) const override {
    }
};

}  // namespace

void FrontEnd::parsePReLU(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto weightsIt = layer->blobs.find("weights");
    if (weightsIt == layer->blobs.end()) {
        THROW_IE_EXCEPTION << "[VPU] PReLU doesn't have weights";
    }

    auto weightsBlob = weightsIt->second;
    IE_ASSERT(weightsBlob != nullptr);

    auto channelShared = layer->GetParamAsInt("channel_shared", 0);

    auto output = outputs[0];

    auto weights = model->addConstData(
        layer->name + "@weights",
        DataDesc({output->desc().dim(Dim::C)}),
        ieBlobContent(
            weightsBlob,
            channelShared ? output->desc().dim(Dim::C) : 1));

    model->addNewStage<PReluStage>(
        layer->name,
        StageType::PRelu,
        layer,
        {inputs[0], weights},
        outputs);
}

}  // namespace vpu
