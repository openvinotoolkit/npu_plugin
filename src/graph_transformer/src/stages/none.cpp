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

#include <string>
#include <vector>
#include <list>
#include <memory>

namespace vpu {

namespace {

class NoneStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<NoneStage>(*this);
    }

    void propagateScaleFactorsImpl(
            const SmallVector<float>&,
            ScalePropagationStep) override {
        for (const auto& outEdge : _outputEdges) {
            _scaleInfo.setOutput(outEdge, 1.0f);
        }
    }

    void propagateDataOrderImpl() const override {
    }

    void getDataStridesRequirementsImpl() const override {
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl() const override {
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::NotNeeded;
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer&) const override {
    }

    void serializeDataImpl(BlobSerializer&) const override {
    }
};

}  // namespace

Stage StageBuilder::addNoneStage(
        const Model::Ptr& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    return model->addNewStage<NoneStage>(
        name,
        StageType::None,
        layer,
        {inputs},
        {outputs});
}

}  // namespace vpu
