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

#include <memory>
#include <string>
#include <vector>
#include <set>
#include <unordered_set>
#include <algorithm>

#include <vpu/utils/extra.hpp>

namespace vpu {

namespace {

class ExpandStage final : public StageNode {
protected:
    StagePtr cloneImpl() const override {
        return std::make_shared<ExpandStage>(*this);
    }

    void propagateScaleFactorsImpl(
            const SmallVector<float>&,
            ScalePropagationStep) override {
        VPU_THROW_EXCEPTION << "Must never be called";
    }

    void propagateDataOrderImpl() const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();

        _orderInfo.setOutput(_outputEdges[0], input->desc().dimsOrder());
    }

    void getDataStridesRequirementsImpl() const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        auto dimsOrder = output->desc().dimsOrder();

        //
        // Get smallest Dim over which Expand is done.
        //

        auto minExpandDimInd = dimsOrder.numDims();

        for (const auto& p : output->desc().dims()) {
            if (input->desc().dim(p.first) != p.second) {
                minExpandDimInd = std::min(minExpandDimInd, dimsOrder.dimInd(p.first));
            }
        }

        IE_ASSERT(minExpandDimInd < dimsOrder.numDims());

        //
        // Initial StridesRequirement for input and output.
        //

        auto outputReqs = output->requiredStrides();

        auto inputReqs = outputReqs;
        for (int i = minExpandDimInd + 1; i < dimsOrder.numDims(); ++i) {
            inputReqs.remove(i);
        }

        //
        // Merge output consumers StridesRequirement.
        //

        for (const auto& consumerEdge : output->consumerEdges()) {
            const auto& consumerInfo = consumerEdge->consumer()->getDataStridesRequirements();

            if (consumerInfo.hasInput(consumerEdge)) {
                const auto& consumerReqs = consumerInfo.getInput(consumerEdge);

                for (int i = 0; i < minExpandDimInd + 1; ++i) {
                    if (outputReqs.get(i) == DimStride::Any) {
                        if (consumerReqs.get(i) != DimStride::Any) {
                            inputReqs.add(i, consumerReqs.get(i));
                            outputReqs.add(i, consumerReqs.get(i));
                        }
                    }
                }
            }
        }

        //
        // Return merged StridesRequirements.
        //

        _stridesInfo.setInput(_inputEdges[0], inputReqs);
        _stridesInfo.setOutput(_outputEdges[0], outputReqs);
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl() const override {
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer&) const override {
        VPU_THROW_EXCEPTION << "Must never be called";
    }

    void serializeDataImpl(BlobSerializer&) const override {
        VPU_THROW_EXCEPTION << "Must never be called";
    }
};

}  // namespace

Stage StageBuilder::addExpandStage(
        const Model::Ptr& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const Data& input,
        const Data& output,
        const DimValues& offset) {
    auto stage = model->addNewStage<ExpandStage>(
        name,
        StageType::Expand,
        layer,
        {input},
        {output});

    stage->attrs().set<DimValues>("offset", offset);

    return stage;
}

}  // namespace vpu
