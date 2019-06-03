//
// Copyright 2016-2018 Intel Corporation.
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
#include <set>
#include <string>

#include <vpu/sw/post_op_stage.hpp>

namespace vpu {

namespace {

class ClampStage final : public PostOpStage {
protected:
    StagePtr cloneImpl() const override {
        return std::make_shared<ClampStage>(*this);
    }

    void propagateScaleFactorsImpl(
            const SmallVector<float>& inputScales,
            ScalePropagationStep step) override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        if (step == ScalePropagationStep::Propagate) {
            auto inputScale = inputScales[0];

            _scaleInfo.setOutput(_outputEdges[0], inputScale);

            attrs().get<float>("min_value") *= inputScale;
            attrs().get<float>("max_value") *= inputScale;
        } else {
            // Clamp can only propagate scaling, not generate.
            _scaleInfo.setInput(_inputEdges[0], 1.0f);
            _scaleInfo.setOutput(_outputEdges[0], 1.0f);
        }
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto min_value = attrs().get<float>("min_value");
        auto max_value = attrs().get<float>("max_value");

        serializer.append(static_cast<float>(min_value));
        serializer.append(static_cast<float>(max_value));
    }
};

}  // namespace

void FrontEnd::parseClamp(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& _layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<ie::ClampLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    _stageBuilder->addClampStage(model, layer->name, layer, layer->min_value,  layer->max_value, inputs[0], outputs[0]);
}

Stage StageBuilder::addClampStage(
            const Model::Ptr& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            float min,
            float max,
            const Data& input,
            const Data& output) {
        auto stage = model->addNewStage<ClampStage>(
                name,
                StageType::Clamp,
                layer,
                {input},
                {output});

        stage->attrs().set<float>("min_value", min);
        stage->attrs().set<float>("max_value", max);

        return stage;
    }


}  // namespace vpu
