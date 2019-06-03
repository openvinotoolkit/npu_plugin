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
#include <string>

#include <vpu/sw/post_op_stage.hpp>

namespace vpu {

void FrontEnd::parseBias(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 2);
    IE_ASSERT(outputs.size() == 1);

    auto input = inputs[0];
    auto biases = inputs[1];

    auto biasesDims = biases->desc().dims();
    if (biasesDims.size() < 4 && input->desc().numDims() == 4) {
        biasesDims.set(Dim::N, 1);
    }

    if (input->desc().dims() != biasesDims) {
        VPU_THROW_EXCEPTION
            << "Current Bias layer implementation supports only equal inputs (axis 0, 1 for 4D tensor, axis 0 for other dimensions),"
            << " layer name is " << layer->name;
    }

    if (biases->desc().numDims() < 4 && input->desc().numDims() == 4) {
        DataDesc newBiasesDesc({
            biases->desc().dim(Dim::W),
            biases->desc().dim(Dim::H),
            biases->desc().dim(Dim::C),
            1});

        auto newBiases = model->duplicateData(
            biases,
            "@reshaped",
            newBiasesDesc);

        _stageBuilder->addReshapeStage(
            model,
            newBiases->name(),
            layer,
            biases,
            newBiases);

        biases = newBiases;
    }

    _stageBuilder->addSumStage(
        model,
        layer->name,
        layer,
        input, biases,
        outputs[0]);
}

namespace {

class BiasStage final : public PostOpStage {
protected:
    StagePtr cloneImpl() const override {
        return std::make_shared<BiasStage>(*this);
    }

    void propagateScaleFactorsImpl(
            const SmallVector<float>& inputScales,
            ScalePropagationStep step) override {
        IE_ASSERT(_inputEdges.size() == 2);
        IE_ASSERT(_outputEdges.size() == 1);

        if (step == ScalePropagationStep::Propagate) {
            auto inputScale = inputScales[0];

            _scaleInfo.setInput(_inputEdges[1], inputScale);
            _scaleInfo.setOutput(_outputEdges[0], inputScale);
        } else {
            // Bias can only propagate scaling, not generate.
            _scaleInfo.setInput(_inputEdges[0], 1.0f);
            _scaleInfo.setInput(_inputEdges[1], 1.0f);
            _scaleInfo.setOutput(_outputEdges[0], 1.0f);
        }
    }

    void serializeParamsImpl(BlobSerializer&) const override {
    }
};

}  // namespace

Stage StageBuilder::addBiasStage(
        const Model::Ptr& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const Data& input,
        const Data& biases,
        const Data& output) {
    return model->addNewStage<BiasStage>(
        name,
        StageType::Bias,
        layer,
        {input, biases},
        {output});
}

}  // namespace vpu
