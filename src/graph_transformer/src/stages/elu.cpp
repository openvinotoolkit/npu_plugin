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
#include <set>

#include <vpu/sw/post_op_stage.hpp>

namespace vpu {

namespace {

class EluStage final : public PostOpStage {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<EluStage>(*this);
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto alpha = attrs().get<float>("alpha");

        serializer.append(static_cast<float>(alpha));
    }
};

}  // namespace

void FrontEnd::parseELU(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto alpha = layer->GetParamAsFloat("alpha", 1.0f);

    auto stage = model->addNewStage<EluStage>(
        layer->name,
        StageType::Elu,
        layer,
        inputs,
        outputs);

    stage->attrs().set<float>("alpha", alpha);
}

}  // namespace vpu
