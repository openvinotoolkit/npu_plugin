//
// Copyright 2019 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "kmb_test_mul_def.hpp"

#include <blob_factory.hpp>

#include <ngraph/runtime/reference/autobroadcast_binop.hpp>
#include <ngraph/runtime/reference/multiply.hpp>

namespace {

BlobVector refMultiply(const TestNetwork::NodePtr& layer, const BlobVector& inputs, const TestNetwork&) {
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(inputs.size() == 2);

    const auto mulLayer = std::dynamic_pointer_cast<ngraph::op::v1::Multiply>(layer);
    IE_ASSERT(mulLayer != nullptr);

    const auto input1 = inputs.at(0);
    const auto input2 = inputs.at(1);

    const auto& outDims = layer->output(0).get_shape();
    const auto outDesc = TensorDesc(Precision::FP32, outDims, TensorDesc::getLayoutByDims(outDims));
    const auto output = make_blob_with_precision(outDesc);
    output->allocate();

    const auto input1Ptr = input1->cbuffer().as<const float*>();
    const auto input2Ptr = input2->cbuffer().as<const float*>();
    auto outputPtr = output->buffer().as<float*>();

    IE_ASSERT(input1Ptr != nullptr);
    IE_ASSERT(input2Ptr != nullptr);
    IE_ASSERT(outputPtr != nullptr);

    ngraph::runtime::reference::multiply(
        input1Ptr, input2Ptr, outputPtr,
        layer->input(0).get_shape(), layer->input(1).get_shape(),
        mulLayer->get_autob());

    return {output};
};

}  // namespace

TestNetwork& MultiplyLayerDef::build() {
    const auto node =
        std::make_shared<ngraph::op::v1::Multiply>(
            testNet.getPort(input1Port), testNet.getPort(input2Port),
            broadcastSpec);

    return testNet.addLayer(name, node, refMultiply);
}
