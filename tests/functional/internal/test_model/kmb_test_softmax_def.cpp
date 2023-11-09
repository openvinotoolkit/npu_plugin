//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "kmb_test_softmax_def.hpp"

#include <blob_factory.hpp>
#include <ngraph/runtime/reference/softmax.hpp>

namespace {

BlobVector refSoftmax(const TestNetwork::NodePtr& layer, const BlobVector& inputs, const TestNetwork&) {
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(inputs.size() == 1);

    const auto softmaxLayer = std::dynamic_pointer_cast<ngraph::op::v1::Softmax>(layer);
    IE_ASSERT(softmaxLayer != nullptr);

    const auto input = inputs.at(0);
    const auto output = vpux::makeSplatBlob(input->getTensorDesc(), 0.0f);

    const auto inputPtr = input->cbuffer().as<const float*>();
    auto outputPtr = output->buffer().as<float*>();

    IE_ASSERT(inputPtr != nullptr);
    IE_ASSERT(outputPtr != nullptr);

    ngraph::runtime::reference::softmax(inputPtr, outputPtr, layer->input(0).get_shape(),
                                        ngraph::AxisSet({softmaxLayer->get_axis()}));

    return {output};
};

}  // namespace

TestNetwork& SoftmaxLayerDef::build() {
    const auto node = std::make_shared<ngraph::op::v1::Softmax>(testNet.getPort(inputPort), axisSet);

    return testNet.addLayer(name, node, refSoftmax);
}
