//
// Copyright 2019 Intel Corporation.
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

#include "kmb_test_add_def.hpp"

#include <blob_factory.hpp>

#include <ngraph/runtime/reference/add.hpp>

namespace {

BlobVector refAdd(const TestNetwork::NodePtr& layer, const BlobVector& inputs, const TestNetwork&) {
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(inputs.size() == 2);

    const auto addLayer = std::dynamic_pointer_cast<ngraph::op::v1::Add>(layer);
    IE_ASSERT(addLayer != nullptr);

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

    ngraph::runtime::reference::add(
        input1Ptr, input2Ptr, outputPtr,
        layer->input(0).get_shape(), layer->input(1).get_shape(),
        addLayer->get_autob());

    return {output};
};

}  // namespace

TestNetwork& AddLayerDef::build() {
    const auto node =
        std::make_shared<ngraph::op::v1::Add>(
            testNet.getPort(input1Port), testNet.getPort(input2Port),
            broadcastSpec);

    return testNet.addLayer(name, node, refAdd);
}
