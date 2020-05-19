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

#include "kmb_test_sigmoid_def.hpp"

#include <blob_factory.hpp>

#include <ngraph/runtime/reference/sigmoid.hpp>

namespace {

BlobVector refSigmoid(const TestNetwork::NodePtr& layer, const BlobVector& inputs, const TestNetwork&) {
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(inputs.size() == 1);

    const auto sigmoidLayer = std::dynamic_pointer_cast<ngraph::op::v0::Sigmoid>(layer);
    IE_ASSERT(sigmoidLayer != nullptr);

    const auto input = toDefLayout(toFP32(inputs.at(0)));

    const auto& outDims = layer->output(0).get_shape();
    const auto outDesc = TensorDesc(Precision::FP32, outDims, TensorDesc::getLayoutByDims(outDims));
    const auto output = make_blob_with_precision(outDesc);
    output->allocate();

    const auto inputPtr = input->cbuffer().as<const float*>();
    auto outputPtr = output->buffer().as<float*>();

    IE_ASSERT(inputPtr != nullptr);
    IE_ASSERT(outputPtr != nullptr);

    ngraph::runtime::reference::sigmoid(inputPtr, outputPtr, layer->get_input_size());

    return {output};
};

} // namespace

TestNetwork& SigmoidLayerDef::build() {
    const auto node =
        std::make_shared<ngraph::op::v0::Sigmoid>(testNet.getPort(inputPort));

    return testNet.addLayer(name, node, refSigmoid);
}
