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

#include "kmb_test_sigmoid_def.hpp"

#include <blob_factory.hpp>

#include <ngraph/runtime/reference/sigmoid.hpp>

namespace {

BlobVector refSigmoid(const TestNetwork::NodePtr& layer, const BlobVector& inputs, const TestNetwork&) {
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(inputs.size() == 1);

    const auto sigmoidLayer = std::dynamic_pointer_cast<ngraph::op::v0::Sigmoid>(layer);
    IE_ASSERT(sigmoidLayer != nullptr);

    const auto input = vpux::toDefLayout(vpux::toFP32(as<MemoryBlob>(inputs.at(0))));

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
