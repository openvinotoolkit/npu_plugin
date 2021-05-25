//
// Copyright 2020 Intel Corporation.
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

#include "kmb_test_fully_connected_def.hpp"
#include "kmb_test_add_def.hpp"

#include <blob_factory.hpp>

#include <ngraph/runtime/reference/add.hpp>

namespace {

BlobVector refFC(const TestNetwork::NodePtr& layer, const BlobVector& inputs, const TestNetwork&) {
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(inputs.size() == 2);

    const auto input = inputs.at(0);
    const auto weights = inputs.at(1);

    const auto& outDims = layer->output(0).get_shape();
    const auto outDesc = TensorDesc(Precision::FP32, outDims, TensorDesc::getLayoutByDims(outDims));
    const auto output = make_blob_with_precision(outDesc);
    output->allocate();

    const auto inputPtr = input->cbuffer().as<const float*>();
    const auto weightsPtr = weights->cbuffer().as<const float*>();

    IE_ASSERT(inputPtr != nullptr);
    IE_ASSERT(weightsPtr != nullptr);

    const auto fcLayer = std::dynamic_pointer_cast<ngraph::op::MatMul>(layer);
    IE_ASSERT(fcLayer != nullptr);

    size_t IC = input->size();
    size_t OC = outDims[1];

    auto* outputPtr = output->buffer().as<float*>();

    for (size_t oc = 0; oc < OC; oc++) {
        outputPtr[oc] = 0.0;
        for (size_t ic = 0; ic < IC; ic++) {
            size_t iidx = ic;
            size_t widx = oc * IC + ic;

            outputPtr[oc] += inputPtr[iidx] * weightsPtr[widx];
        }
    }

    return {output};
}

}  // namespace

TestNetwork& FullyConnectedLayerDef::build() {
    const auto fcNode =
        std::make_shared<ngraph::op::MatMul>(
            testNet.getPort(inputPort), testNet.getPort(weightsPort), false, true);

    if (biasesPort.layerName.empty()) {
        return testNet.addLayer(name, fcNode, refFC);
    } else {
        testNet.addLayer(name + "_fc", fcNode, refFC);

        return
            testNet.addLayer<AddLayerDef>(name)
                .input1(name + "_fc")
                .input2(biasesPort.layerName, biasesPort.index)
                .build();
    }
}

TensorDesc getFCWeightsDesc(const FullyConnectedParams& params, size_t inChannels, Precision precision) {
    return {precision, {params._outChannels, inChannels}, Layout::NC};
}

TensorDesc getFCBiasesDesc(const FullyConnectedParams& params, Precision precision) {
    return {precision, {1, params._outChannels}, Layout::NC};
}
