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

#include "kmb_test_fake_quantize_def.hpp"

#include <blob_factory.hpp>

namespace {

BlobVector refFQ(const TestNetwork::NodePtr& layer, const BlobVector& inputs, const TestNetwork&) {
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(inputs.size() == 5);

    const auto fqLayer = std::dynamic_pointer_cast<ngraph::op::FakeQuantize>(layer);
    IE_ASSERT(fqLayer != nullptr);

    const auto input = inputs.at(0);
    const auto inputLowBlob = inputs.at(1);
    const auto inputHighBlob = inputs.at(2);
    const auto outputLowBlob = inputs.at(3);
    const auto outputHighBlob = inputs.at(4);

    IE_ASSERT(inputLowBlob->size() == 1);
    IE_ASSERT(inputHighBlob->size() == 1);
    IE_ASSERT(outputLowBlob->size() == 1);
    IE_ASSERT(outputHighBlob->size() == 1);

    const auto inputLow = inputLowBlob->cbuffer().as<const float*>()[0];
    const auto inputHigh = inputHighBlob->cbuffer().as<const float*>()[0];
    const auto outputLow = outputLowBlob->cbuffer().as<const float*>()[0];
    const auto outputHigh = outputHighBlob->cbuffer().as<const float*>()[0];
    const auto levels = fqLayer->get_levels();

    const auto& outDims = layer->output(0).get_shape();
    const auto outDesc = TensorDesc(Precision::FP32, outDims, TensorDesc::getLayoutByDims(outDims));
    const auto output = make_blob_with_precision(outDesc);
    output->allocate();

    IE_ASSERT(input->size() == output->size());

    const auto inputPtr = input->cbuffer().as<const float*>();
    auto outputPtr = output->buffer().as<float*>();

    for (size_t i = 0; i < output->size(); ++i) {
        const auto inputVal = inputPtr[i];

        float outputVal = 0.0f;
        if (inputVal <= inputLow) {
            outputVal = outputLow;
        } else if (inputVal > inputHigh) {
            outputVal = outputHigh;
        } else {
            const auto inputScale = (inputHigh - inputLow) / (levels - 1);
            const auto outputScale = (outputHigh - outputLow) / (levels - 1);
            const auto valQuant = std::round((inputVal - inputLow) / inputScale);
            outputVal = valQuant * outputScale + outputLow;
        }

        outputPtr[i] = outputVal;
    }

    return {output};
}

}  // namespace

TestNetwork& FakeQuantizeLayerDef::build() {
    const auto node =
        std::make_shared<ngraph::op::FakeQuantize>(
            testNet.getPort(inputPort),
            testNet.getPort(inputLowPort), testNet.getPort(inputHighPort),
            testNet.getPort(outputLowPort), testNet.getPort(outputHighPort),
            levels);

    return testNet.addLayer(name, node, refFQ);
}
