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

    const auto& outDims = layer->output(0).get_shape();
    const auto outDesc = TensorDesc(Precision::FP32, outDims, TensorDesc::getLayoutByDims(outDims));
    const auto output = make_blob_with_precision(outDesc);
    output->allocate();

    const auto inputPtr = input->cbuffer().as<const float*>();
    auto outputPtr = output->buffer().as<float*>();

    IE_ASSERT(inputPtr != nullptr);
    IE_ASSERT(outputPtr != nullptr);
    std::memset(outputPtr, 0, output->byteSize());

    ngraph::runtime::reference::softmax(inputPtr, outputPtr,
            layer->input(0).get_shape(), ngraph::AxisSet({softmaxLayer->get_axis()}));

    return {output};
};

} // namespace

TestNetwork& SoftmaxLayerDef::build() {
    const auto node =
        std::make_shared<ngraph::op::v1::Softmax>(testNet.getPort(inputPort), axisSet);

    return testNet.addLayer(name, node, refSoftmax);
}
