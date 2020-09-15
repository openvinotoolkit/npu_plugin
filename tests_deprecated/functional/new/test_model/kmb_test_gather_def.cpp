//
// Copyright 2020 Intel Corporation.
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

#include "kmb_test_gather_def.hpp"

#include <ngraph/runtime/reference/gather.hpp>
#include <blob_factory.hpp>

namespace {

BlobVector refGather(const TestNetwork::NodePtr& layer, const BlobVector& inputs, const TestNetwork&) {
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(inputs.size() == 3);

    const auto gatherLayer = std::dynamic_pointer_cast<ngraph::op::v1::Gather>(layer);
    IE_ASSERT(gatherLayer != nullptr);

    const auto params = toDefLayout(toFP32(inputs.at(0)));
    const auto indices = inputs.at(1);

    const auto& outDims = layer->output(0).get_shape();
    const auto outDesc = TensorDesc(Precision::FP32, outDims,
                                    TensorDesc::getLayoutByDims(outDims));
    const auto output = make_blob_with_precision(outDesc);
    output->allocate();

    const auto paramsPtr = params->cbuffer().as<const float*>();
    const auto indicesPtr = indices->cbuffer().as<const int*>();
    auto outputPtr = output->buffer().as<float*>();

    IE_ASSERT(paramsPtr != nullptr);
    IE_ASSERT(indicesPtr != nullptr);
    IE_ASSERT(outputPtr != nullptr);

    std::cout << "Out shape:" << layer->output(0).get_shape() <<std::endl;

    ngraph::runtime::reference::gather(paramsPtr,
                                       indicesPtr,
                                       outputPtr,
                                       layer->input(0).get_shape(),
                                       layer->input(1).get_shape(),
                                       layer->output(0).get_shape(),
                                       size_t(gatherLayer->get_axis()));
    std::cout << "output byte size: " << output->byteSize() <<std::endl;
    return {output};
};

}  // namespace

TestNetwork& GatherLayerDef::build() {
    const auto node = std::make_shared<ngraph::op::v1::Gather>(
        testNet.getPort(inputPort), testNet.getPort(indicesPort), testNet.getPort(axisPort));

    return testNet.addLayer(name, node, refGather);
}

