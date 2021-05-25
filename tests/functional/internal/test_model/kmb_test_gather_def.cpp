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

#include "kmb_test_gather_def.hpp"

#include <ngraph/runtime/reference/gather.hpp>
#include <blob_factory.hpp>

namespace {

BlobVector refGather(const TestNetwork::NodePtr& layer, const BlobVector& inputs, const TestNetwork&) {
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(inputs.size() == 3);

    const auto gatherLayer = std::dynamic_pointer_cast<ngraph::op::v1::Gather>(layer);
    IE_ASSERT(gatherLayer != nullptr);

    const auto params = vpux::toDefLayout(vpux::toFP32(as<MemoryBlob>(inputs.at(0))));
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

    ngraph::runtime::reference::gather(paramsPtr,
                                       indicesPtr,
                                       outputPtr,
                                       layer->input(0).get_shape(),
                                       layer->input(1).get_shape(),
                                       layer->output(0).get_shape(),
                                       size_t(gatherLayer->get_axis()));
    return {output};
};

}  // namespace

TestNetwork& GatherLayerDef::build() {
    const auto node = std::make_shared<ngraph::op::v1::Gather>(
        testNet.getPort(inputPort), testNet.getPort(indicesPort), testNet.getPort(axisPort));

    return testNet.addLayer(name, node, refGather);
}

