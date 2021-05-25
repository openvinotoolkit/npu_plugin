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

#include "kmb_test_tile_def.hpp"
#include "kmb_test_add_def.hpp"

#include <blob_factory.hpp>

#include <ngraph/runtime/reference/tile.hpp>

using namespace ngraph;

namespace {

BlobVector refTile(const TestNetwork::NodePtr& layer, const BlobVector& inputs, const TestNetwork&) {
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(inputs.size() == 2);

    const auto tileLayer = std::dynamic_pointer_cast<op::v0::Tile>(layer);
    IE_ASSERT(tileLayer != nullptr);

    // Input
    const auto input = inputs.at(0);
    const auto inputPtr = input->cbuffer().as<const char*>();
    auto inputShape = input->getTensorDesc().getDims();
    IE_ASSERT(inputPtr != nullptr);

    // OutputShape
    std::vector<int64_t> repeats;
    repeats.assign(inputs.at(1)->cbuffer().as<const int64_t*>(),
                   inputs.at(1)->cbuffer().as<const int64_t*>() + inputShape.size());

    std::vector<uint64_t>outShape(inputShape.size());

    std::transform(inputShape.begin(), inputShape.end(), repeats.begin(), outShape.begin(), 
                   std::multiplies<uint64_t>());

    // Output
    const auto outDesc = TensorDesc(Precision::FP32, outShape, TensorDesc::getLayoutByDims(outShape));
    const auto output = make_blob_with_precision(outDesc);
    output->allocate();
    auto outputPtr = output->buffer().as<char*>();
    IE_ASSERT(outputPtr != nullptr);

    runtime::reference::tile(inputPtr, outputPtr, inputShape, outShape, sizeof(float), repeats);

    return {output};
};

}  // namespace

TestNetwork& TileLayerDef::build() {

    const auto tileNode =
        std::make_shared<op::v0::Tile>(
            testNet.getPort(inputPort), 
            testNet.getPort(repeatsPort));

    return testNet.addLayer(name, tileNode, refTile);
}
