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
