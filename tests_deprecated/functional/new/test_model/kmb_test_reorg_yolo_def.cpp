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

#include "kmb_test_reorg_yolo_def.hpp"

#include <precision_utils.h>

#include <blob_factory.hpp>
#include <blob_transform.hpp>

namespace {

void refReorgYoloFromVPU(const Blob::Ptr& src, Blob::Ptr& dst, int stride) {
    IE_ASSERT(src != nullptr);
    IE_ASSERT(dst != nullptr);

    const auto srcData = src->buffer().as<const float*>();
    const auto dstData = dst->buffer().as<float*>();
    IE_ASSERT(srcData != nullptr);
    IE_ASSERT(dstData != nullptr);

    const auto& dims = src->getTensorDesc().getDims();
    IE_ASSERT(dims[0] == 1);
    const int C = dims[1];
    const int H = dims[2];
    const int W = dims[3];

    const int C2 = C / (stride * stride);
    const int H2 = H * stride;
    const int W2 = W * stride;

    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                const int offset = c / C2;
                const int c2 = c - C2 * offset;
                const int h2 = h * stride + offset / stride;
                const int w2 = w * stride + offset - stride * (offset / stride);

                dstData[c * H * W + h * W + w] = srcData[c2 * H2 * W2 + h2 * W2 + w2];
            }
        }
    }
}

BlobVector refReorgYolo(const TestNetwork::NodePtr& layer, const BlobVector& inputs, const TestNetwork&) {
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(inputs.size() == 1);

    const auto reorgYoloLayer = std::dynamic_pointer_cast<ngraph::op::v0::ReorgYolo>(layer);
    IE_ASSERT(reorgYoloLayer != nullptr);

    const auto stride = reorgYoloLayer->get_strides()[0];  // ngraph returns Stride structure, we use only first value

    const auto input = inputs.at(0);
    auto output = makeSingleValueBlob(input->getTensorDesc(), 0.0f);

    refReorgYoloFromVPU(input, output, stride);

    return {output};
}

}  // namespace

TestNetwork& ReorgYoloLayerDef::build() {
    std::shared_ptr<ngraph::Node> reorgYoloNode =
        std::make_shared<ngraph::op::v0::ReorgYolo>(
            testNet.getPort(inputPort), ngraph::Strides{static_cast<size_t>(params.stride)});

    return testNet.addLayer(name, reorgYoloNode, refReorgYolo);
}
