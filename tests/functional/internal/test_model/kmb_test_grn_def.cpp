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

#include "kmb_test_grn_def.hpp"

#include <blob_transform.hpp>

namespace {

void refGRNFromVPU(const Blob::Ptr src, Blob::Ptr dst, float bias) {
    IE_ASSERT(src != nullptr);
    IE_ASSERT(dst != nullptr);

    const auto srcHWC = vpux::toLayout(as<MemoryBlob>(src), Layout::NHWC);
    auto dstHWC = vpux::toLayout(as<MemoryBlob>(dst), Layout::NHWC);

    const auto srcData = srcHWC->buffer().as<const float*>();
    const auto dstData = dstHWC->buffer().as<float*>();
    IE_ASSERT(srcData != nullptr);
    IE_ASSERT(dstData != nullptr);

    const auto& dims = srcHWC->getTensorDesc().getDims();
    IE_ASSERT(dims[0] == 1);
    const int C = dims[1];
    const int H = dims[2];
    const int W = dims[3];

    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            float variance = 1e-9f;
            for (int c = 0; c < C; c++) {
                const auto value = srcData[h * W * C + w * C + c]; // h * W * C + w * C + c
                variance += value * value;
            }
            variance = sqrtf(variance + bias);
            for (int c = 0; c < C; c++) {
                const auto value = srcData[h * W * C + w * C + c];
                dstData[h * W * C + w * C + c] = value / variance;
            }
        }
    }

    blob_copy(dstHWC, dst);
}

BlobVector refGRN(const TestNetwork::NodePtr& layer, const BlobVector& inputs, const TestNetwork&) {
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(inputs.size() == 1);

    const auto grnLayer = std::dynamic_pointer_cast<ngraph::op::v0::GRN>(layer);
    IE_ASSERT(grnLayer != nullptr);

    const auto bias = grnLayer->get_bias();

    const auto input = inputs.at(0);
    auto output = vpux::makeSplatBlob(input->getTensorDesc(), 0.0f);

    refGRNFromVPU(input, output, bias);

    return {output};
}

}  // namespace

TestNetwork& GRNLayerDef::build() {
    std::shared_ptr<ngraph::Node> grnNode =
        std::make_shared<ngraph::op::v0::GRN>(
            testNet.getPort(inputPort), params.bias);

    return testNet.addLayer(name, grnNode, refGRN);
}
