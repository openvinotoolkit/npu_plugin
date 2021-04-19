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

#include "kmb_test_prelu_def.hpp"

#include <blob_transform.hpp>
#include <blob_factory.hpp>
namespace {

void refPReluFromVPU(const Blob::Ptr src, const Blob::Ptr& weight, Blob::Ptr dst) {
    
    IE_ASSERT(src != nullptr);
    IE_ASSERT(weight != nullptr);
    IE_ASSERT(dst != nullptr);

    const auto srcData    = src->buffer().as<const float*>();
    const auto weightData = weight->buffer().as<const float*>();
    const auto dstData    = dst->buffer().as<float*>();
    IE_ASSERT(srcData    != nullptr);
    IE_ASSERT(weightData != nullptr);
    IE_ASSERT(dstData    != nullptr);

    const auto& dims = src->getTensorDesc().getDims();
    IE_ASSERT(dims[0] == 1);
    const int C = dims[1];
    const int H = dims[2];
    const int W = dims[3];

    for (int c = 0; c < C; c++)
    {
        float slope = weightData[c];
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
            {
                float in = srcData[c * H * W + h * W + w];
                float maxVal = std::max(in, 0.0f);
                float minVal = std::min(in, 0.0f);
                dstData[c * H * W + h * W + w] = maxVal + slope * minVal;
            }
    }
}

BlobVector refPRelu(const TestNetwork::NodePtr& layer, const BlobVector& inputs, const TestNetwork&) {
    
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(inputs.size() == 2);

    const auto preluLayer = std::dynamic_pointer_cast<ngraph::op::v0::PRelu>(layer);
    IE_ASSERT(preluLayer != nullptr);

    const auto input = inputs.at(0);
    const auto weights = inputs.at(1);

    const auto& outDims = layer->output(0).get_shape();
    const auto outDesc = TensorDesc(Precision::FP32, outDims, TensorDesc::getLayoutByDims(outDims));
    const auto output = make_blob_with_precision(outDesc);
    output->allocate();

    refPReluFromVPU(input, weights, output);

    return {output};
}

}  // namespace

TestNetwork& PReluLayerDef::build() {
    std::shared_ptr<ngraph::Node> PReluNode =
        std::make_shared<ngraph::op::v0::PRelu>(
            testNet.getPort(inputPort), testNet.getPort(weightsPort));     

    return testNet.addLayer(name, PReluNode, refPRelu);
}
