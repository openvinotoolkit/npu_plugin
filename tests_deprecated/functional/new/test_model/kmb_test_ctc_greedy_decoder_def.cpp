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

#include "kmb_test_ctc_greedy_decoder_def.hpp"

#include <blob_transform.hpp>

namespace {

void refCTCGreedyDecoderFromVPU(const Blob::Ptr& src, const Blob::Ptr& seqInd, Blob::Ptr& dst) {
    IE_ASSERT(src != nullptr);
    IE_ASSERT(seqInd != nullptr);
    IE_ASSERT(dst != nullptr);

    const auto srcData = src->buffer().as<const float*>();
    const auto seqIndData = seqInd->buffer().as<const float*>();
    const auto dstData = dst->buffer().as<float*>();
    IE_ASSERT(srcData != nullptr);
    IE_ASSERT(seqIndData != nullptr);
    IE_ASSERT(dstData != nullptr);

    const auto& dims = src->getTensorDesc().getDims();
    IE_ASSERT(dims[0] == 1);
    const int T = dims[1];  // Time
    const int B = dims[2];  // Batches
    const int C = dims[3];  // Chars

    std::fill_n(dstData, B*T, -1.0f);

    for (int b = 0; b < B; ++b) {
        const auto curSeqInd = seqIndData + b*T;
        // first value in sequence indicators is ignored by historical reasons
        const auto seqLast = std::find(curSeqInd + 1, curSeqInd + T, 0.0f);
        const int time = std::distance(curSeqInd, seqLast);

        int prevClassIdx = -1;
        int outIdx = 0;

        for (int t = 0; t < time; ++t) {
            const float* probs = srcData + b * C + t * C * B;
            const auto maximum = std::max_element(probs, probs + C);
            const auto maxClassIdx = std::distance(probs, maximum);

            if ((maxClassIdx < C - 1) && (maxClassIdx != prevClassIdx)) {
                dstData[b*T + outIdx] = (float)maxClassIdx;
                outIdx++;
            }

            prevClassIdx = maxClassIdx;
        }
    }
}

BlobVector refCTCGreedyDecoder(const TestNetwork::NodePtr& layer, const BlobVector& inputs, const TestNetwork&) {
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(inputs.size() == 2);

    const auto ctcLayer = std::dynamic_pointer_cast<ngraph::op::v0::CTCGreedyDecoder>(layer);
    IE_ASSERT(ctcLayer != nullptr);

    const auto mergeRepeated = ctcLayer->get_ctc_merge_repeated();
    IE_ASSERT(mergeRepeated);

    const auto input0 = inputs.at(0);
    const auto input1 = inputs.at(1);
    auto output = makeSingleValueBlob(input0->getTensorDesc(), 0.0f);

    refCTCGreedyDecoderFromVPU(input0, input1, output);

    return {output};
}

}  // namespace

TestNetwork& CTCGreedyDecoderLayerDef::build() {
    std::shared_ptr<ngraph::Node> ctcNode =
        std::make_shared<ngraph::op::v0::CTCGreedyDecoder>(
            testNet.getPort(inputPorts[0]), testNet.getPort(inputPorts[1]), params.mergeRepeated);

    return testNet.addLayer(name, ctcNode, refCTCGreedyDecoder);
}
