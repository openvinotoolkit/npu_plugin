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

#include "kmb_test_region_yolo_def.hpp"

#include <precision_utils.h>

#include <blob_transform.hpp>

namespace {

inline float logisticActivate(float x) {
    float res = 1.0f / (1.0f + expf(-x));
    return res;
}

void computeRegionYolo(const Blob::Ptr& src, Blob::Ptr dst, int coords, int classes, int regions, int maskSize,
    int doSoftmax) {
    IE_ASSERT(src != nullptr);
    IE_ASSERT(dst != nullptr);

    const auto srcHWC = vpux::toLayout(as<MemoryBlob>(src), Layout::NHWC);
    auto dstHWC = vpux::toLayout(as<MemoryBlob>(dst), Layout::NHWC);

    const auto srcData = srcHWC->buffer().as<const float*>();
    const auto dstData = dstHWC->buffer().as<float*>();
    IE_ASSERT(srcData != nullptr);
    IE_ASSERT(dstData != nullptr);

    const auto& dims = src->getTensorDesc().getDims();
    IE_ASSERT(dims[0] == 1);
    const int C = dims[1];
    const int H = dims[2];
    const int W = dims[3];

    if (!doSoftmax) {
        regions = maskSize;
    }

    IE_ASSERT(C == (coords + classes + 1) * regions);
    IE_ASSERT(coords == 4);

    blob_copy(srcHWC, dstHWC);

    for (int w = 0; w < W; w++) {
        for (int h = 0; h < H; h++) {
            for (int region = 0; region < regions; region++) {
                auto dataPtr = dstData + region * (coords + classes + 1) + w*C + h*W*C;

                std::transform(dataPtr, dataPtr + 2, dataPtr, logisticActivate);
                std::transform(dataPtr + 4, dataPtr + 5, dataPtr + 4, logisticActivate);

                const auto begin = dataPtr + coords + 1;
                const auto end   = dataPtr + (coords + classes + 1);
                if (doSoftmax) {
                    const auto max = *std::max_element(begin, end);

                    std::transform(begin, end, begin, [&](float val) {
                        return expf(val - max);
                    });

                    const auto sum = std::accumulate(begin, end, 0.f);

                    std::transform(begin, end, begin, [&](float val) {
                        return val / sum;
                    });
                } else {
                    std::transform(begin, end, begin, logisticActivate);
                }
            }
        }
    }

    blob_copy(dstHWC, dst);
}

BlobVector refRegionYolo(const TestNetwork::NodePtr& layer, const BlobVector& inputs, const TestNetwork&) {
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(inputs.size() == 1);

    const auto RegionYoloLayer = std::dynamic_pointer_cast<ngraph::op::v0::RegionYolo>(layer);
    IE_ASSERT(RegionYoloLayer != nullptr);

    const auto classes = RegionYoloLayer->get_num_classes();
    const auto coords = RegionYoloLayer->get_num_coords();
    const auto regions = RegionYoloLayer->get_num_regions();
    const auto maskSize = RegionYoloLayer->get_mask().size();
    const auto doSoftmax = RegionYoloLayer->get_do_softmax();

    const auto input = inputs.at(0);
    auto output = vpux::makeSplatBlob(input->getTensorDesc(), 0.0f);

    computeRegionYolo(input, output, coords, classes, regions, maskSize, doSoftmax);

    return {output};
}

}  // namespace

TestNetwork& RegionYoloLayerDef::build() {
    std::shared_ptr<ngraph::Node> RegionYoloNode = std::make_shared<ngraph::op::v0::RegionYolo>(
        testNet.getPort(inputPort), params.coords, params.classes, params.regions, params.doSoftmax, params.mask, 0, 0);

    return testNet.addLayer(name, RegionYoloNode, refRegionYolo);
}
