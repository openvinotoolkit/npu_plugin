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

#include "kmb_test_mvn_def.hpp"

#include <blob_transform.hpp>

namespace {

static void refMVNFromVPU(const Blob::Ptr src, Blob::Ptr dst, int normalize_variance,
                          int across_channels, const float eps) {
    IE_ASSERT(src != nullptr);
    IE_ASSERT(dst != nullptr);

    const auto srcData = src->buffer().as<const float*>();
    const auto dstData = dst->buffer().as<float*>();
    IE_ASSERT(srcData != nullptr);
    IE_ASSERT(dstData != nullptr);

    const auto &dims = src->getTensorDesc().getDims();
    const auto &size = dims.size();
    IE_ASSERT(size == 3 || size == 4);
    IE_ASSERT(size == 3 || dims[0] == 1);
    const int IC = dims[dims.size() - 3];
    const int IH = dims[dims.size() - 2];
    const int IW = dims[dims.size() - 1];

    // Calculate mean value
    if (across_channels) {
        float mean = 0;
        for (int i = 0; i < IH*IW*IC; i++) {
            mean += srcData[i];
        }
        mean /= static_cast<float>(IC * IH * IW);
        for (int i = 0; i < IH*IW*IC; i++) {
            dstData[i] = srcData[i] - mean;
        }
    } else {
        for (int c = 0; c < IC; c++) {
            float mean = 0;
            for (int h = 0; h < IH; h++) {
                for (int w = 0; w < IW; w++) {
                    int ind = c * IH * IW + h * IW + w;
                    mean += srcData[ind];
                }
            }
            mean /= static_cast<float>(IH * IW);
            for (int h = 0; h < IH; h++) {
                for (int w = 0; w < IW; w++) {
                    int ind = c * IH * IW + h * IW + w;
                    dstData[ind] = srcData[ind] - mean;
                }
            }
        }
    }

    // Calculate variances value
    if (normalize_variance) {
        if (across_channels) {
            float variance = 0;
            for (int i = 0; i < IH*IW*IC; i++) {
                variance += dstData[i] * dstData[i];
            }
            variance /= static_cast<float>(IC * IH * IW);
            variance = sqrtf(variance);
            variance += eps;
            for (int i = 0; i < IH*IW*IC; i++) {
               dstData[i] /= variance;
            }
        } else {
            for (int c = 0; c < IC; c++) {
                float variance = 0;
                for (int h = 0; h < IH; h++) {
                    for (int w = 0; w < IW; w++) {
                        int ind = c * IH * IW + h * IW + w;
                        variance += dstData[ind] * dstData[ind];
                    }
                }
                variance /= static_cast<float>(IH * IW);
                variance = sqrtf(variance);
                variance += eps;
                for (int h = 0; h < IH; h++) {
                    for (int w = 0; w < IW; w++) {
                        int ind = c * IH * IW + h * IW + w;
                        dstData[ind] /= variance;
                    }
                }
            }
        }
    }
}

BlobVector refMVN(const TestNetwork::NodePtr& layer, const BlobVector& inputs, const TestNetwork&) {
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(inputs.size() == 1);

    const auto mvnLayer = std::dynamic_pointer_cast<ngraph::op::v0::MVN>(layer);
    IE_ASSERT(mvnLayer != nullptr);

    const auto normalize_variance = mvnLayer->get_normalize_variance();
    const auto across_channels = mvnLayer->get_across_channels();
    const auto eps = mvnLayer->get_eps();

    auto input = inputs.at(0);
    auto output = vpux::makeSplatBlob(input->getTensorDesc(), 0.0f);

    refMVNFromVPU(input, output, normalize_variance, across_channels, eps);

    return {output};
}

}  // namespace

TestNetwork& MVNLayerDef::build() {
    std::shared_ptr<ngraph::Node> mvnNode =
        std::make_shared<ngraph::op::v0::MVN>(
            testNet.getPort(inputPort), params.across_channels, params.normalize_variance, params.eps);

    return testNet.addLayer(name, mvnNode, refMVN);
}
