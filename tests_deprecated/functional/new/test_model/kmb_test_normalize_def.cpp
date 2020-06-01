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

#include "kmb_test_normalize_def.hpp"

#include <blob_factory.hpp>
#include <ngraph/op/fused/normalize_l2.hpp>
#include <precision_utils.h>
#include "vpu/utils/ie_helpers.hpp"

namespace {

static void refNormalizeFromVPU(const Blob::Ptr src,
                         Blob::Ptr dst,
                         int64_t* axes,
			 ngraph::op::EpsMode eps_mode,
                         float eps = 1e-10f) {

    IE_ASSERT(Layout::NCHW == src->getTensorDesc().getLayout());

    IE_ASSERT(*axes == 1);

    auto src_data = src->buffer().as<float*>();
    auto dst_data = dst->buffer().as<float*>();

    const auto& dims = src->getTensorDesc().getDims();
    auto N = dims[0];
    auto C = dims[1];
    auto H = dims[2];
    auto W = dims[3];

    for (size_t n = 0; n < N; ++n) {
        auto psrc = src_data + n * (C * H * W);
        auto pdst = dst_data + n * (C * H * W);

        std::vector<float*> src_ptrs_by_channels;
        std::vector<float*> dst_ptrs_by_channels;

        for(size_t c = 0; c < C; ++c) {
            src_ptrs_by_channels.push_back(psrc + c * H * W);
        }
        for(size_t c = 0; c < C; ++c) {
            dst_ptrs_by_channels.push_back(pdst + c * H * W);
        }
        for(size_t hw = 0; hw < H * W; ++hw) {
            float norm = 0;
            std::for_each(src_ptrs_by_channels.begin(), src_ptrs_by_channels.end(), [&norm](float* val) {
                norm += (*val) * (*val);
            });
	    if(eps_mode == ngraph::op::EpsMode::MAX) {
	        norm = 1.f / std::max(std::sqrt(norm), eps);
	    } else if(eps_mode == ngraph::op::EpsMode::ADD) {
	        norm = 1.f / std::sqrt(norm + eps);
	    }
            for(size_t i = 0; i < dst_ptrs_by_channels.size(); ++i) {
                *dst_ptrs_by_channels[i] = *src_ptrs_by_channels[i] * norm;
            }
            //increment all pointers
            for(size_t i = 0; i < src_ptrs_by_channels.size(); ++i) {
                src_ptrs_by_channels[i]++;
            }
            for(size_t i = 0; i < dst_ptrs_by_channels.size(); ++i) {
                dst_ptrs_by_channels[i]++;
            }
        }
    }
}

BlobVector refNorm(const TestNetwork::NodePtr& layer, const BlobVector& inputs, const TestNetwork&) {
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(inputs.size() == 2);

    const auto normLayer = std::dynamic_pointer_cast<ngraph::op::v0::NormalizeL2>(layer);
    IE_ASSERT(normLayer != nullptr);

    const auto input = toDefLayout(toFP32(inputs.at(0)));
    const auto axes = inputs.at(1);

    const auto& outDims = layer->output(0).get_shape();
    const auto outDesc = TensorDesc(Precision::FP32, outDims, TensorDesc::getLayoutByDims(outDims));
    auto output = make_blob_with_precision(outDesc);
    output->allocate();

    refNormalizeFromVPU(input, output, axes->cbuffer().as<int64_t*>(), normLayer->get_eps_mode(), normLayer->get_eps());

    return {output};
}

}  // namespace

TestNetwork& NormalizeLayerDef::build() {
    std::shared_ptr<ngraph::Node> normNode =
        std::make_shared<ngraph::op::v0::NormalizeL2>(
            testNet.getPort(inputPort), testNet.getPort(axesPort), params._eps, params._eps_mode);

    return testNet.addLayer(name, normNode, refNorm);
}

std::ostream& operator<<(std::ostream& os, const NormalizeParams& p) {
    vpu::formatPrint(
        os, "[_eps:%v,_eps_mode:%v]", p._eps, p._eps_mode);
    return os;
}
