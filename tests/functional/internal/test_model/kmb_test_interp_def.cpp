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

#include "kmb_test_interp_def.hpp"

#include <blob_factory.hpp>

namespace {

// Reference code from CPU plugin
// It's reference only for case when Interpolation cast to Interp
static void refInterpKernel(const Blob::Ptr src,Blob::Ptr dst,bool align_corners) {
    IE_ASSERT(src != nullptr);
    IE_ASSERT(dst != nullptr);
    IE_ASSERT(Layout::NCHW == src->getTensorDesc().getLayout());

    MemoryBlob::Ptr msrc = as<MemoryBlob>(src);
    MemoryBlob::Ptr mdst = as<MemoryBlob>(dst);
    auto msrcHolder = msrc->wmap();
    auto mdstHolder = mdst->wmap();
    auto src_data = msrcHolder.as<float*>();
    auto dst_data = mdstHolder.as<float*>();

    const auto& inDims = msrc->getTensorDesc().getDims();
    const auto& outDims = mdst->getTensorDesc().getDims();
    size_t N = 1;
    auto IC = inDims[1];
    auto IH = inDims[2];
    auto IW = inDims[3];
    auto OH = outDims[2];
    auto OW = outDims[3];
    auto C = IC;

    if (IH == OH && IW == OW) {
        IE_ASSERT(N * C * OH * OW <= msrc->size());
        IE_ASSERT(N * C * OH * OW <= msrc->size());
        for (size_t i = 0; i < N * C * OH * OW; i++) {
            dst_data[i] = src_data[i];
        }
        return;
    }

    const float rh = (OH > 1 && align_corners) ? static_cast<float>(IH - 1) / (OH - 1) : static_cast<float>(IH) / OH;
    const float rw = (OW > 1 && align_corners) ? static_cast<float>(IW - 1) / (OW - 1) : static_cast<float>(IW) / OW;

    for (size_t n = 0; n < N; n++) {
        for (size_t cb = 0; cb < IC; ++cb) {
            for (size_t h = 0; h < OH; ++h) {
                float fh = rh * h;
                size_t ih0 = static_cast<size_t>(fh);
                size_t ih1 = (ih0 < IH - 1) ? ih0 + 1 : ih0;

                float h_lambda0 = fh - ih0;
                float h_lambda1 = 1.0f - h_lambda0;

                for (size_t w = 0; w < OW; ++w) {
                    float fw = rw * w;
                    size_t iw0 = static_cast<size_t>(fw);
                    size_t iw1 = (iw0 < IW - 1) ? iw0 + 1 : iw0;

                    float w_lambda0 = fw - iw0;
                    float w_lambda1 = 1.0f - w_lambda0;

                    size_t iidx00 = cb * IW * IH + ih0 * IW  + iw0;
                    size_t iidx01 = cb * IW * IH + ih0 * IW  + iw1;
                    size_t iidx10 = cb * IW * IH + ih1 * IW  + iw0;
                    size_t iidx11 = cb * IW * IH + ih1 * IW  + iw1;

                    IE_ASSERT(iidx00 <= msrc->size());
                    IE_ASSERT(iidx01 <= msrc->size());
                    IE_ASSERT(iidx10 <= msrc->size());
                    IE_ASSERT(iidx11 <= msrc->size());

                    float src00 = src_data[iidx00];
                    float src01 = src_data[iidx01];
                    float src10 = src_data[iidx10];
                    float src11 = src_data[iidx11];

                    size_t oidx = cb  * OW * OH +  h * OW  + w;
                    IE_ASSERT(oidx <= mdst->size());

                    dst_data[oidx] = h_lambda1 * (w_lambda1 * src00 + w_lambda0 * src01) +
                                  h_lambda0 * (w_lambda1 * src10 + w_lambda0 * src11);
                }
            }
        }
    }
}

BlobVector refInterp(const TestNetwork::NodePtr& layer, const BlobVector& inputs, const TestNetwork&) {
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(inputs.size() == 2);

    const auto interpLayer = std::dynamic_pointer_cast<ngraph::op::v0::Interpolate>(layer);
    IE_ASSERT(interpLayer != nullptr);
    auto align_corners = interpLayer->get_attrs().align_corners;

    const auto input = toDefLayout(toFP32(inputs.at(0)));
    const auto& outDims = layer->output(0).get_shape();
    const auto outDesc = TensorDesc(Precision::FP32, outDims, TensorDesc::getLayoutByDims(outDims));
    const auto output = make_blob_with_precision(outDesc);
    output->allocate();

    refInterpKernel(input, output, align_corners);

    return {output};
};

} // namespace

TestNetwork& InterpLayerDef::build() {
    ngraph::op::InterpolateAttrs attr;
    attr.axes = {2,3};
    attr.mode.assign("linear");
    attr.align_corners = params._alignCorners;
    attr.antialias = params._antialias; 
    attr.pads_begin.push_back(params._padBeg);
    attr.pads_end.push_back(params._padEnd);

    const auto node =
        std::make_shared<ngraph::op::v0::Interpolate>(testNet.getPort(inputPort), testNet.getPort(outshapePort), attr);

    return testNet.addLayer(name, node, refInterp);
}

std::ostream& operator<<(std::ostream& os, const InterpParams& p) {
    vpu::formatPrint(
        os, "[_align_corners:%v,_antialias:%v, _pad_begin:%v,_pad_end:%v]", p._alignCorners, p._antialias, p._padBeg, p._padEnd);
    return os;
}
