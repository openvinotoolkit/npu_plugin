// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_VPUAL
#include "kmb_preproc_gapi.hpp"

#include <ie_blob.h>
#include <ie_compound_blob.h>

#include <opencv2/gapi.hpp>
#include <opencv2/gapi_sipp/sippinitinfo.hpp>

#include "kmb_preproc_gapi_kernels.hpp"
#include "kmb_preproc_gapi_kernels_sipp.hpp"

#include <utility>
#include <vector>

namespace InferenceEngine {

class SIPPPreprocEngine::Priv {
    cv::GCompiled _lastCompiled;
    SizeVector    _lastInYDims;
    unsigned int  _shaveFirst;
    unsigned int  _shaveLast;

public:
    Priv(unsigned int shaveFirst, unsigned int shaveLast)
        : _shaveFirst(shaveFirst), _shaveLast(shaveLast) {}

    void preprocWithSIPP(const Blob::Ptr &inBlob, Blob::Ptr &outBlob,
                         const ResizeAlgorithm& algorithm, ColorFormat in_fmt,
                         bool omp_serial, int batch_size);
};

namespace {
namespace G {
    struct Strides {int N; int C; int H; int W;};
    struct Dims    {int N; int C; int H; int W;};
    struct Desc    {Dims d; Strides s;};

    void fix_strides_nhwc(const Dims &d, Strides &s) {
        if (s.W > d.C) {
            s.C = 1;
            s.W = s.C*d.C;
            s.H = s.W*d.W;
            s.N = s.H*d.H;
        }
    }

    Desc decompose(const TensorDesc& ie_desc) {
        const auto& ie_blk_desc = ie_desc.getBlockingDesc();
        const auto& ie_dims     = ie_desc.getDims();
        const auto& ie_strides  = ie_blk_desc.getStrides();
        const bool  nhwc_layout = ie_desc.getLayout() == NHWC;

        Dims d = {
            static_cast<int>(ie_dims[0]),
            static_cast<int>(ie_dims[1]),
            static_cast<int>(ie_dims[2]),
            static_cast<int>(ie_dims[3])
        };

        Strides s = {
            static_cast<int>(ie_strides[0]),
            static_cast<int>(nhwc_layout ? ie_strides[3] : ie_strides[1]),
            static_cast<int>(nhwc_layout ? ie_strides[1] : ie_strides[2]),
            static_cast<int>(nhwc_layout ? ie_strides[2] : ie_strides[3]),
        };

        if (nhwc_layout) fix_strides_nhwc(d, s);

        return Desc{d, s};
    }

    Desc decompose(const Blob::Ptr& blob) {
        return decompose(blob->getTensorDesc());
    }
}  // namespace G

inline int get_cv_depth(const TensorDesc &ie_desc) {
    switch (ie_desc.getPrecision()) {
    case Precision::U8:   return CV_8U;
    case Precision::FP32: return CV_32F;
    default: THROW_IE_EXCEPTION << "Unsupported data type";
    }
}

std::vector<std::vector<cv::gapi::own::Mat>> bind_to_blob(const Blob::Ptr& blob,
                                                          int batch_size) {
    const auto& ie_desc     = blob->getTensorDesc();
    const auto& ie_desc_blk = ie_desc.getBlockingDesc();
    const auto     desc     = G::decompose(blob);
    const auto cv_depth     = get_cv_depth(ie_desc);
    const auto stride       = desc.s.H*blob->element_size();
    const auto planeSize    = cv::gapi::own::Size(desc.d.W, desc.d.H);
    // Note: operating with strides (desc.s) rather than dimensions (desc.d) which is vital for ROI
    //       blobs (data buffer is shared but dimensions are different due to ROI != original image)
    const auto batch_offset = desc.s.N * blob->element_size();

    std::vector<std::vector<cv::gapi::own::Mat>> result(batch_size);

    uint8_t* blob_ptr = static_cast<uint8_t*>(blob->buffer());
    if (blob_ptr == nullptr) {
        THROW_IE_EXCEPTION << "Blob buffer is nullptr";
    }
    blob_ptr += blob->element_size()*ie_desc_blk.getOffsetPadding();

    for (int i = 0; i < batch_size; ++i) {
        uint8_t* curr_data_ptr = blob_ptr + i * batch_offset;

        std::vector<cv::gapi::own::Mat> planes;
        if (ie_desc.getLayout() == Layout::NHWC) {
            planes.emplace_back(planeSize.height, planeSize.width, CV_MAKETYPE(cv_depth, desc.d.C),
                curr_data_ptr, stride);
        } else {  // NCHW
            if (desc.d.C <= 0) {
                THROW_IE_EXCEPTION << "Invalid number of channels in blob tensor descriptor, "
                                      "expected >0, actual: " << desc.d.C;
            }
            const auto planeType = CV_MAKETYPE(cv_depth, 1);
            for (int ch = 0; ch < desc.d.C; ch++) {
                cv::gapi::own::Mat plane(planeSize.height*3, planeSize.width, planeType,
                    curr_data_ptr + ch*desc.s.C*blob->element_size(), stride);
                planes.emplace_back(plane);
            }
        }

        result[i] = std::move(planes);
    }
    return result;
}

cv::gapi::own::Size getFullImageSize(const Blob::Ptr& blob) {
    const auto desc = blob->getTensorDesc();
    IE_ASSERT(desc.getLayout() == Layout::NHWC);

    auto strides = desc.getBlockingDesc().getStrides();

    int w = strides[1] / strides[2];
    int h = strides[0] / strides[1];
    return {w, h};
}
}  // anonymous namespace

void SIPPPreprocEngine::Priv::preprocWithSIPP(const Blob::Ptr &inBlob, Blob::Ptr &outBlob,
                     const ResizeAlgorithm& algorithm, ColorFormat in_fmt,
                     bool /*omp_serial*/, int batch_size) {
    IE_ASSERT(algorithm == RESIZE_BILINEAR);
    IE_ASSERT(in_fmt == NV12);

    using namespace cv;
    using namespace cv::gapi;

    auto inNV12Blob = as<NV12Blob>(inBlob);
    IE_ASSERT(inNV12Blob != nullptr);

    const auto& y_blob = inNV12Blob->y();
    const auto& uv_blob = inNV12Blob->uv();

    auto inputs_y  = bind_to_blob(y_blob,  batch_size);
    auto inputs_uv = bind_to_blob(uv_blob, batch_size);
    auto outputs   = bind_to_blob(outBlob, batch_size);

    // FIXME: add batch

    if (!_lastCompiled) {
        GMat in_y, in_uv;
        own::Size out_sz{outputs[0][0].cols, outputs[0][0].rows/3};

        auto rgb = gapi::NV12toBGRp(in_y, in_uv);
        auto out = gapi::resizeP(rgb, out_sz);

        _lastCompiled = GComputation(GIn(in_y, in_uv), GOut(out))
                                    .compile(own::descr_of(inputs_y[0][0]), own::descr_of(inputs_uv[0][0]),
                                             compile_args(InferenceEngine::gapi::preproc::sipp::kernels(),
                                                          GSIPPBackendInitInfo{_shaveFirst, _shaveLast, 8},
                                                          GSIPPMaxFrameSizes{{getFullImageSize(y_blob),
                                                                              getFullImageSize(uv_blob)}}));
    } else if (y_blob->getTensorDesc().getDims() != _lastInYDims) {
        cv::GMetaArgs meta(2);
        meta[0] = own::descr_of(inputs_y[0][0]);
        meta[1] = own::descr_of(inputs_uv[0][0]);
        _lastCompiled.reshape(meta, {});
        _lastInYDims = y_blob->getTensorDesc().getDims();
    }
    _lastCompiled(gin(inputs_y[0][0], inputs_uv[0][0]), gout(outputs[0][0]));
}

SIPPPreprocEngine::SIPPPreprocEngine(unsigned int shaveFirst, unsigned int shaveLast)
    : _priv(new Priv(shaveFirst, shaveLast)) {}

SIPPPreprocEngine::~SIPPPreprocEngine() = default;

void SIPPPreprocEngine::preprocWithSIPP(const Blob::Ptr &inBlob, Blob::Ptr &outBlob,
                                        const ResizeAlgorithm& algorithm, ColorFormat in_fmt,
                                        bool omp_serial, int batch_size) {
    return _priv->preprocWithSIPP(inBlob, outBlob, algorithm, in_fmt, omp_serial, batch_size);
}

}  // namespace InferenceEngine
#endif
