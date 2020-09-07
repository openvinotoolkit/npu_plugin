// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if defined(__arm__) || defined(__aarch64__)
#include "kmb_preproc_gapi.hpp"

#include <ie_blob.h>
#include <ie_compound_blob.h>

#include <memory>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi_m2i/preproc.hpp>
#include <opencv2/gapi_sipp/sippinitinfo.hpp>
#include <utility>
#include <vector>

#include "kmb_preproc_gapi_kernels.hpp"
#include "kmb_preproc_gapi_kernels_sipp.hpp"

// clang-format off
namespace InferenceEngine {
namespace KmbPreproc {

class PreprocEngine::Priv {
public:
    virtual ~Priv() = default;

    virtual void go(const Blob::Ptr &inBlob, Blob::Ptr &outBlob,
                    const ResizeAlgorithm& algorithm,
                    ColorFormat in_fmt, ColorFormat out_fmt) = 0;
};

class PrivSIPP final: public PreprocEngine::Priv {
    std::unique_ptr<cv::GComputation> _comp = nullptr;
    unsigned int _shaveFirst;
    unsigned int _shaveLast;
    unsigned int _lpi;

public:
    PrivSIPP(unsigned int shaveFirst, unsigned int shaveLast, unsigned int lpi)
        : _shaveFirst(shaveFirst)
        , _shaveLast(shaveLast)
        , _lpi(lpi) {
    }

    void go(const Blob::Ptr &inBlob, Blob::Ptr &outBlob,
            const ResizeAlgorithm& algorithm,
            ColorFormat in_fmt, ColorFormat out_fmt) override;
};

class PrivM2I final: public PreprocEngine::Priv {
    std::unique_ptr<cv::GComputation> _comp = nullptr;

public:
    void go(const Blob::Ptr &inBlob, Blob::Ptr &outBlob,
            const ResizeAlgorithm& algorithm,
            ColorFormat in_fmt, ColorFormat out_fmt) override;
};

namespace {
namespace G {
struct Strides {
    int N;
    int C;
    int H;
    int W;
};
struct Dims {
    int N;
    int C;
    int H;
    int W;
};
struct Desc {
    Dims d;
    Strides s;
};

void fix_strides_nhwc(const Dims& d, Strides& s) {
    if (s.W > d.C) {
        s.C = 1;
        s.W = s.C * d.C;
        s.H = s.W * d.W;
        s.N = s.H * d.H;
    }
}

Desc decompose(const TensorDesc& ie_desc) {
    const auto& ie_blk_desc = ie_desc.getBlockingDesc();
    const auto& ie_dims = ie_desc.getDims();
    const auto& ie_strides = ie_blk_desc.getStrides();
    const bool nhwc_layout = ie_desc.getLayout() == NHWC;

    Dims d = {static_cast<int>(ie_dims[0]), static_cast<int>(ie_dims[1]), static_cast<int>(ie_dims[2]),
        static_cast<int>(ie_dims[3])};

    Strides s = {
        static_cast<int>(ie_strides[0]),
        static_cast<int>(nhwc_layout ? ie_strides[3] : ie_strides[1]),
        static_cast<int>(nhwc_layout ? ie_strides[1] : ie_strides[2]),
        static_cast<int>(nhwc_layout ? ie_strides[2] : ie_strides[3]),
    };

    if (nhwc_layout) fix_strides_nhwc(d, s);

    return Desc {d, s};
}

Desc decompose(const Blob::Ptr& blob) { return decompose(blob->getTensorDesc()); }
}  // namespace G

inline int get_cv_depth(const TensorDesc& ie_desc) {
    switch (ie_desc.getPrecision()) {
    case Precision::U8:
        return CV_8U;
    case Precision::FP32:
        return CV_32F;
    default:
        THROW_IE_EXCEPTION << "Unsupported data type";
    }
}

cv::gapi::own::Mat bind_to_blob(const Blob::Ptr& blob) {
    const auto& ie_desc     = blob->getTensorDesc();
    const auto& ie_desc_blk = ie_desc.getBlockingDesc();
    const auto     desc     = G::decompose(blob);
    const auto cv_depth     = get_cv_depth(ie_desc);
    const auto stride       = desc.s.H*blob->element_size();
    const auto size         = cv::gapi::own::Size(desc.d.W, desc.d.H);
    // Note: operating with strides (desc.s) rather than dimensions (desc.d) which is vital for ROI
    //       blobs (data buffer is shared but dimensions are different due to ROI != original image)

    uint8_t* blob_ptr = static_cast<uint8_t*>(blob->buffer());
    if (blob_ptr == nullptr) {
        THROW_IE_EXCEPTION << "Blob buffer is nullptr";
    }
    blob_ptr += blob->element_size() * ie_desc_blk.getOffsetPadding();

    if (ie_desc.getLayout() == Layout::NHWC) {
        return {size.height, size.width, CV_MAKETYPE(cv_depth, desc.d.C),
            blob_ptr, stride};
    } else {  // NCHW
        if (desc.d.C != 3) {
            THROW_IE_EXCEPTION << "Invalid number of channels in blob tensor descriptor, "
                                  "expected 3, actual: " << desc.d.C;
        }
        const auto planeType = CV_MAKETYPE(cv_depth, 1);
            return {size.height*desc.d.C, size.width, planeType,
                blob_ptr, stride};
    }
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

void PrivSIPP::go(const Blob::Ptr &inBlob, Blob::Ptr &outBlob,
                  const ResizeAlgorithm& algorithm,
                  ColorFormat in_fmt, ColorFormat out_fmt) {
    IE_ASSERT(algorithm == RESIZE_BILINEAR);
    IE_ASSERT(in_fmt == NV12);
    IE_ASSERT(out_fmt == ColorFormat::RGB || out_fmt == ColorFormat::BGR);

    using namespace cv;
    using namespace cv::gapi;

    auto inNV12Blob = as<NV12Blob>(inBlob);
    IE_ASSERT(inNV12Blob != nullptr);

    const auto& y_blob = inNV12Blob->y();
    const auto& uv_blob = inNV12Blob->uv();

    auto input_y  = bind_to_blob(y_blob);
    auto input_uv = bind_to_blob(uv_blob);
    auto output   = bind_to_blob(outBlob);

    // FIXME: add batch

    if (!_comp) {
        GMat in_y, in_uv, out;
        own::Size out_sz{output.cols, output.rows};

        auto rgb = out_fmt == ColorFormat::RGB ? gapi::NV12toRGBp(in_y, in_uv)
                                               : gapi::NV12toBGRp(in_y, in_uv);
        if (outBlob->getTensorDesc().getLayout() == NCHW) {
            out_sz.height /= 3;
            out = gapi::resizeP(rgb, out_sz);
        } else {
            auto resized = gapi::resizeP(rgb, out_sz);
            out = gapi::merge3p(resized);
        }

        _comp.reset(new GComputation(GIn(in_y, in_uv), GOut(out)));
    }

    _comp->apply(gin(input_y, input_uv), gout(output),
                 compile_args(InferenceEngine::gapi::preproc::sipp::kernels(),
                              GSIPPBackendInitInfo {_shaveFirst, _shaveLast, _lpi},
                              GSIPPMaxFrameSizes {{getFullImageSize(y_blob), getFullImageSize(uv_blob)}}));
}

void PrivM2I::go(const Blob::Ptr &inBlob, Blob::Ptr &outBlob,
                 const ResizeAlgorithm& algorithm,
                 ColorFormat in_fmt, ColorFormat out_fmt) {
    // NB.: Still follow the same constraints as with SIPP
    IE_ASSERT(algorithm == RESIZE_BILINEAR);
    IE_ASSERT(in_fmt == NV12);
    IE_ASSERT(out_fmt == ColorFormat::RGB || out_fmt == ColorFormat::BGR);

    if (out_fmt == ColorFormat::RGB) {
        THROW_IE_EXCEPTION << "M2I PP: RGB output color format is not supported";
    }

    auto inNV12Blob = as<NV12Blob>(inBlob);
    IE_ASSERT(inNV12Blob != nullptr);

    const auto& y_blob = inNV12Blob->y();
    const auto& uv_blob = inNV12Blob->uv();

    auto input_y  = bind_to_blob(y_blob);
    auto input_uv = bind_to_blob(uv_blob);
    auto output   = bind_to_blob(outBlob);

    // FIXME: add batch??

    if (!_comp) {
        cv::GMat in_y;
        cv::GMat in_uv;
        cv::GMat  out_i; // in the case of interleaved output
        cv::GMatP out_p; // in the case of planar output

        const cv::gapi::m2i::CSC csc_code = [out_fmt]() {
            switch (out_fmt) {
            case ColorFormat::RGB: return cv::gapi::m2i::CSC::NV12toRGB;
            case ColorFormat::BGR: return cv::gapi::m2i::CSC::NV12toBGR;
            default: THROW_IE_EXCEPTION << "M2I PP: Unsupported color space conversion";
            }
        }();

        cv::gapi::own::Size out_sz{output.cols, output.rows};
        if (outBlob->getTensorDesc().getLayout() == NCHW) {
            // planar output case
            out_sz.height /= 3; // see details in bind_to_blob()
            out_p = cv::gapi::M2Ip(in_y, in_uv, csc_code, out_sz);
            _comp.reset(new cv::GComputation(GIn(in_y, in_uv), cv::GOut(out_p)));
        } else {
            // interleaved output case
            out_i = cv::gapi::M2Ii(in_y, in_uv, csc_code, out_sz);
            _comp.reset(new cv::GComputation(GIn(in_y, in_uv), cv::GOut(out_i)));
        }
        IE_ASSERT(_comp != nullptr);
    }
    _comp->apply(cv::gin(input_y, input_uv), cv::gout(output),
                 cv::compile_args(cv::gapi::preproc::m2i::kernels()));
}

PreprocEngine::PreprocEngine(unsigned int shaveFirst, unsigned int shaveLast,
                             unsigned int lpi, Path ppPath) {
    IE_ASSERT(ppPath == Path::SIPP || ppPath == Path::M2I);
    if (ppPath == Path::SIPP) {
        _priv.reset(new PrivSIPP(shaveFirst, shaveLast, lpi));
    } else if (ppPath == Path::M2I) {
        _priv.reset(new PrivM2I());
    } else {
        THROW_IE_EXCEPTION << "Error: unsupported preprocessing path with code "
                           << std::to_string(static_cast<int>(ppPath));
    }
}

PreprocEngine::~PreprocEngine() = default;

void PreprocEngine::preproc(const Blob::Ptr &inBlob, Blob::Ptr &outBlob,
                            const ResizeAlgorithm& algorithm,
                            ColorFormat in_fmt, ColorFormat out_fmt) {
    return _priv->go(inBlob, outBlob, algorithm, in_fmt, out_fmt);
}

}  // namespace KmbPreproc
}  // namespace InferenceEngine
// clang-format on
#endif
