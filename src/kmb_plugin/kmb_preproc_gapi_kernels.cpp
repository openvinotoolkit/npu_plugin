// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if defined(__arm__) || defined(__aarch64__)
#include "kmb_preproc_gapi_kernels.hpp"

// clang-format off
namespace InferenceEngine {
namespace gapi {

cv::GMatP NV12toRGBp(const cv::GMat& y, const cv::GMat& uv) { return preproc::GNV12toRGBp::on(y, uv); }

cv::GMatP NV12toBGRp(const cv::GMat& y, const cv::GMat& uv) { return preproc::GNV12toBGRp::on(y, uv); }

cv::GMatP resizeP(const cv::GMatP& src, const cv::gapi::own::Size& dsize, int interpolation) {
    return preproc::GResizeP::on(src, dsize, interpolation);
}

cv::GMat merge3p(const cv::GMatP& src) {
    return preproc::GMerge3p::on(src);
}

cv::GMat  M2Ii(const cv::GMat &src_nv12, cv::gapi::m2i::CSC csc_code, const cv::gapi::own::Size &sz) {
    return preproc::GM2Ii::on(src_nv12, csc_code, sz);
}

cv::GMatP M2Ip(const cv::GMat &src_nv12, cv::gapi::m2i::CSC csc_code, const cv::gapi::own::Size &sz) {
    return preproc::GM2Ip::on(src_nv12, csc_code, sz);
}

}  // namespace gapi
}  // namespace InferenceEngine
// clang-format on
#endif
