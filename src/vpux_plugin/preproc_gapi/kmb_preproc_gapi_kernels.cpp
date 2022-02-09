// Copyright (C) 2019 Intel Corporation
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

#if defined(__arm__) || defined(__aarch64__)
#include "kmb_preproc_gapi_kernels.hpp"

// clang-format off
namespace InferenceEngine {
namespace gapi {

cv::GMatP NV12toRGBp(const cv::GMat& y, const cv::GMat& uv) {
    return preproc::GNV12toRGBp::on(y, uv);
}

cv::GMatP NV12toBGRp(const cv::GMat& y, const cv::GMat& uv) {
    return preproc::GNV12toBGRp::on(y, uv);
}

cv::GMatP resizeP(const cv::GMatP& src, const cv::gapi::own::Size& dsize, int interpolation) {
    return preproc::GResizeP::on(src, dsize, interpolation);
}

cv::GMat merge3p(const cv::GMatP& src) {
    return preproc::GMerge3p::on(src);
}

// FIXME: Kernels below don't have vpu version yet

// area/bilinear 1,2,3,4 planes
cv::GMatP scalePlanes(const cv::GMatP& in, const cv::gapi::own::Size& dsize, int interp) {
    return preproc::GScalePlanes::on(in, dsize, interp);
}

cv::GMat merge2(const cv::GMatP& in) {
    return preproc::GMerge2::on(in);
}

cv::GMat merge4(const cv::GMatP& in) {
    return preproc::GMerge4::on(in);
}

// convert 4 chan to 3 chan planar image
cv::GMatP drop4(const cv::GMatP& in) {
    return preproc::GDrop4::on(in);
}

// 3-chan only
cv::GMatP swapChan(const cv::GMatP& in) {
    return preproc::GSwapChan::on(in);
}

// 2,3,4 chans
cv::GMatP interleaved2planar(const cv::GMat& in) {
    return preproc::GInterleaved2planar::on(in);
}

}  // namespace gapi
}  // namespace InferenceEngine
// clang-format on
#endif
