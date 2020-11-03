// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#if defined(__arm__) || defined(__aarch64__)
#include <opencv2/gapi.hpp>

namespace InferenceEngine {
namespace gapi {
namespace preproc {
// clang-format off
// we can't let clang-format tool work with this code. It would ruin everything

G_TYPED_KERNEL(GNV12toRGBp, <cv::GMatP(cv::GMat, cv::GMat)>, "ie.preproc.nv12torgbp") {
    static cv::GMatDesc outMeta(const cv::GMatDesc &inY, const cv::GMatDesc &inUV) {
        GAPI_Assert(inY.depth == CV_8U);
        GAPI_Assert(inUV.depth == CV_8U);
        GAPI_Assert(inY.chan == 1);
        GAPI_Assert(inY.planar == false);
        GAPI_Assert(inUV.chan == 2);
        GAPI_Assert(inUV.planar == false);
        GAPI_Assert(inY.size.width  == 2 * inUV.size.width);
        GAPI_Assert(inY.size.height == 2 * inUV.size.height);
        return inY.withType(CV_8U, 3).asPlanar();
    }
};

G_TYPED_KERNEL(GNV12toBGRp, <cv::GMatP(cv::GMat, cv::GMat)>, "ie.preproc.nv12tobgrp") {
    static cv::GMatDesc outMeta(const cv::GMatDesc &inY, const cv::GMatDesc &inUV) {
        GAPI_Assert(inY.depth == CV_8U);
        GAPI_Assert(inUV.depth == CV_8U);
        GAPI_Assert(inY.chan == 1);
        GAPI_Assert(inY.planar == false);
        GAPI_Assert(inUV.chan == 2);
        GAPI_Assert(inUV.planar == false);
        GAPI_Assert(inY.size.width  == 2 * inUV.size.width);
        GAPI_Assert(inY.size.height == 2 * inUV.size.height);
        return inY.withType(CV_8U, 3).asPlanar();
    }
};

G_TYPED_KERNEL(GResizeP, <cv::GMatP(cv::GMatP, cv::gapi::own::Size, int)>, "ie.preproc.resizeP") {
    static cv::GMatDesc outMeta(const cv::GMatDesc &in, const cv::gapi::own::Size& sz, int interp) {
        GAPI_Assert(in.depth == CV_8U);
        GAPI_Assert(in.chan == 3);
        GAPI_Assert(in.planar);
        GAPI_Assert(interp == cv::INTER_LINEAR);
        return in.withSize(sz);
    }
};

G_TYPED_KERNEL(GMerge3p, <cv::GMat(cv::GMatP)>, "ie.preproc.merge3p") {
    static cv::GMatDesc outMeta(const cv::GMatDesc &in) {
        GAPI_Assert(in.depth == CV_8U);
        GAPI_Assert(in.chan == 3);
        GAPI_Assert(in.planar);
        return in.asInterleaved();
    }
};

G_TYPED_KERNEL(GScalePlanes, <cv::GMatP(cv::GMatP, cv::gapi::own::Size, int)>, "ie.preproc.scalePlanes") {
    static cv::GMatDesc outMeta(const cv::GMatDesc& in, const cv::gapi::own::Size& dsize, int interp) {
        GAPI_Assert(in.depth == CV_8U);
        GAPI_Assert(in.planar);
        GAPI_Assert(in.chan == 2 || in.chan == 3 || in.chan == 4);
        GAPI_Assert(interp == cv::INTER_LINEAR || interp == cv::INTER_AREA);
        return in.withSize(dsize);
    }
};

G_TYPED_KERNEL(GMerge2, <cv::GMat(cv::GMatP)>, "ie.preproc.merge2") {
    static cv::GMatDesc outMeta(const cv::GMatDesc& in) {
        GAPI_Assert(in.depth == CV_8U);
        GAPI_Assert(in.chan == 2);
        GAPI_Assert(in.planar);
        return in.asInterleaved();
    }
};

G_TYPED_KERNEL(GMerge4, <cv::GMat(cv::GMatP)>, "ie.preproc.merge4") {
    static cv::GMatDesc outMeta(const cv::GMatDesc& in) {
        GAPI_Assert(in.depth == CV_8U);
        GAPI_Assert(in.chan == 4);
        GAPI_Assert(in.planar);
        return in.asInterleaved();
    }
};

G_TYPED_KERNEL(GDrop4, <cv::GMatP(cv::GMatP)>, "ie.preproc.drop4") {
    static cv::GMatDesc outMeta(const cv::GMatDesc& in) {
        GAPI_Assert(in.depth == CV_8U);
        GAPI_Assert(in.chan == 4);
        GAPI_Assert(in.planar);
        return in.withType(in.depth, 3);
    }
};

G_TYPED_KERNEL(GSwapChan, <cv::GMatP(cv::GMatP)>, "ie.preproc.swapChan") {
    static cv::GMatDesc outMeta(const cv::GMatDesc& in) {
        GAPI_Assert(in.depth == CV_8U);
        GAPI_Assert(in.chan == 3);
        GAPI_Assert(in.planar);
        return in;
    }
};

G_TYPED_KERNEL(GInterleaved2planar, <cv::GMatP(cv::GMat)>, "ie.preproc.interleaved2planar") {
    static cv::GMatDesc outMeta(const cv::GMatDesc& in) {
        GAPI_Assert(in.depth == CV_8U);
        GAPI_Assert(in.chan == 2 || in.chan == 3 || in.chan == 4);
        GAPI_Assert(!in.planar);
        return in.asPlanar();
    }
};

}  // namespace preproc

// FIXME? remove?
cv::GMatP NV12toRGBp(const cv::GMat& src_y, const cv::GMat& src_uv);
cv::GMatP NV12toBGRp(const cv::GMat& src_y, const cv::GMat& src_uv);
cv::GMatP resizeP(const cv::GMatP& src, const cv::gapi::own::Size& dsize, int interp = cv::INTER_LINEAR);
cv::GMat merge3p(const cv::GMatP& src);
// FIXME: Kernels below don't have vpu version yet
cv::GMatP scalePlanes(const cv::GMatP& in, const cv::gapi::own::Size& dsize, int interp);
cv::GMat merge2(const cv::GMatP& in);
cv::GMat merge4(const cv::GMatP& in);
cv::GMatP drop4(const cv::GMatP& in);
cv::GMatP swapChan(const cv::GMatP& in);
cv::GMatP interleaved2planar(const cv::GMat& in);

// clang-format on

}  // namespace gapi
}  // namespace InferenceEngine

#endif
