// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <opencv2/gapi.hpp>

namespace InferenceEngine {
namespace gapi {
namespace preproc {

// clang-format off
// we can't let clang-format tool work with this code. It would ruin everything

G_TYPED_KERNEL(GNV12toRGBp, <cv::GMatP(cv::GMat, cv::GMat)>, "ie.preproc.nv12torgbp") {
    static cv::GMatDesc outMeta(cv::GMatDesc inY, cv::GMatDesc inUV) {
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
    static cv::GMatDesc outMeta(cv::GMatDesc inY, cv::GMatDesc inUV) {
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
    static cv::GMatDesc outMeta(cv::GMatDesc in, const cv::gapi::own::Size& sz, int interp) {
        GAPI_Assert(in.depth == CV_8U);
        GAPI_Assert(in.chan == 3);
        GAPI_Assert(in.planar);
        GAPI_Assert(interp == cv::INTER_LINEAR);
        return in.withSize(sz);
    }
};

G_TYPED_KERNEL(GMerge3p, <cv::GMat(cv::GMatP)>, "ie.preproc.merge3p") {
    static cv::GMatDesc outMeta(cv::GMatDesc in) {
        GAPI_Assert(in.depth == CV_8U);
        GAPI_Assert(in.chan == 3);
        GAPI_Assert(in.planar);
        return in.asInterleaved();
    }
};
}  // namespace preproc

// FIXME? remove?
cv::GMatP NV12toRGBp(const cv::GMat& src_y, const cv::GMat& src_uv);
cv::GMatP NV12toBGRp(const cv::GMat& src_y, const cv::GMat& src_uv);
cv::GMatP resizeP(const cv::GMatP& src, const cv::gapi::own::Size& dsize, int interpolation = cv::INTER_LINEAR);
cv::GMat merge3p(const cv::GMatP& src);

// clang-format on

}  // namespace gapi
}  // namespace InferenceEngine
