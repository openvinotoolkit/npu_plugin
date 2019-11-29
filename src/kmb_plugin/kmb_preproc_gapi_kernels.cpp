// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_VPUAL
#include "kmb_preproc_gapi_kernels.hpp"

namespace InferenceEngine {
namespace gapi {

cv::GMatP NV12toRGBp(const cv::GMat& y, const cv::GMat& uv) { return preproc::GNV12toRGBp::on(y, uv); }

cv::GMatP NV12toBGRp(const cv::GMat& y, const cv::GMat& uv) { return preproc::GNV12toBGRp::on(y, uv); }

cv::GMatP resizeP(const cv::GMatP& src, const cv::gapi::own::Size& dsize, int interpolation) {
    return preproc::GResizeP::on(src, dsize, interpolation);
}

}  // namespace gapi
}  // namespace InferenceEngine
#endif
