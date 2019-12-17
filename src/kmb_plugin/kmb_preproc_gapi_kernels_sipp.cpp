// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_VPUAL
#include "kmb_preproc_gapi_kernels.hpp"
#include <opencv2/gapi.hpp>

#include <opencv2/gapi_sipp/gsippkernel.hpp>

#include <sippDefs.h>
#include <sippSWConfig.h>

namespace InferenceEngine {
namespace gapi {
namespace preproc {

// clang-format off
// we can't let clang-format tool work with this code. It would ruin everything

GAPI_SIPP_KERNEL(GSippNV12toRGBp, GNV12toRGBp) {
    static cv::gimpl::GSIPPKernel::InitInfo Init(cv::GMatDesc, cv::GMatDesc) {
        return {SVU_SYM(svucvtColorNV12toRGB), 0, 1, 1, SIPP_RESIZE};
    }

    static void Configure(cv::GMatDesc, cv::GMatDesc, const cv::GSippConfigUserContext&)
    {}
};

GAPI_SIPP_KERNEL(GSippNV12toBGRp, GNV12toBGRp) {
    static cv::gimpl::GSIPPKernel::InitInfo Init(cv::GMatDesc, cv::GMatDesc) {
        return {SVU_SYM(svucvtColorNV12toBGR), 0, 1, 1, SIPP_RESIZE};
    }

    static void Configure(cv::GMatDesc, cv::GMatDesc, const cv::GSippConfigUserContext&)
    {}
};

GAPI_SIPP_KERNEL(GSippResizeP, GResizeP) {
    static cv::gimpl::GSIPPKernel::InitInfo Init(cv::GMatDesc, cv::gapi::own::Size, int) {
        int paramSize = sizeof(ScaleBilinearPlanarParams);
        return {SVU_SYM(svuScaleBilinearPlanar), paramSize, 2, 2, SIPP_RESIZE, true};
    }

    static void Configure(cv::GMatDesc, cv::gapi::own::Size, int, const cv::GSippConfigUserContext& ctx) {
        ScaleBilinearPlanarParams sclParams;
        sclParams.nChan = 3;
        sclParams.firstSlice = ctx.firstSlice;

        // FIXME?
        // hide in GSippFilter (return param structure)?
        sippSendFilterConfig(ctx.filter, &sclParams, sizeof(ScaleBilinearPlanarParams));
    }
};

// clang-format on

namespace sipp {
    cv::gapi::GKernelPackage kernels() {
        static auto pkg = cv::gapi::kernels
            < GSippNV12toBGRp
            , GSippNV12toRGBp
            , GSippResizeP
            >();
        return pkg;
    }
}  // namespace sipp
}  // namespace preproc
}  // namespace gapi
}  // namespace InferenceEngine
#endif
