//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>

#include <opencv2/gapi/cpu/gcpukernel.hpp>  // GAPI_OCV_KERNEL
#include <opencv2/gapi/gkernel.hpp>         // G_API_OP
#include <opencv2/gapi/util/variant.hpp>

#include "utils.hpp"

G_API_OP(GDummy, <cv::GMat(cv::GMat, int64_t, cv::Mat)>,
         "custom.dummy"){static cv::GMatDesc outMeta(const cv::GMatDesc& /* in */, int64_t /* delay_in_us */,
                                                     const cv::Mat& const_data){return cv::descr_of(const_data);
}
}
;

GAPI_OCV_KERNEL(GCPUDummy,
                GDummy){static void run(const cv::Mat& /* in_mat */, int64_t delay_in_us, const cv::Mat& const_data,
                                        cv::Mat& out_mat){using namespace std::chrono;
int64_t elapsed = utils::measure<microseconds>([&]() {
    const_data.copyTo(out_mat);
});
utils::busyWait(microseconds{std::max(delay_in_us - elapsed, int64_t{0})});
}
}
;
