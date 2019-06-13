// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layout_tests.hpp"

static auto params_myriad = Combine(
        Values(conv_p),
        Values(std::make_pair(Precision::FP16, 1e-1)),
        Values(NCHW, NHWC),
        Values(NCHW, NHWC),
        Values(Precision::FP32, Precision::U8)  // TODO: What about U16/I8/FP16?
);

#ifdef USE_MYRIAD
    PLUGING_CASE_WITH_SUFFIX(myriad, _nightly, LayoutTTTest, params_myriad);
#endif
#ifdef USE_HDDL
    PLUGING_CASE_WITH_SUFFIX(HDDL, _nightly, LayoutTTTest, params_myriad);
#endif
#ifdef USE_KMB
    PLUGING_CASE_WITH_SUFFIX(kmb, _nightly, LayoutTTTest, params_myriad);
#endif
