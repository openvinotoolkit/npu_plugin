//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//
#include <gtest/gtest.h>
#include "hddl2_metrics.h"

#   define SKIP_IF_NO_DEVICE()                                              \
    do {                                                                    \
        if (!vpu::HDDL2Plugin::HDDL2Metrics::isServiceAvailable()) {        \
            SKIP() << "Skip test due to absence of HDDL2 device";           \
        }                                                                   \
    } while (false)

#   define SKIP_IF_DEVICE()                                                 \
    do {                                                                    \
        if (vpu::HDDL2Plugin::HDDL2Metrics::isServiceAvailable()) {         \
            SKIP() << "This test require device disabled";                  \
        }                                                                   \
    } while (false)
