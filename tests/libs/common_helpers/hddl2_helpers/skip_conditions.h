//
// Copyright 2020 Intel Corporation.
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
#include <gtest/gtest.h>

#include "hddl2_backend.h"

#   define SKIP_IF_NO_DEVICE()                                              \
    do {                                                                    \
        if (!vpux::hddl2::HDDL2Backend::isServiceAvailable()) {        \
            GTEST_SKIP() << "Skip test due to absence of HDDL2 device";           \
        }                                                                   \
    } while (false)

#   define SKIP_IF_DEVICE()                                                 \
    do {                                                                    \
        if (vpux::hddl2::HDDL2Backend::isServiceAvailable()) {         \
            GTEST_SKIP() << "This test require device disabled";                  \
        }                                                                   \
    } while (false)
