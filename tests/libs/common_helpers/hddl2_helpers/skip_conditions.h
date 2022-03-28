//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
#include <gtest/gtest.h>
#include <string>

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

inline bool isEmulatorDevice() {
    const std::string deviceId = std::getenv("IE_KMB_TESTS_DEVICE_NAME") != nullptr ?
        std::getenv("IE_KMB_TESTS_DEVICE_NAME") :
        "VPUX";
    return deviceId.find("EMU") != std::string::npos;
}
