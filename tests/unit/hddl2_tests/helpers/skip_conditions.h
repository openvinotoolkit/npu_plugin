//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
#include <gtest/gtest.h>

#include "hddl2_backend.h"

inline bool canWorkWithDevice() {
    return vpux::hddl2::HDDL2Backend::isServiceAvailable();
}

#   define SKIP_IF_NO_DEVICE()                                              \
    do {                                                                    \
        if (!canWorkWithDevice()) {             \
            GTEST_SKIP() << "Skip test due to absence of HDDL2 device";           \
        }                                                                   \
    } while (false)
