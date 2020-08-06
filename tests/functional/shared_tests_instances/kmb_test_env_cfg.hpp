// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

namespace LayerTestsUtils {

class KmbTestEnvConfig {
public:
    std::string IE_KMB_TESTS_DEVICE_NAME;
    std::string IE_KMB_TESTS_DUMP_PATH;
    std::string IE_KMB_TESTS_LOG_LEVEL;
    bool IE_KMB_TESTS_RAW_EXPORT;
    bool IE_KMB_TESTS_RUN_INFER;
public:
    explicit KmbTestEnvConfig();
};

}  // namespace LayerTestsUtils
