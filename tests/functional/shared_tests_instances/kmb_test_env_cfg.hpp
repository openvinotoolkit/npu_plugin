// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#ifdef __aarch64__
constexpr bool IS_ON_ARM = true;
#else
constexpr bool IS_ON_ARM = false;
#endif

namespace LayerTestsUtils {

class KmbTestEnvConfig {
public:
    std::string IE_KMB_TESTS_DEVICE_NAME;
    std::string IE_KMB_TESTS_DUMP_PATH;
    std::string IE_KMB_TESTS_LOG_LEVEL;

    bool IE_KMB_TESTS_RUN_COMPILER = true;
    bool IE_KMB_TESTS_RUN_EXPORT = !IS_ON_ARM;
    bool IE_KMB_TESTS_RUN_IMPORT = false;
    bool IE_KMB_TESTS_RUN_INFER = IS_ON_ARM;
    bool IE_KMB_TESTS_EXPORT_INPUT = !IS_ON_ARM;
    bool IE_KMB_TESTS_EXPORT_REF = false;
    bool IE_KMB_TESTS_IMPORT_INPUT = false;
    bool IE_KMB_TESTS_IMPORT_REF = false;

    bool IE_KMB_TESTS_RAW_EXPORT = false;
    bool IE_KMB_TESTS_LONG_FILE_NAME = false;

    bool IE_VPUX_USE_EXPERIMENTAL_COMPILER = false;
public:
    explicit KmbTestEnvConfig();
};

}  // namespace LayerTestsUtils
