//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <openvino/runtime/device_id_parser.hpp>
#include <string>

#ifdef __aarch64__
constexpr bool IS_ON_ARM = true;
#else
constexpr bool IS_ON_ARM = false;
#endif

namespace LayerTestsUtils {

/**
 * Reads configuration environment variables
 */
class VpuTestEnvConfig {
public:
    std::string IE_NPU_TESTS_DEVICE_NAME;
    std::string IE_NPU_TESTS_DUMP_PATH;
    std::string IE_NPU_TESTS_LOG_LEVEL;
    std::string IE_NPU_TESTS_PLATFORM = "3700";

    bool IE_NPU_TESTS_RUN_COMPILER = true;
    bool IE_NPU_TESTS_RUN_EXPORT = !IS_ON_ARM;
    bool IE_NPU_TESTS_RUN_IMPORT = false;
    bool IE_NPU_TESTS_RUN_INFER = true;
    bool IE_NPU_TESTS_EXPORT_INPUT = !IS_ON_ARM;
    bool IE_NPU_TESTS_EXPORT_OUTPUT = false;
    bool IE_NPU_TESTS_EXPORT_REF = false;
    bool IE_NPU_TESTS_IMPORT_INPUT = false;
    bool IE_NPU_TESTS_IMPORT_REF = false;

    bool IE_NPU_TESTS_RAW_EXPORT = false;
    bool IE_NPU_TESTS_LONG_FILE_NAME = false;

public:
    static const VpuTestEnvConfig& getInstance();

private:
    explicit VpuTestEnvConfig();
};

std::string getTestsDeviceNameFromEnvironmentOr(const std::string& instead);
std::string getTestsPlatformFromEnvironmentOr(const std::string& instead);

std::string getDeviceNameTestCase(const std::string& str);
std::string getDeviceName();
std::string getDeviceNameID(const std::string& str);
}  // namespace LayerTestsUtils
