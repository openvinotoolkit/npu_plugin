//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_test_env_cfg.hpp"

#include <details/ie_exception.hpp>
#include "vpux/utils/IE/config.hpp"

#include <cstdlib>
#include <stdexcept>

namespace LayerTestsUtils {

KmbTestEnvConfig::KmbTestEnvConfig() {
    if (auto var = std::getenv("IE_KMB_TESTS_DEVICE_NAME")) {
        IE_KMB_TESTS_DEVICE_NAME = var;
    }

    if (auto var = std::getenv("IE_KMB_TESTS_DUMP_PATH")) {
        IE_KMB_TESTS_DUMP_PATH = var;
    }

    if (auto var = std::getenv("IE_KMB_TESTS_LOG_LEVEL")) {
        IE_KMB_TESTS_LOG_LEVEL = var;
    }

    if (auto var = std::getenv("IE_KMB_TESTS_RUN_COMPILER")) {
        IE_KMB_TESTS_RUN_COMPILER = vpux::envVarStrToBool("IE_KMB_TESTS_RUN_COMPILER", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_RUN_EXPORT")) {
        IE_KMB_TESTS_RUN_EXPORT = vpux::envVarStrToBool("IE_KMB_TESTS_RUN_EXPORT", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_RUN_IMPORT")) {
        IE_KMB_TESTS_RUN_IMPORT = vpux::envVarStrToBool("IE_KMB_TESTS_RUN_IMPORT", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_RUN_INFER")) {
        IE_KMB_TESTS_RUN_INFER = vpux::envVarStrToBool("IE_KMB_TESTS_RUN_INFER", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_EXPORT_INPUT")) {
        IE_KMB_TESTS_EXPORT_INPUT = vpux::envVarStrToBool("IE_KMB_TESTS_EXPORT_INPUT", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_EXPORT_OUTPUT")) {
        IE_KMB_TESTS_EXPORT_OUTPUT = vpux::envVarStrToBool("IE_KMB_TESTS_EXPORT_OUTPUT", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_EXPORT_REF")) {
        IE_KMB_TESTS_EXPORT_REF = vpux::envVarStrToBool("IE_KMB_TESTS_EXPORT_REF", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_IMPORT_INPUT")) {
        IE_KMB_TESTS_IMPORT_INPUT = vpux::envVarStrToBool("IE_KMB_TESTS_IMPORT_INPUT", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_IMPORT_REF")) {
        IE_KMB_TESTS_IMPORT_REF = vpux::envVarStrToBool("IE_KMB_TESTS_IMPORT_REF", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_RAW_EXPORT")) {
        IE_KMB_TESTS_RAW_EXPORT = vpux::envVarStrToBool("IE_KMB_TESTS_RAW_EXPORT", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_LONG_FILE_NAME")) {
        IE_KMB_TESTS_LONG_FILE_NAME = vpux::envVarStrToBool("IE_KMB_TESTS_LONG_FILE_NAME", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_PLATFORM")) {
        IE_KMB_TESTS_PLATFORM = var;
    }
}

const KmbTestEnvConfig& KmbTestEnvConfig::getInstance() {
    static KmbTestEnvConfig instance{};
    return instance;
}

}  // namespace LayerTestsUtils
