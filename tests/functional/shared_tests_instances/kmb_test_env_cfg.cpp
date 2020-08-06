// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_test_env_cfg.hpp"

#include <cstdlib>
#include <stdexcept>
#include <vpu/utils/error.hpp>

namespace LayerTestsUtils {

namespace {

bool strToBool(const char* varName, const char* varValue) {
    try {
        const auto intVal = std::stoi(varValue);
        if (intVal != 0 && intVal != 1) {
            throw std::invalid_argument("Only 0 and 1 values are supported");
        }
        return (intVal != 0);
    } catch (const std::exception& e) {
        THROW_IE_EXCEPTION << "Environment variable " << varName << " has wrong value : " << e.what();
    }
}

}  // namespace

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
    if (auto var = std::getenv("IE_KMB_TESTS_RAW_EXPORT")) {
        IE_KMB_TESTS_RAW_EXPORT = strToBool("IE_KMB_TESTS_RAW_EXPORT", var);
    }
    if (auto var = std::getenv("IE_KMB_TESTS_RUN_INFER")) {
        IE_KMB_TESTS_RUN_INFER = strToBool("IE_KMB_TESTS_RUN_INFER", var);
    } else {
        IE_KMB_TESTS_RUN_INFER = true;
    }
}

}  // namespace LayerTestsUtils
