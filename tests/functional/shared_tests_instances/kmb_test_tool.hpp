// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <gtest/gtest.h>
#include <ie_common.h>
#include <ie_core.hpp>
#include "kmb_test_env_cfg.hpp"

namespace LayerTestsUtils {

class KmbTestTool {
public:
    static const KmbTestEnvConfig envConfig;
    const std::string DEVICE_NAME;
public:
    explicit KmbTestTool();

    void exportNetwork(InferenceEngine::ExecutableNetwork& exeNet, const std::string& testName);
    InferenceEngine::ExecutableNetwork importNetwork(const std::shared_ptr<InferenceEngine::Core>& core, const std::string& testName);
};

std::string filesysTestName(const testing::TestInfo*);

}  // namespace LayerTestsUtils
