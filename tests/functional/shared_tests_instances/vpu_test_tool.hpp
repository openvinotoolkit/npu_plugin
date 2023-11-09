//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <gtest/gtest.h>
#include <ie_common.h>
#include <ie_core.hpp>
#include <iostream>
#include <string>
#include "vpu_test_env_cfg.hpp"

namespace LayerTestsUtils {

class VpuTestTool {
public:
    const VpuTestEnvConfig& envConfig;
    const std::string DEVICE_NAME;

public:
    explicit VpuTestTool(const VpuTestEnvConfig& envCfg);

    void exportNetwork(InferenceEngine::ExecutableNetwork& exeNet, const std::string& fsName);
    InferenceEngine::ExecutableNetwork importNetwork(const std::shared_ptr<InferenceEngine::Core>& core,
                                                     const std::string& fsName);
    void exportBlob(const InferenceEngine::Blob::Ptr blob, const std::string& fsName);
    void importBlob(InferenceEngine::Blob::Ptr blob, const std::string& fsName);
    std::string getDeviceMetric(std::string name);
};

std::string filesysName(const testing::TestInfo* testInfo, const std::string& ext, bool limitAbsPathLength);

}  // namespace LayerTestsUtils
