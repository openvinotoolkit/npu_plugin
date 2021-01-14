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
    const KmbTestEnvConfig& envConfig;
    const std::string DEVICE_NAME;
public:
    explicit KmbTestTool(const KmbTestEnvConfig& envCfg);

    void exportNetwork(InferenceEngine::ExecutableNetwork& exeNet, const std::string& fsName);
    InferenceEngine::ExecutableNetwork importNetwork(const std::shared_ptr<InferenceEngine::Core>& core, const std::string& fsName);
    void exportBlob(const InferenceEngine::Blob::Ptr blob, const std::string& fsName);
    void importBlob(InferenceEngine::Blob::Ptr blob, const std::string& fsName);
};

std::string filesysName(const testing::TestInfo* testInfo, const std::string& ext, bool limitAbsPathLength);

}  // namespace LayerTestsUtils
