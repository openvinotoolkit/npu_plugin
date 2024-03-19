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
#include "common/vpu_test_env_cfg.hpp"
#include "vpux/utils/core/logger.hpp"

namespace ov::test::utils {

class VpuTestTool {
public:
    const VpuTestEnvConfig& envConfig;
    const std::string DEVICE_NAME;
    vpux::Logger _log;

public:
    explicit VpuTestTool(const VpuTestEnvConfig& envCfg);

    void exportNetwork(InferenceEngine::ExecutableNetwork& exeNet, const std::string& fsName);
    void exportModel(ov::CompiledModel& compiledModel, const std::string& fsName);
    InferenceEngine::ExecutableNetwork importNetwork(const std::shared_ptr<InferenceEngine::Core>& core,
                                                     const std::string& fsName);
    ov::CompiledModel importModel(const std::shared_ptr<ov::Core>& core, const std::string& fsName);
    void exportBlob(const InferenceEngine::Blob::Ptr blob, const std::string& fsName);
    void exportBlob(const ov::Tensor& tensor, const std::string& fsName);
    void importBlob(InferenceEngine::Blob::Ptr blob, const std::string& fsName);
    void importBlob(ov::Tensor& tensor, const std::string& fsName);
    std::string getDeviceMetric(std::string name);
};

std::string filesysName(const testing::TestInfo* testInfo, const std::string& ext, bool limitAbsPathLength);

}  // namespace ov::test::utils

namespace LayerTestsUtils {
using ov::test::utils::filesysName;
using ov::test::utils::VpuTestTool;
}  // namespace LayerTestsUtils
