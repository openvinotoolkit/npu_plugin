//
// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include "vpu_test_env_cfg.hpp"
#include "vpu_test_tool.hpp"

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsUtils {

class VpuOv1LayerTestsCommon : virtual public LayerTestsCommon {
protected:
    static const VpuTestEnvConfig& envConfig;
    VpuTestTool testTool;

public:
    explicit VpuOv1LayerTestsCommon();
    void Run() override;

protected:
    void BuildNetworkWithoutCompile();
    virtual void ImportNetwork();
    void ExportNetwork();
    void ImportInput();
    void ExportInput();
    void ExportOutput();
    void ImportReference(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>& refs);
    void ExportReference(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>& refs);

    std::vector<InferenceEngine::Blob::Ptr> ImportOutputs();

    void Validate() override;

    virtual void SkipBeforeLoad() {
    }
    virtual void SkipBeforeImport() {
    }
    virtual void SkipBeforeInfer() {
    }
    virtual void SkipBeforeValidate() {
    }

    void setReferenceSoftwareModeMLIR();
    void setDefaultHardwareModeMLIR();
    void setPlatformVPU3700();
    void setPlatformVPU3720();

    void setSingleClusterMode();

    void setPerformanceHintLatency();
    void useELFCompilerBackend();

    void TearDown() override;
};

class VpuSkipTestException : public std::runtime_error {
public:
    VpuSkipTestException(const std::string& what_arg): runtime_error(what_arg){};
};

TargetDevice testPlatformTargetDevice();

}  // namespace LayerTestsUtils
