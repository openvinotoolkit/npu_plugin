// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <common/utils.hpp>
#include <functional>
#include <optional>
#include <shared_test_classes/base/ov_subgraph.hpp>
#include <sstream>
#include <vpux/utils/core/logger.hpp>
#include <vpux_private_properties.hpp>
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"
#include "vpu_test_tool.hpp"

namespace ov::test::utils {

using SkipMessage = std::optional<std::string>;
using SkipCallback = std::function<void(std::stringstream&)>;
using VPUXPlatform = InferenceEngine::VPUXConfigParams::VPUXPlatform;

enum class VpuCompilationMode {
    ReferenceSW,
    ReferenceHW,
    DefaultHW,
};

class VpuOv2LayerTest : virtual public ov::test::SubgraphBaseTest {
protected:
    static const ov::test::utils::VpuTestEnvConfig& envConfig;
    VpuTestTool testTool;

public:
    VpuOv2LayerTest();

    void setSkipCompilationCallback(SkipCallback skipCallback);
    void setSkipInferenceCallback(SkipCallback skipCallback);

protected:
    void importModel();
    void exportModel();
    void importInput();
    void exportInput();
    void exportOutput();
    std::vector<ov::Tensor> importReference();
    void exportReference(const std::vector<ov::Tensor>& refs);

    std::vector<InferenceEngine::Blob::Ptr> ImportOutputs();

private:
    bool skipCompilationImpl();

    void printNetworkConfig() const;

    using ErrorMessage = std::optional<std::string>;
    [[nodiscard]] ErrorMessage runTest();
    [[nodiscard]] ErrorMessage skipInferenceImpl();

public:
    void setReferenceSoftwareMode();
    void setDefaultHardwareMode();

    void setSingleClusterMode();
    void setPerformanceHintLatency();
    void useELFCompilerBackend();

    bool isReferenceSoftwareMode() const;
    bool isDefaultHardwareMode() const;

    void run(VPUXPlatform platform);

    void validate() override;

private:
    // use public run(VPUXPlatform) function to always set platform explicitly
    void run() override;
    void setPlatform(VPUXPlatform platform);

    SkipCallback skipCompilationCallback = nullptr;
    SkipCallback skipInferenceCallback = nullptr;
    vpux::Logger _log = vpux::Logger::global();
};

}  // namespace ov::test::utils

namespace ov::test::subgraph {
using ov::test::utils::VpuOv2LayerTest;
using ov::test::utils::VPUXPlatform;
}  // namespace ov::test::subgraph

namespace LayerTestsDefinitions {
using ov::test::utils::VpuOv2LayerTest;
using ov::test::utils::VPUXPlatform;
}  // namespace LayerTestsDefinitions

namespace SubgraphTestsDefinitions {
using ov::test::utils::VpuOv2LayerTest;
using ov::test::utils::VPUXPlatform;
}  // namespace SubgraphTestsDefinitions

namespace ov::test {
using ov::test::utils::VpuOv2LayerTest;
using ov::test::utils::VPUXPlatform;
}  // namespace ov::test
