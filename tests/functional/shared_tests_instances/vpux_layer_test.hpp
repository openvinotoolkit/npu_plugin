// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common/utils.hpp>
#include <functional>
#include <optional>
#include <shared_test_classes/base/ov_subgraph.hpp>
#include <sstream>
#include <vpux/utils/core/logger.hpp>
#include <vpux_private_properties.hpp>

using SkipMessage = std::optional<std::string>;
using SkipCallback = std::function<void(std::stringstream&)>;
using VPUXPlatform = InferenceEngine::VPUXConfigParams::VPUXPlatform;

static const ov::test::TargetDevice targetDevice = CommonTestUtils::DEVICE_KEEMBAY;

enum class VPUXCompilationMode {
    ReferenceSW,
    ReferenceHW,
    DefaultHW,
};

class VPUXLayerTest : virtual public ov::test::SubgraphBaseTest {
public:
    VPUXLayerTest();

    void setSkipCompilationCallback(SkipCallback skipCallback);
    void setSkipInferenceCallback(SkipCallback skipCallback);

private:
    bool skipCompilationImpl();
    bool skipInferenceImpl();

    void printNetworkConfig() const;

    using ErrorMessage = std::optional<std::string>;
    [[nodiscard]] ErrorMessage runTest();

public:
    void setReferenceSoftwareMode();
    void setDefaultHardwareMode();

    void setSingleClusterMode();
    void setPerformanceHintLatency();
    void useELFCompilerBackend();

    bool isReferenceSoftwareMode() const;
    bool isDefaultHardwareMode() const;

    void run(VPUXPlatform platform);

private:
    // use public run(VPUXPlatform) function to always set platform explicitly
    void run() override;
    void setPlatform(VPUXPlatform platform);

    SkipCallback skipCompilationCallback = nullptr;
    SkipCallback skipInferenceCallback = nullptr;
    vpux::Logger _log = vpux::Logger::global();
};
