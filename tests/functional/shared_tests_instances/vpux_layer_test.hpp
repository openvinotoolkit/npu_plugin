// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common/utils.hpp>
#include <shared_test_classes/base/ov_subgraph.hpp>
#include <vpux/utils/core/logger.hpp>
#include <vpux/utils/core/optional.hpp>
#include "kmb_test_env_cfg.hpp"
#include "kmb_test_tool.hpp"
#include "vpux_private_config.hpp"

namespace VPUXLayerTestsUtils {

using SkipMessage = vpux::Optional<std::string>;

class VPUXLayerTestsCommon : virtual public ov::test::SubgraphBaseTest {
public:
    VPUXLayerTestsCommon();

    static const LayerTestsUtils::KmbTestEnvConfig envConfig;

    virtual SkipMessage SkipBeforeLoad();
    virtual SkipMessage SkipBeforeInfer();
    virtual SkipMessage SkipBeforeValidate();

private:
    bool skipBeforeLoadImpl();
    bool skipBeforeInferImpl();
    bool skipBeforeValidateImpl();

    virtual void importNetwork();
    void exportNetwork();
    void importInput();
    void exportInput();
    void exportOutput();
    void exportReference();

    void printNetworkConfig() const;

public:
    void useCompilerMLIR();
    void setReferenceSoftwareModeMLIR();
    void setDefaultHardwareModeMLIR();
    void setPlatformVPUX37XX();

    bool isCompilerMCM() const;
    bool isCompilerMLIR() const;
    bool isPlatformVPUX37XX() const;

protected:
    void run() override;
    void configure_model() override;
    
private:
    vpux::Logger _log = vpux::Logger::global();
};

const ov::test::TargetDevice testPlatformTargetDevice = []() -> ov::test::TargetDevice {
    if (const auto var = std::getenv("IE_KMB_TESTS_DEVICE_NAME")) {
        return var;
    }

    return CommonTestUtils::DEVICE_KEEMBAY;
}();

}  // namespace VPUXLayerTestsUtils
