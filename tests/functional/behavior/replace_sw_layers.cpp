//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <base/ov_behavior_test_utils.hpp>
#include <functional_test_utils/precision_utils.hpp>
#include <ie_core.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <string>
#include <vector>
#include "common/functions.h"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "vpux/al/config/common.hpp"
#include "vpux/properties.hpp"
#include "vpux_private_properties.hpp"

using CompileWithDummy = ov::test::behavior::OVInferRequestTests;

std::shared_ptr<ov::Model> buildSingleLayerClampNetwork() {  // Clamp is not supported in SW
    ov::Shape inputShape = {1, 3, 4, 3};
    ov::element::Type netPrecision = ov::element::f32;

    const ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(netPrecision, ov::Shape{inputShape})};

    const auto clamp = std::make_shared<ov::op::v0::Clamp>(params.at(0), 0., 1.);

    const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(clamp)};

    auto ov_model = std::make_shared<ov::Model>(results, params, "clamp");

    return ov_model;
}

namespace {

TEST_P(CompileWithDummy, CompilationForSpecificPlatform) {
    if (getBackendName(*core) == "LEVEL0") {
        GTEST_SKIP() << "Skip due to failure on device";
    }
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        const auto& ov_model = buildSingleLayerClampNetwork();
        ASSERT_NO_THROW(auto compiled_model = core->compile_model(ov_model, target_device, configuration));
    }
}

const std::vector<ov::AnyMap> configs = {{{ov::intel_vpux::platform(ov::intel_vpux::VPUXPlatform::VPU3720)},
                                          {ov::intel_vpux::compilation_mode_params("dummy-op-replacement=true")}}};
// Must be successfully compiled with dummy-op-replacement=true

INSTANTIATE_TEST_SUITE_P(smoke_precommit_DummyVPU3720, CompileWithDummy,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         CompileWithDummy::getTestCaseName);
}  // namespace
