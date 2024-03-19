//
// Copyright (C) 2018-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "functions.h"
#include <functional_test_utils/precision_utils.hpp>
#include <ov_models/builders.hpp>
#include <ov_models/utils/ov_helpers.hpp>
#include "base/ov_behavior_test_utils.hpp"
#include "vpux/properties.hpp"

std::shared_ptr<ov::Model> buildSingleLayerSoftMaxNetwork() {
    ov::Shape inputShape = {1, 3, 4, 3};
    ov::element::Type model_type = ov::element::f32;
    size_t axis = 1;

    const ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape({inputShape}))};
    params.at(0)->set_friendly_name("Parameter");

    const auto softMax = std::make_shared<ov::op::v1::Softmax>(params.at(0), axis);
    softMax->set_friendly_name("softMax");

    const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(softMax)};
    results.at(0)->set_friendly_name("Result");

    auto ov_model = std::make_shared<ov::Model>(results, params, "softMax");

    return ov_model;
}

const std::string PlatformEnvironment::PLATFORM = []() -> std::string {
    if (const auto var = std::getenv("IE_KMB_TESTS_PLATFORM")) {
        return var;
    } else {
        IE_THROW() << "Environment variable not set: IE_KMB_TESTS_PLATFORM.";
    }
}();

std::string getBackendName(const InferenceEngine::Core& core) {
    return core.GetMetric("NPU", ov::intel_vpux::backend_name.name()).as<std::string>();
}

std::vector<std::string> getAvailableDevices(const InferenceEngine::Core& core) {
    return core.GetMetric("NPU", METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
}
