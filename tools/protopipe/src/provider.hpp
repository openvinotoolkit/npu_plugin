//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "config.hpp"
#include "scenario_provider.hpp"

#include <opencv2/gapi/infer.hpp>  // GNetPackage

#include <string>
#include <vector>

class ScenarioProvider : public IScenarioProvider {
public:
    ScenarioProvider(const std::string& filepath, const bool use_ov_old_api);

    virtual std::vector<Scenario> createScenarios() override;

private:
    cv::gapi::GNetPackage createInferenceParams(const std::string& tag, const Network& network);
    cv::gapi::GNetPackage createIEParams(const std::string& tag, const Network& network);
    cv::gapi::GNetPackage createOVParams(const std::string& tag, const Network& network);

private:
    bool m_use_ov_old_api;
    Config m_config;
};
