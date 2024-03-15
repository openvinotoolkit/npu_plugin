//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/gapi/gcommon.hpp>  // cv::GCompileArgs

#include "parser/config.hpp"
#include "scenario/criterion.hpp"
#include "scenario/scenario_graph.hpp"

struct StreamDesc {
    std::string name;
    ScenarioGraph graph;
    ITermCriterion::Ptr criterion;
    cv::GCompileArgs compile_args;
    cv::util::optional<uint32_t> target_latency;
    cv::util::optional<std::filesystem::path> per_iter_outputs_path;
};

struct ScenarioDesc {
    std::string name;
    std::vector<StreamDesc> streams;
};

struct IScenarioParser {
    virtual std::vector<ScenarioDesc> parseScenarios() = 0;
    virtual ~IScenarioParser() = default;
};

// NB: There is only one parser so far.
// In fact it's not even a parser split it on parser and builder later.
class ScenarioParser : public IScenarioParser {
public:
    ScenarioParser(const std::string& filepath, const bool use_ov_old_api);

    std::vector<ScenarioDesc> parseScenarios() override;

private:
    cv::gapi::GNetPackage createInferenceParams(const std::string& tag, const Network& network);
    cv::gapi::GNetPackage createIEParams(const std::string& tag, const Network& network);
    cv::gapi::GNetPackage createOVParams(const std::string& tag, const Network& network);

private:
    bool m_use_ov_old_api;
    ScenarioConfig m_config;
};
