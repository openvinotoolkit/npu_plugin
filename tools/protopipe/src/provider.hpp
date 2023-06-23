//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "simulation.hpp"

struct ExecutionProtocol {
    Simulation::Ptr simulation;
    cv::GRunArgs inputs;
    cv::GCompileArgs compile_args;
    ITermCriterion::Ptr criterion;
};

struct Scenario {
    using EP = std::shared_ptr<ExecutionProtocol>;
    std::vector<EP> protocols;
};

struct IScenarioProvider {
    virtual std::vector<Scenario> createScenarios() = 0;
    virtual ~IScenarioProvider() = default;
};
