//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include <string>
#include <vpu_layers_tests.hpp>

namespace KmbRegressionTarget {

struct CompilationParameter {
    CompilationParameter() = default;
    CompilationParameter(std::string pName, std::string pathToNetwork, std::string pathToWeights)
            : name(pName), path_to_network(pathToNetwork), path_to_weights(pathToWeights){};

    std::string name;
    std::string path_to_network;
    std::string path_to_weights;
};
}  // namespace KmbRegressionTarget
