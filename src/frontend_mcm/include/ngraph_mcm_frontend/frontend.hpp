//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

// clang-format off

#include "vpux/utils/IE/config.hpp"

#include <include/mcm/compiler/compilation_unit.hpp>

#include <cpp/ie_cnn_network.h>
#include <ngraph/function.hpp>

#include <memory>
#include <string>
#include <vector>

namespace ie = InferenceEngine;

//
// Useful environment variables:
//
//   * NGRAPH_ENABLE_VISUALIZE_TRACING=1
//   * NGRAPH_VISUALIZE_TRACING_FORMAT=dot
//   * NGRAPH_VISUALIZE_TREE_OUTPUT_SHAPES=1
//   * NGRAPH_VISUALIZE_TREE_OUTPUT_TYPES=1
//

std::unique_ptr<mv::CompilationUnit> compileNGraphIntoCompilationUnit(
        const std::shared_ptr<ngraph::Function>& func,
        const std::string& netName,
        const ie::InputsDataMap& inputsInfo,
        const ie::OutputsDataMap& outputsInfo,
        const vpux::Config& config,
        std::string & errMsg);

std::vector<char> serializeCompilationUnit(
        const std::unique_ptr<mv::CompilationUnit>& compUnit,
        std::string & errMsg);

std::vector<char> compileNGraph(
        const std::shared_ptr<ngraph::Function>& func,
        const std::string& netName,
        const ie::InputsDataMap& inputsInfo,
        const ie::OutputsDataMap& outputsInfo,
        const vpux::Config& config,
        std::string & errMsg);

std::shared_ptr<std::unordered_set<std::string>> getSupportedLayers(
        const InferenceEngine::CNNNetwork& func,
        const vpux::Config& config);

// clang-format on
