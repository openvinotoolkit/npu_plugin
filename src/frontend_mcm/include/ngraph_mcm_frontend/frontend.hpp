//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
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
