//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/IE/config.hpp"
#include "vpux_compiler.hpp"

#include <ie_common.h>
#include <description_buffer.hpp>

#include <memory>

class MCMCompiler final : public vpux::ICompiler {
public:
    std::shared_ptr<vpux::INetworkDescription> compile(const std::shared_ptr<ngraph::Function>& func,
                                                       const std::string& netName,
                                                       const InferenceEngine::InputsDataMap& inputsInfo,
                                                       const InferenceEngine::OutputsDataMap& outputsInfo,
                                                       const vpux::Config& config) override;
    InferenceEngine::QueryNetworkResult query(const InferenceEngine::CNNNetwork& network,
                                              const vpux::Config& config) override;

    std::shared_ptr<vpux::INetworkDescription> parse(const std::vector<char>& network, const vpux::Config& config,
                                                     const std::string& graphName = "") override;
};
