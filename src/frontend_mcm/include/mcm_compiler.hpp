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
