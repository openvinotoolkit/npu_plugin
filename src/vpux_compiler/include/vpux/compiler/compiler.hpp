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

#include "vpux_compiler.hpp"

namespace vpux {

class CompilerImpl final : public ICompiler {
public:
    std::shared_ptr<INetworkDescription> compile(const std::shared_ptr<ngraph::Function>& func,
                                                 const std::string& netName,
                                                 const InferenceEngine::InputsDataMap& inputsInfo,
                                                 const InferenceEngine::OutputsDataMap& outputsInfo,
                                                 const Config& config) final;

    InferenceEngine::QueryNetworkResult query(const InferenceEngine::CNNNetwork& network,
                                              const vpux::Config& config) final;

    std::shared_ptr<INetworkDescription> parse(const std::vector<char>& network, const Config& config,
                                               const std::string& graphName) final;
};

}  // namespace vpux
