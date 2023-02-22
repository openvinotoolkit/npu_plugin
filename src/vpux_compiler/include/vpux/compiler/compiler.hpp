//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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

/**
 * @enum IRPrintingOrder
 * @brief VPUX IR pass printing before/after or before and after
 */
enum class IRPrintingOrder {
    BEFORE,
    AFTER,
    BEFORE_AFTER,
};

}  // namespace vpux
