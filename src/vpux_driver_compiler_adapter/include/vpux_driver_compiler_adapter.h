//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once
#include "iexternal_compiler.h"
#include "vpux_compiler.hpp"

namespace vpux {
namespace driverCompilerAdapter {

/**
 * @brief Adapter for Compiler in driver
 * @details Wrap compiler in driver calls and do preliminary actions (like opset conversion)
 */
class LevelZeroCompilerAdapter final : public ICompiler {
public:
    LevelZeroCompilerAdapter();
    explicit LevelZeroCompilerAdapter(const IExternalCompiler::Ptr& compilerAdapter);

    std::shared_ptr<INetworkDescription> compile(const std::shared_ptr<ngraph::Function>& func,
                                                 const std::string& netName,
                                                 const InferenceEngine::InputsDataMap& inputsInfo,
                                                 const InferenceEngine::OutputsDataMap& outputsInfo,
                                                 const vpux::Config& config) final;

    InferenceEngine::QueryNetworkResult query(const InferenceEngine::CNNNetwork& network,
                                              const vpux::Config& config) final;

    std::shared_ptr<vpux::INetworkDescription> parse(const std::vector<char>& network, const vpux::Config& config,
                                                     const std::string& netName) final;

private:
    /**
     * @brief Separate externals calls to separate class
     */
    IExternalCompiler::Ptr apiAdapter;

    Logger _logger;
};

}  // namespace driverCompilerAdapter
}  // namespace vpux
