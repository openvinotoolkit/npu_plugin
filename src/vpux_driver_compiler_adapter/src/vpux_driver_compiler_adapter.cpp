//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux_driver_compiler_adapter.h"
#include "network_description.h"
#include "ngraph_transformations.h"
#include "vpux/al/config/common.hpp"
#include "zero_compiler_in_driver.h"

namespace vpux {
namespace driverCompilerAdapter {

LevelZeroCompilerAdapter::LevelZeroCompilerAdapter(): _logger("LevelZeroCompilerAdapter", LogLevel::Warning) {
    apiAdapter = std::make_shared<LevelZeroCompilerInDriver>();
}

LevelZeroCompilerAdapter::LevelZeroCompilerAdapter(const IExternalCompiler::Ptr& compilerAdapter)
        : apiAdapter(compilerAdapter), _logger("LevelZeroCompilerAdapter", LogLevel::Warning) {
}

std::shared_ptr<INetworkDescription> LevelZeroCompilerAdapter::compile(
        const std::shared_ptr<ngraph::Function>& ngraphFunc, const std::string& netName,
        const InferenceEngine::InputsDataMap& inputsInfo, const InferenceEngine::OutputsDataMap& outputsInfo,
        const vpux::Config& config) {
    _logger.setLevel(config.get<LOG_LEVEL>());
    auto IR = ngraphTransformations::serializeToIR(ngraphFunc);
    return apiAdapter->compileIR(netName, IR.xml, IR.weights, inputsInfo, outputsInfo, config);
}

// TODO #-29924: Implement query method
InferenceEngine::QueryNetworkResult LevelZeroCompilerAdapter::query(const InferenceEngine::CNNNetwork& /* network */,
                                                                    const vpux::Config& /* config */) {
    THROW_IE_EXCEPTION << "vpux::LevelZeroCompilerAdapter::query is not implemented.";
    return InferenceEngine::QueryNetworkResult();
}

std::shared_ptr<vpux::INetworkDescription> LevelZeroCompilerAdapter::parse(const std::vector<char>& blob,
                                                                           const vpux::Config& config,
                                                                           const std::string& netName) {
    _logger.setLevel(config.get<LOG_LEVEL>());
    return apiAdapter->parseBlob(netName, blob, config);
}

INFERENCE_PLUGIN_API(void)
CreateVPUXCompiler(std::shared_ptr<ICompiler>& compiler) {
    compiler = std::make_shared<LevelZeroCompilerAdapter>();
}

}  // namespace driverCompilerAdapter
}  // namespace vpux
