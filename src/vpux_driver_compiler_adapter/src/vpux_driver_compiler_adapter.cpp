//
// Copyright Intel Corporation.
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

#include "vpux_driver_compiler_adapter.h"
#include "network_description.h"
#include "ngraph_transformations.h"
#include "zero_compiler_in_driver.h"
#include "vpux/al/config/common.hpp"

namespace vpux {
namespace driverCompilerAdapter {

LevelZeroCompilerAdapter::LevelZeroCompilerAdapter(): _logger("LevelZeroCompilerAdapter", LogLevel::None) {
    apiAdapter = std::make_shared<LevelZeroCompilerInDriver>();
}

LevelZeroCompilerAdapter::LevelZeroCompilerAdapter(const IExternalCompiler::Ptr& compilerAdapter)
        : apiAdapter(compilerAdapter), _logger("LevelZeroCompilerAdapter", LogLevel::None) {
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
