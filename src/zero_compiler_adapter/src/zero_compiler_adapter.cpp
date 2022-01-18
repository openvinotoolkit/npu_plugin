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

#include "zero_compiler_adapter.h"
#include <chrono>
#include "network_description.h"
#include "ngraph_transformations.h"
#include "zero_compiler_in_driver.h"

namespace vpux {
namespace zeroCompilerAdapter {

// FIXME Fix logger, incorrect log level
LevelZeroCompilerAdapter::LevelZeroCompilerAdapter(): _logger("LevelZeroCompilerAdapter", LogLevel::Debug) {
    apiAdapter = std::make_shared<LevelZeroCompilerInDriver>();
}

// FIXME Fix logger, incorrect log level
LevelZeroCompilerAdapter::LevelZeroCompilerAdapter(const IExternalCompiler::Ptr& compilerAdapter)
        : apiAdapter(compilerAdapter), _logger("LevelZeroCompilerAdapter", LogLevel::Debug) {
}

// TODO How to use inputsInfo, outputsInfo ?
// TODO Fix config usage
std::shared_ptr<INetworkDescription> LevelZeroCompilerAdapter::compile(
        const std::shared_ptr<ngraph::Function>& ngraphFunc, const std::string& netName,
        const InferenceEngine::InputsDataMap& /*inputsInfo*/, const InferenceEngine::OutputsDataMap& /*outputsInfo*/,
        const vpux::Config& /*config*/) {
    auto IR = ngraphTransformations::serializeToIR(ngraphFunc);
    return apiAdapter->compileIR(netName, IR.xml, IR.weights);
}

InferenceEngine::QueryNetworkResult LevelZeroCompilerAdapter::query(const InferenceEngine::CNNNetwork& /* network */,
                                                                    const vpux::Config& /* config */) {
    THROW_IE_EXCEPTION << "vpux::LevelZeroCompilerAdapter::query is not implemented.";
    return InferenceEngine::QueryNetworkResult();
}

// TODO config usage
std::shared_ptr<vpux::INetworkDescription> LevelZeroCompilerAdapter::parse(const std::vector<char>& blob,
                                                                           const vpux::Config& /* config */,
                                                                           const std::string& netName) {
    return apiAdapter->parseBlob(netName, blob);
}

INFERENCE_PLUGIN_API(void)
CreateVPUXCompiler(std::shared_ptr<ICompiler>& compiler) {
    compiler = std::make_shared<LevelZeroCompilerAdapter>();
}

}  // namespace zeroCompilerAdapter
}  // namespace vpux
