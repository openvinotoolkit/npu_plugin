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
#include "network_description.h"
#include "ngraph_transformations.h"
#include "vpux_compiler_l0_adapter.h"
#include "zero_api_adapter.h"
#include <chrono>

namespace vpux {
namespace zeroCompilerAdapter {

ZeroCompilerAdapter::ZeroCompilerAdapter() {
    // apiAdapter = std::make_shared<VPUXCompilerL0>();
    apiAdapter = std::make_shared<ZeroAPICompilerInDriver>();
}

// TODO How to use inputsInfo, outputsInfo ?
// TODO Fix netName usage
std::shared_ptr<INetworkDescription> ZeroCompilerAdapter::compile(
        const std::shared_ptr<ngraph::Function>& ngraphFunc, const std::string& /*netName*/,
        const InferenceEngine::InputsDataMap& /*inputsInfo*/, const InferenceEngine::OutputsDataMap& /*outputsInfo*/,
        const VPUXConfig& /*config*/) {
    using ms = std::chrono::milliseconds;
    auto start = std::chrono::high_resolution_clock::now();
    
    //------------------------------------------------------------------------------
    _logger->debug("Get information about opset versions from compiler");
    //------------------------------------------------------------------------------
    // TODO Not implemented
    //------------------------------------------------------------------------------
    _logger->debug("Modify network (ngraph) according to supported opset");
    //------------------------------------------------------------------------------
    // TODO Not implemented
    //------------------------------------------------------------------------------
    _logger->debug("Use compiler in driver for IR compilation");
    //------------------------------------------------------------------------------
    auto IR = ngraphTransformations::serializeToIR(ngraphFunc);

    // void* graphHandle = apiAdapter->compileIRReturnHandle(IR.xml, IR.weights);
    static const auto blob = apiAdapter->compileIR(IR.xml, IR.weights);
    
    // Get networkDesc (input/output information) from Graph compiler API
    // Emulate getting information from Graph compiler by calling VPUX/MCM Compiler instead and using data from it
    // static const auto networkMeta = apiAdapter->getNetworkMeta(graphHandle);
    static const auto networkMeta = apiAdapter->getNetworkMeta();
   
    auto finish = std::chrono::high_resolution_clock::now();
    _logger->info("|| Timer ||;ZeroCompilerAdapter::compile (ms);\t{}", std::chrono::duration_cast<ms>(finish - start).count());
    
    // return std::make_shared<NetworkDescription>(graphHandle, networkMeta);
    return std::make_shared<NetworkDescription>(blob->data, networkMeta);
}

InferenceEngine::QueryNetworkResult ZeroCompilerAdapter::query(const InferenceEngine::CNNNetwork& /* network */,
                                                               const VPUXConfig& /* config */) {
    THROW_IE_EXCEPTION << "vpux::ZeroCompilerAdapter::query is not implemented.";
    return InferenceEngine::QueryNetworkResult();
}

/** TODO How to handle this case? */
std::shared_ptr<vpux::INetworkDescription> ZeroCompilerAdapter::parse(const std::vector<char>& blob,
                                                                      const VPUXConfig& /* config */,
                                                                      const std::string& /* netName */) {
    static const auto networkMeta = apiAdapter->getNetworkMeta(blob);
    return std::make_shared<NetworkDescription>(blob, networkMeta);
}

INFERENCE_PLUGIN_API(void)
CreateVPUXCompiler(std::shared_ptr<ICompiler>& compiler) {
    compiler = std::make_shared<ZeroCompilerAdapter>();
}

}  // namespace zeroCompilerAdapter
}  // namespace vpux
