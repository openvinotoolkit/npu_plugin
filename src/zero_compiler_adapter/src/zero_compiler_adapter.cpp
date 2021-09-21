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

namespace vpux {
namespace zeroCompilerAdapter {


ZeroCompilerAdapter::ZeroCompilerAdapter() {
    apiAdapter = std::make_shared<VPUXCompilerL0>();
}

ZeroCompilerAdapter::ZeroCompilerAdapter(const ICompiler_Adapter::Ptr& compilerAdapter) : apiAdapter(compilerAdapter) {
}

// TODO How to use inputsInfo, outputsInfo ?
// TODO Fix netName usage
std::shared_ptr<INetworkDescription> ZeroCompilerAdapter::compile(
        const std::shared_ptr<ngraph::Function>& ngraphFunc, const std::string& /*netName*/,
        const InferenceEngine::InputsDataMap& /*inputsInfo*/, const InferenceEngine::OutputsDataMap& /*outputsInfo*/,
        const VPUXConfig& /*config*/) {
    //------------------------------------------------------------------------------
    _logger->debug("Get information about opset versions from compiler");
    //------------------------------------------------------------------------------
    const auto opset = apiAdapter->getSupportedOpset();
    //------------------------------------------------------------------------------
    _logger->debug("Modify network (ngraph) according to supported opset");
    //------------------------------------------------------------------------------
    ngraphTransformations::applyLoweringPasses(ngraphFunc, opset);
    //------------------------------------------------------------------------------
    _logger->debug("Use compiler in driver for IR compilation");
    //------------------------------------------------------------------------------
    auto IR = ngraphTransformations::serializeToIR(ngraphFunc);

    const auto blob = apiAdapter->compileIR(IR.xml, IR.weights);

    // Get networkDesc (input/output information) from Graph compiler API
    // Emulate getting information from Graph compiler by calling VPUX/MCM Compiler instead and using data from it
    const auto networkMeta = apiAdapter->getNetworkMeta(blob);

    return std::make_shared<NetworkDescription>(blob->data, networkMeta);
}

InferenceEngine::QueryNetworkResult ZeroCompilerAdapter::query(const InferenceEngine::CNNNetwork& /* network */,
                                                               const VPUXConfig& /* config */) {
    THROW_IE_EXCEPTION << "vpux::ZeroCompilerAdapter::query is not implemented.";
    return InferenceEngine::QueryNetworkResult();
}

/** TODO How to handle this case? */
std::shared_ptr<vpux::INetworkDescription> ZeroCompilerAdapter::parse(const std::vector<char>& /* network */,
                                                                      const VPUXConfig& /* config */,
                                                                      const std::string& /* netName */) {
    THROW_IE_EXCEPTION << "vpux::ZeroCompilerAdapter::parse is not implemented.";
    return std::shared_ptr<vpux::INetworkDescription>();
}


INFERENCE_PLUGIN_API(void)
CreateVPUXCompiler(std::shared_ptr<ICompiler>& compiler) {
    compiler = std::make_shared<ZeroCompilerAdapter>();
}

}  // namespace zeroCompilerAdapter
}  // namespace vpux
