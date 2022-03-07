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

#pragma once
#include "vpux_compiler.hpp"

namespace vpux {
namespace driverCompilerAdapter {

struct IR {
    const std::vector<char> xml;
    const std::vector<char> weights;
};

/**
 * @brief Interface for external compiler
 * @details Isolate external API calls from general logic
 */
class IExternalCompiler {
public:
    virtual ~IExternalCompiler() = default;

    using Ptr = std::shared_ptr<IExternalCompiler>;

    /**
     * @brief Get opset supported by compiler
     */
    virtual size_t getSupportedOpset() = 0;

    /**
     * @brief compile NGraph and return blob file
     * @return compiled graph (blob)
     */
    virtual std::shared_ptr<INetworkDescription> compileIR(const std::string& graphName, const std::vector<char>& xml,
                                                           const std::vector<char>& weights,
                                                           const InferenceEngine::InputsDataMap& inputsInfo,
                                                           const InferenceEngine::OutputsDataMap& outputsInfo,
                                                           const vpux::Config& config) = 0;
    virtual std::shared_ptr<INetworkDescription> parseBlob(const std::string& graphName, const std::vector<char>& blob,
                                                           const vpux::Config& config) = 0;
};
}  // namespace driverCompilerAdapter
}  // namespace vpux
