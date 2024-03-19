//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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
    virtual uint32_t getSupportedOpset() = 0;

    /**
     * @brief Get query result for current network
     */
    virtual std::unordered_set<std::string> getQueryResult(const std::vector<char>& xml,
                                                           const std::vector<char>& weights,
                                                           const vpux::Config& config) = 0;

    /**
     * @brief Sends the serialized model and its I/O metadata to the driver for compilation.
     * @return The compiled model descriptor corresponding to the previously given network.
     */
    virtual std::shared_ptr<INetworkDescription> compileIR(const std::shared_ptr<ov::Model> model,
                                                           const std::string& graphName, const std::vector<char>& xml,
                                                           const std::vector<char>& weights,
                                                           const vpux::Config& config) = 0;
    virtual std::shared_ptr<INetworkDescription> parseBlob(const std::string& graphName, const std::vector<char>& blob,
                                                           const vpux::Config& config) = 0;
};
}  // namespace driverCompilerAdapter
}  // namespace vpux
