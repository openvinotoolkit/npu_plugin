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
namespace zeroCompilerAdapter {

struct IR {
    std::vector<char> xml;
    std::vector<char> weights;
};

struct Opset {
    size_t version;
};

// TODO Extend with unique id
// TODO Class?
// TODO Blob + meta = NetworkDesc. If getBlob() + getMeta() will be one call to compiler, we can get rid of this struct.
struct Blob {
    Blob(const std::vector<char>& data);

    using Ptr = std::shared_ptr<Blob>;
    std::vector<char> data;
};

/**
 * @brief Interface for compiler. Contain specific function which should be implemented by compiler.
 */
class ICompiler_Adapter {
public:
    using Ptr = std::shared_ptr<ICompiler_Adapter>;
    /**
     * @brief Get opset supported by compiler
     */
    virtual Opset getSupportedOpset() = 0;

    /**
     * @brief compile NGraph and return blob file
     * @return compiled graph (blob)
     */
    virtual Blob::Ptr compileIR(std::vector<char>& xml, std::vector<char>& weights) = 0;

    // TODO Should it be one call with compileIR() to avoid double parsing?
    // TODO In general we can pass only blob identification to compiler, but it this case, after each compileIR()
    //  call it should store compiled graph (which can be a problem, if we will compile 10 vgg networks).
    /**
     * @brief Get all meta information about graph (name, network inputs/outputs, device inputs/outputs)
     * @details ImportNetwork case
     */
    virtual std::tuple<const std::string, const DataMap, const DataMap, const DataMap, const DataMap> getNetworkMeta(
            const Blob::Ptr compiledNetwork) = 0;

    // TODO Do we need such function?
    /**
     * @brief Get meta information about device inputs/outputs
     * @details LoadNetwork case. Networks inputs/outputs and graph name should be already available
     */
    virtual std::tuple<const DataMap, const DataMap> getDeviceNetworkMeta(const Blob::Ptr compiledNetwork) = 0;
};
}  // namespace zeroCompilerAdapter
}  // namespace vpux
