//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once
#include <details/ie_so_pointer.hpp>
#include <ie_icnn_network.hpp>
#include <ie_input_info.hpp>
#include <ie_remote_context.hpp>
#include <set>
#include <vpux_config.hpp>

namespace vpux {

using DataMap = std::map<std::string, InferenceEngine::DataPtr>;
class NetworkDescription {
public:
    using Ptr = std::shared_ptr<NetworkDescription>;
    using CPtr = std::shared_ptr<const NetworkDescription>;

    virtual const std::string& getName() const = 0;
    virtual const DataMap& getInputsInfo() const = 0;
    virtual const DataMap& getOutputsInfo() const = 0;
    virtual const DataMap& getDeviceInputsInfo() const = 0;
    virtual const DataMap& getDeviceOutputsInfo() const = 0;

    virtual const std::vector<char>& getCompiledNetwork() const = 0;

    virtual ~NetworkDescription() = default;
};

enum class CompilerType {
    MCMCompiler,
};

class ICompiler : public InferenceEngine::details::IRelease {
public:
    using Ptr = std::shared_ptr<ICompiler>;
    // TODO: In future it can be replaced from CompilerType to the path to the compiler lib
    // to avoid adding new CompilerType with new compiler
    static std::shared_ptr<ICompiler> create(CompilerType t);

    virtual std::shared_ptr<NetworkDescription> compile(
        InferenceEngine::ICNNNetwork& network, const VPUXConfig& config = {}) = 0;

    virtual std::shared_ptr<NetworkDescription> compile(const std::shared_ptr<ngraph::Function>& func,
        const std::string& netName, const InferenceEngine::InputsDataMap& inputsInfo,
        const InferenceEngine::OutputsDataMap& outputsInfo, const VPUXConfig& config = {}) = 0;

    virtual std::shared_ptr<vpux::NetworkDescription> parse(
        const std::vector<char>& network, const VPUXConfig& config = {}, const std::string& graphName = "") = 0;

    virtual std::shared_ptr<vpux::NetworkDescription> parse(const std::string& filename, const VPUXConfig& config = {});
    virtual std::shared_ptr<vpux::NetworkDescription> parse(
        std::istream& stream, const VPUXConfig& config = {}, const std::string& graphName = "");

    virtual std::set<std::string> getSupportedLayers(InferenceEngine::ICNNNetwork& network) = 0;

    virtual void Release() noexcept override { delete this; }
};

namespace helpers {
InferenceEngine::InputsDataMap dataMapIntoInputsDataMap(const vpux::DataMap& dataMap);

InferenceEngine::OutputsDataMap dataMapIntoOutputsDataMap(const vpux::DataMap& dataMap);
}  // namespace helpers

}  // namespace vpux

namespace InferenceEngine {
namespace details {
template <>
class SOCreatorTrait<vpux::ICompiler> {
public:
    static constexpr auto name = "CreateVPUXCompiler";
};
}  // namespace details
}  // namespace InferenceEngine
