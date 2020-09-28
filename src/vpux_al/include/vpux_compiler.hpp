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
#include <details/ie_so_loader.h>

#include <details/ie_so_pointer.hpp>
#include <ie_icnn_network.hpp>
#include <ie_input_info.hpp>
#include <ie_remote_context.hpp>
#include <set>
#include <vpux_config.hpp>

namespace vpux {

///////////////////////////////////// INetworkDescription /////////////////////////////////////////
using DataMap = std::map<std::string, InferenceEngine::DataPtr>;
class INetworkDescription : public InferenceEngine::details::IRelease {
public:
    using Ptr = std::shared_ptr<INetworkDescription>;
    using CPtr = std::shared_ptr<const INetworkDescription>;

    virtual const std::string& getName() const = 0;
    virtual const DataMap& getInputsInfo() const = 0;
    virtual const DataMap& getOutputsInfo() const = 0;
    virtual const DataMap& getDeviceInputsInfo() const = 0;
    virtual const DataMap& getDeviceOutputsInfo() const = 0;
    virtual const std::vector<char>& getCompiledNetwork() const = 0;
    virtual ~INetworkDescription() = default;
    virtual void Release() noexcept override { delete this; }
};

///////////////////////////////////// NetworkDescription //////////////////////////////////////////
class NetworkDescription final {
public:
    using Ptr = std::shared_ptr<NetworkDescription>;
    using CPtr = std::shared_ptr<const NetworkDescription>;

    NetworkDescription(INetworkDescription::Ptr actual, InferenceEngine::details::SharedObjectLoader::Ptr plg = {});

    const std::string& getName() const { return _actual->getName(); }
    const DataMap& getInputsInfo() const { return _actual->getInputsInfo(); }
    const DataMap& getOutputsInfo() const { return _actual->getOutputsInfo(); }
    const DataMap& getDeviceInputsInfo() const { return _actual->getDeviceInputsInfo(); }
    const DataMap& getDeviceOutputsInfo() const { return _actual->getDeviceOutputsInfo(); }
    const std::vector<char>& getCompiledNetwork() const { return _actual->getCompiledNetwork(); }

    ~NetworkDescription() { _actual = nullptr; }

private:
    INetworkDescription::Ptr _actual = nullptr;
    // NB: NetworkDescription is created by Compiler which is object from shared library,
    // so it has to keep pointer to this lib in case Compiler lifecycle less than objects which created by it
    InferenceEngine::details::SharedObjectLoader::Ptr _plg;
};

//////////////////////////////////////////ICompiler ///////////////////////////////////////////////
class ICompiler : public InferenceEngine::details::IRelease {
public:
    using Ptr = std::shared_ptr<ICompiler>;
    using CPtr = std::shared_ptr<const ICompiler>;

    virtual std::shared_ptr<INetworkDescription> compile(
        InferenceEngine::ICNNNetwork& network, const VPUXConfig& config = {}) = 0;

    virtual std::shared_ptr<INetworkDescription> compile(const std::shared_ptr<ngraph::Function>& func,
        const std::string& netName, const InferenceEngine::InputsDataMap& inputsInfo,
        const InferenceEngine::OutputsDataMap& outputsInfo, const VPUXConfig& config = {}) = 0;

    virtual std::shared_ptr<vpux::INetworkDescription> parse(
        const std::vector<char>& network, const VPUXConfig& config = {}, const std::string& graphName = "") = 0;

    virtual std::shared_ptr<vpux::INetworkDescription> parse(
        const std::string& filename, const VPUXConfig& config = {});
    virtual std::shared_ptr<vpux::INetworkDescription> parse(
        std::istream& stream, const VPUXConfig& config = {}, const std::string& graphName = "");

    virtual std::set<std::string> getSupportedLayers(InferenceEngine::ICNNNetwork& network) = 0;

    virtual std::unordered_set<std::string> getSupportedOptions() { return {}; };

    virtual void Release() noexcept override { delete this; }

    virtual ~ICompiler() = default;
};

enum class CompilerType {
    MCMCompiler,
};

//////////////////////////////////////////Compiler ////////////////////////////////////////////////
class Compiler final {
public:
    using Ptr = std::shared_ptr<Compiler>;
    using CPtr = std::shared_ptr<const Compiler>;

    // TODO: In future it can be replaced from CompilerType to the path to the compiler lib
    // to avoid adding new CompilerType with new compiler
    static Ptr create(CompilerType t);

    Compiler(std::string libpath): _actual(std::move(libpath)){};

    std::shared_ptr<vpux::NetworkDescription> compile(
        InferenceEngine::ICNNNetwork& network, const vpux::VPUXConfig& config = {}) {
        return std::make_shared<NetworkDescription>(_actual->compile(network, config), _actual);
    }

    std::shared_ptr<vpux::NetworkDescription> compile(const std::shared_ptr<ngraph::Function>& func,
        const std::string& netName, const InferenceEngine::InputsDataMap& inputsInfo,
        const InferenceEngine::OutputsDataMap& outputsInfo, const vpux::VPUXConfig& config = {}) {
        return std::make_shared<NetworkDescription>(
            _actual->compile(func, netName, inputsInfo, outputsInfo, config), _actual);
    }

    std::shared_ptr<vpux::NetworkDescription> parse(
        const std::vector<char>& network, const vpux::VPUXConfig& config = {}) {
        return std::make_shared<NetworkDescription>(_actual->parse(network, config), _actual);
    }

    std::shared_ptr<vpux::NetworkDescription> parse(const std::string& filename, const VPUXConfig& config = {}) {
        return std::make_shared<NetworkDescription>(_actual->parse(filename, config), _actual);
    }

    std::shared_ptr<vpux::NetworkDescription> parse(std::istream& stream, const VPUXConfig& config = {}) {
        return std::make_shared<NetworkDescription>(_actual->parse(stream, config), _actual);
    }

    std::set<std::string> getSupportedLayers(InferenceEngine::ICNNNetwork& network) {
        return _actual->getSupportedLayers(network);
    }

    std::unordered_set<std::string> getSupportedOptions() { return _actual->getSupportedOptions(); };

private:
    using CompilerPluginPtr = InferenceEngine::details::SOPointer<vpux::ICompiler>;
    CompilerPluginPtr _actual;
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
