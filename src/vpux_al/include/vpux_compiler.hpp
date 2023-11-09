//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <cstddef>
#include <set>

#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include <ie_icnn_network.hpp>
#include <ie_input_info.hpp>
#include <ie_remote_context.hpp>

#include "vpux/utils/IE/config.hpp"
#include "vpux/utils/core/quant_params.hpp"

namespace vpux {

class ICompiler;

/**
 * @brief A helper vector of pairs to represent descriptions for inputs and outputs
 * of a network
 */
using NetworkIOVector = std::vector<std::pair<std::string, InferenceEngine::DataPtr>>;

/**
 * @brief A helper type to keep OV 2.0 node raw data
 *
 */
struct OVRawNode {
    std::string friendlyName;
    ov::element::Type_t type;
    ov::Shape shape;
    std::unordered_set<std::string> tensorNames;
    std::string inputName;
};

/**
 * @brief A helper type to represent vectors of OV 2.0 nodes
 *
 */
using OVNodes = std::vector<std::shared_ptr<const ov::Node>>;

///////////////////////////////////// INetworkDescription /////////////////////////////////////////
/**
 * @interface INetworkDescription
 * @brief The interface to be implemented by a concrete compiler
 * to provide such information about a network as description of inputs and outputs,
 * name and compiled network in a format executable by device
 */
class INetworkDescription : public std::enable_shared_from_this<INetworkDescription> {
public:
    /**
     * @brief A shared pointer to INetworkDescription interface
     */
    using Ptr = std::shared_ptr<INetworkDescription>;
    /**
     * @brief A const shared pointer to INetworkDescription interface
     */
    using CPtr = std::shared_ptr<const INetworkDescription>;

    /**
     * @brief Returns name of a network
     * @return Network name
     */
    virtual const std::string& getName() const = 0;

    /**
     * @brief Returns a vector of pairs with information about network inputs which
     * will be executed. The inputs are defined by a compiler and can be different
     * from the original inputs due to compiler restrictions or optimizations
     * @return Constant reference to an internally held NetworkIOVector object
     */
    virtual const NetworkIOVector& getDeviceInputsInfo() const = 0;

    /**
     * @brief Returns a vector of pairs with information about network outputs which
     * will be executed. The outputs are defined by a compiler and can be different
     * from the original outputs due to compiler restrictions or optimizations
     * @return Constant reference to an internally held NetworkIOVector object
     */
    virtual const NetworkIOVector& getDeviceOutputsInfo() const = 0;

    /**
     * @brief Returns a vector of pairs with information about profiling outputs.
     * The outputs are defined by a compiler.
     * @return Constant reference to an internally held NetworkIOVector object
     */
    virtual const NetworkIOVector& getDeviceProfilingOutputsInfo() const = 0;

    virtual const std::vector<vpux::OVRawNode>& getOVParameters() const = 0;
    virtual const std::vector<vpux::OVRawNode>& getOVResults() const = 0;

    // TODO Remove interface returning std::vector<char>.
    /**
     * @deprecated Return type should follow the function below.
     * The name itself can be reused once the old return type is dropped.
     */
    virtual const std::vector<char>& getCompiledNetwork() const = 0;

    /**
     * @brief Returns a raw pointer to the compiled model
     * @return Pointer to void
     */
    virtual const void* getNetworkModel() const = 0;

    /**
     * @brief Returns size of the compiled model in bytes
     * @return size in bytes
     */
    virtual std::size_t getNetworkModelSize() const = 0;

    /**
     * @brief Get the number of streams, that can be executed in parallel
     * with current network configuration.
     */
    virtual int getNumStreams() const = 0;

    virtual ~INetworkDescription() = default;
};

/**
 * @brief NetworkDescription is a wrapper around INetworkDescription and
 * duplicates all its methods.
 * NetworkDescription is created by Compiler which is object from shared
 * library, so it has to keep pointer to this lib in case Compiler lifecycle
 * less than objects which created by it
 */
class NetworkDescription final {
public:
    using Ptr = std::shared_ptr<NetworkDescription>;
    using CPtr = std::shared_ptr<const NetworkDescription>;

    NetworkDescription(INetworkDescription::Ptr impl, const std::shared_ptr<void>& so = {});
    NetworkDescription(const NetworkDescription&) = default;
    NetworkDescription& operator=(const NetworkDescription&) = default;

    // Destructor preserves unload order of implementation object and reference to library.
    // To preserve destruction order inside default generated assignment operator we store `_impl` before `_so`.
    // And use destructor to remove implementation object before reference to library explicitly.
    ~NetworkDescription() {
        _impl = {};
    }

    const std::string& getName() const {
        return _impl->getName();
    }
    const NetworkIOVector& getDeviceInputsInfo() const {
        return _impl->getDeviceInputsInfo();
    }
    const NetworkIOVector& getDeviceOutputsInfo() const {
        return _impl->getDeviceOutputsInfo();
    }
    const NetworkIOVector& getDeviceProfilingOutputsInfo() const {
        return _impl->getDeviceProfilingOutputsInfo();
    }
    const std::vector<OVRawNode>& getOVParameters() const {
        return _impl->getOVParameters();
    }
    const std::vector<OVRawNode>& getOVResults() const {
        return _impl->getOVResults();
    }
    const std::vector<char>& getCompiledNetwork() const {
        return _impl->getCompiledNetwork();
    }
    const void* getNetworkModel() const {
        return _impl->getNetworkModel();
    }
    std::size_t getNetworkModelSize() const {
        return _impl->getNetworkModelSize();
    }

    int getNumStreams() const {
        return _impl->getNumStreams();
    }

private:
    INetworkDescription::Ptr _impl;

    // Keep pointer to `_so` to avoid shared library unloading prior destruction of the `_impl` object.
    std::shared_ptr<void> _so;
};

/**
 * @interface ICompiler
 * @brief An interface to be implemented by a concrete compiler to provide
 * methods for preparing a network for execution on a VPU device
 */
class ICompiler : public std::enable_shared_from_this<ICompiler> {
public:
    using Ptr = std::shared_ptr<ICompiler>;
    using CPtr = std::shared_ptr<const ICompiler>;

    /**
     * @brief Transforms a network from ngraph representation to a format executable
     * by a VPU device
     * @param model a shared pointer to the OpenVINO model to be compiled
     * @param netName a reference to the string describing network name
     *        to be used for creating network description
     * @param config a reference to VPUXConfig containing plugin config options
     *        including config options related to compilation
     * @return a shared pointer on an object implementing INetworkDescription interface
     */
    virtual std::shared_ptr<INetworkDescription> compile(std::shared_ptr<ov::Model>& model,
                                                         const std::string& networkName, const Config& config) = 0;

    /**
     * @brief Returns information about supported layers of the network passed
     * @param model The model to be queried
     * @param config A reference to VPUXConfig containing plugin config options
     *        including config options related to compilation
     * @returns SupportedOpsMap structure with information about supported layers
     */
    virtual ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model, const Config& config) = 0;

    /**
     * @brief Parses already compiled network to extract meta information:
     *        inputs and outputs descriptions
     * @param network compiled network represented as a vector of char
     * @param config a reference to VPUXConfig containing plugin config options
     *        Note: compilation options will be ignored,
     *        since the network is already compiled
     * @param netName a reference to the string describing network name
     *        to be used for creating network description
     * @return a shared pointer on an object implementing INetworkDescription interface
     */
    virtual std::shared_ptr<vpux::INetworkDescription> parse(const std::vector<char>& network, const Config& config,
                                                             const std::string& netName) = 0;

    virtual std::shared_ptr<vpux::INetworkDescription> parse(const std::string& filename, const Config& config);
    virtual std::shared_ptr<vpux::INetworkDescription> parse(std::istream& stream, const Config& config,
                                                             const std::string& netName);

protected:
    ~ICompiler() = default;
};

//////////////////////////////////////////Compiler ////////////////////////////////////////////////
class Compiler final {
public:
    using Ptr = std::shared_ptr<Compiler>;
    using CPtr = std::shared_ptr<const Compiler>;

    static Ptr create(const Config& config);

#ifdef OPENVINO_STATIC_LIBRARY
    Compiler(const std::shared_ptr<ICompiler>& compiler): _impl(compiler){};
#else
    Compiler(const std::string& libpath);
#endif
    Compiler(const Compiler&) = delete;
    Compiler& operator=(const Compiler&) = delete;

    // Destructor preserves unload order of implementation object and reference to library.
    // To preserve destruction order inside default generated assignment operator we store `_impl` before `_so`.
    // And use destructor to remove implementation object before reference to library explicitly.
    ~Compiler() {
        _impl = {};
    }

    std::shared_ptr<vpux::NetworkDescription> compile(std::shared_ptr<ov::Model>& model, const std::string& networkName,
                                                      const Config& config) {
        return std::make_shared<NetworkDescription>(_impl->compile(model, networkName, config), _impl);
    }

    ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model, const Config& config) {
        return _impl->query(model, config);
    }

    std::shared_ptr<vpux::NetworkDescription> parse(const std::vector<char>& network, const Config& config) {
        return std::make_shared<NetworkDescription>(_impl->parse(network, config, ""), _impl);
    }

    std::shared_ptr<vpux::NetworkDescription> parse(const std::string& filename, const Config& config) {
        return std::make_shared<NetworkDescription>(_impl->parse(filename, config), _impl);
    }

    std::shared_ptr<vpux::NetworkDescription> parse(std::istream& stream, const Config& config,
                                                    const std::string& graphName) {
        return std::make_shared<NetworkDescription>(_impl->parse(stream, config, graphName), _impl);
    }

private:
    std::shared_ptr<ICompiler> _impl;

    // Keep pointer to `_so` to avoid shared library unloading prior destruction of the `_impl` object.
    std::shared_ptr<void> _so;
};

namespace helpers {
InferenceEngine::InputsDataMap networkIOVectorIntoInputsDataMap(const vpux::NetworkIOVector& ioVector);
InferenceEngine::OutputsDataMap networkIOVectorIntoOutputsDataMap(const vpux::NetworkIOVector& ioVector);
vpux::OVNodes ovRawNodesIntoOVNodes(const std::vector<vpux::OVRawNode>& rawNodes, const bool isResult);
}  // namespace helpers

}  // namespace vpux
