//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <cstddef>
#include <set>

#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>

#include "vpux/utils/IE/config.hpp"
#include "vpux/utils/core/quant_params.hpp"

namespace vpux {

class ICompiler;

/**
 * @brief A helper structure used for storing the metadata found within the I/O nodes.
 * @details The "legacyName" attribute holds the name most commonly used as map key for multiple structures.
 * This value also corresponds to the identifier used by the OpenVINO 1.0 API.
 *
 * "originalShape" corresponds to the shape registered in the graph, while "transposedShape" holds the shape obtained
 * upon applying a transposition corresponding to the legacy layout value. Use the "transposedShape" one if not sure
 * which one you need.
 */
struct IONodeDescriptor {
    std::string legacyName;
    std::string currentNodeName;
    std::unordered_set<std::string> outputTensorNames;
    ov::element::Type_t precision;
    ov::PartialShape originalShape;
    ov::PartialShape transposedShape;
};

/**
 * @brief A helper map to represent descriptions for inputs and outputs
 * of a network
 */
using IONodeDescriptorMap = std::unordered_map<std::string, IONodeDescriptor>;

/**
 * @brief A helper type to represent vectors of OV 2.0 nodes
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
     * @return The name of the network
     */
    const std::string& getName() const {
        return _name;
    }

    /**
     * @return The input names which can be used as keys for the parameter descriptors map. The order of the names
     * corresponds to the order in which the compiler received the parameters.
     */
    const std::vector<std::string>& getInputNames() const {
        return _inputNames;
    }

    /**
     * @return The output names which can be used as keys for the result descriptors map. The order of the names
     * corresponds to the order in which the compiler received the parameters.
     */
    const std::vector<std::string>& getOutputNames() const {
        return _outputNames;
    }

    /**
     * @return The state names which can be used as keys for the state descriptors map. The order of the names
     * corresponds to the order in which the compiler received the parameters.
     */
    const std::vector<std::string>& getStateNames() const {
        return _stateNames;
    }

    /**
     * @return Structure describing the parameter nodes (i.e. inputs) of the current network.
     */
    const IONodeDescriptorMap& getParameterDescriptors() const {
        return _parameters;
    }

    /**
     * @return Structure describing the result nodes (i.e. outputs) of the current network.
     */
    const IONodeDescriptorMap& getResultDescriptors() const {
        return _results;
    }

    /**
     * @return Structure describing tensors corresponding to the states of the current network.
     */
    const IONodeDescriptorMap& getStateDescriptors() const {
        return _states;
    }

    /**
     * @return Structure describing tensors corresponding to the profiling outputs of the current network.
     */
    const IONodeDescriptorMap& getProfilingOutputDescriptors() const {
        return _profilingOutputs;
    }

    /**
     * @return A map indicating the order in which each input was found inside the compiled model.
     */
    const std::unordered_map<std::string, size_t>& getInputOrder() const {
        return _inputOrder;
    }

    /**
     * @return A map indicating the order in which each output was found inside the compiled model.
     */
    const std::unordered_map<std::string, size_t>& getOutputOrder() const {
        return _outputOrder;
    }

    // TODO Remove interface returning std::vector<char>.
    /**
     * @deprecated Return type should follow the function below.
     * The name itself can be reused once the old return type is dropped.
     */
    const std::vector<char>& getCompiledNetwork() const {
        return _compiledNetwork;
    }

    /**
     * @return A raw pointer to the compiled model
     */
    const void* getNetworkModel() const {
        return _compiledNetwork.data();
    }

    /**
     * @return The size of the compiled model in bytes
     */
    std::size_t getNetworkModelSize() const {
        return _compiledNetwork.size();
    }

    /**
     * @return The number of streams, that can be executed in parallel with current network configuration.
     */
    int getNumStreams() const {
        return _numStreams;
    }

protected:
    virtual ~INetworkDescription() = default;

    std::vector<char> _compiledNetwork;

    std::string _name;
    std::vector<std::string> _inputNames;
    std::vector<std::string> _outputNames;
    std::vector<std::string> _stateNames;

    IONodeDescriptorMap _parameters;
    IONodeDescriptorMap _results;
    IONodeDescriptorMap _states;
    IONodeDescriptorMap _profilingOutputs;

    // Required only when using the IMD backend
    std::unordered_map<std::string, size_t> _inputOrder;
    std::unordered_map<std::string, size_t> _outputOrder;

    int _numStreams = 1;
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

    const std::vector<std::string>& getInputNames() const {
        return _impl->getInputNames();
    }

    const std::vector<std::string>& getOutputNames() const {
        return _impl->getOutputNames();
    }

    const std::vector<std::string>& getStateNames() const {
        return _impl->getStateNames();
    }

    const IONodeDescriptorMap& getParameterDescriptors() const {
        return _impl->getParameterDescriptors();
    }

    const IONodeDescriptorMap& getResultDescriptors() const {
        return _impl->getResultDescriptors();
    }

    const IONodeDescriptorMap& getStateDescriptors() const {
        return _impl->getStateDescriptors();
    }

    const IONodeDescriptorMap& getProfilingOutputDescriptors() const {
        return _impl->getProfilingOutputDescriptors();
    }

    const std::unordered_map<std::string, size_t>& getInputOrder() const {
        return _impl->getInputOrder();
    }

    const std::unordered_map<std::string, size_t>& getOutputOrder() const {
        return _impl->getOutputOrder();
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
     * @brief Transforms a network from the OpenVINO model representation to a format executable
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
        return std::make_shared<NetworkDescription>(_impl->compile(model, networkName, config), _so);
    }

    ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model, const Config& config) {
        return _impl->query(model, config);
    }

    std::shared_ptr<vpux::NetworkDescription> parse(const std::vector<char>& network, const Config& config) {
        return std::make_shared<NetworkDescription>(_impl->parse(network, config, ""), _so);
    }

    std::shared_ptr<vpux::NetworkDescription> parse(const std::string& filename, const Config& config) {
        return std::make_shared<NetworkDescription>(_impl->parse(filename, config), _so);
    }

    std::shared_ptr<vpux::NetworkDescription> parse(std::istream& stream, const Config& config,
                                                    const std::string& graphName) {
        return std::make_shared<NetworkDescription>(_impl->parse(stream, config, graphName), _so);
    }

private:
    std::shared_ptr<ICompiler> _impl;

    // Keep pointer to `_so` to avoid shared library unloading prior destruction of the `_impl` object.
    std::shared_ptr<void> _so;
};

namespace helpers {

/**
 * @brief Creates node objects in the format used by OpenVINO.
 * @param nodeDescriptors The metadata which shall be used for creating the nodes.
 * @param names A vector of strings used for extracting the nodes' metadata in the order in which the parameters were
 * extracted from the model before compilation.
 * @param isResult Indicates wheter the nodes should be handled as parameter nodes (i.e. inputs) or result nodes (i.e.
 * outputs).
 * @return A vector of ordered nodes in the format used by OpenVINO.
 */
vpux::OVNodes nodeDescriptorsIntoNodes(const IONodeDescriptorMap& nodeDescriptors,
                                       const std::vector<std::string>& names, const bool isResult);

}  // namespace helpers

}  // namespace vpux
