//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include <cstddef>
#include <set>

#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include <ie_icnn_network.hpp>
#include <ie_input_info.hpp>
#include <ie_remote_context.hpp>

#include "vpux/utils/IE/config.hpp"
#include "vpux/utils/core/preprocessing.hpp"
#include "vpux/utils/core/quant_params.hpp"

namespace vpux {

class ICompiler;

/**
 * @brief A helper map to represent descriptions for inputs and outputs
 * of a network
 */
using DataMap = std::map<std::string, InferenceEngine::DataPtr>;

/**
 * @brief A helper type to keep OV 2.0 node raw data
 *
 */
struct OVRawNode {
    const std::string friendlyName;
    const ov::element::Type_t type;
    const ov::Shape shape;
    const std::unordered_set<std::string> tensorNames;
    const std::string inputName;
    const bool isResult;
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
     * @brief Returns a map with information about network inputs. The inputs
     * shall be obtained from the network
     * @return Constant reference to an internally held DataMap object
     */
    virtual const DataMap& getInputsInfo() const = 0;

    /**
     * @brief Returns a map with information about network outputs. The outputs
     * shall be obtained from the network
     * @return Constant reference to an internally held DataMap object
     */
    virtual const DataMap& getOutputsInfo() const = 0;

    /**
     * @brief Returns a map with information about network inputs which
     * will be executed. The inputs are defined by a compiler and can be different
     * from the original inputs due to compiler restrictions or optimizations
     * @return Constant reference to an internally held DataMap object
     */
    virtual const DataMap& getDeviceInputsInfo() const = 0;

    /**
     * @brief Returns a map with information about network outputs which
     * will be executed. The outputs are defined by a compiler and can be different
     * from the original outputs due to compiler restrictions or optimizations
     * @return Constant reference to an internally held DataMap object
     */
    virtual const DataMap& getDeviceOutputsInfo() const = 0;

    /**
     * @brief Returns a map with information about profiling outputs.
     * The outputs are defined by a compiler.
     * @return Constant reference to an internally held DataMap object
     */
    virtual const DataMap& getDeviceProfilingOutputsInfo() const = 0;

    virtual const std::vector<vpux::OVRawNode>& getOVParameters() const = 0;
    virtual const std::vector<vpux::OVRawNode>& getOVResults() const = 0;

    /**
     * @brief Returns a map with information about quantization parameters
     * @return Constant reference to an internally held QuantizationParamMap object
     */
    virtual const vpux::QuantizationParamMap& getQuantParamsInfo() const = 0;

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
     * @brief Returns a map with information about preprocess inputs.
     * @return Constant reference to an internally held map containing information about preprocess for inputs
     */
    const std::unordered_map<std::string, InferenceEngine::PreProcessInfo>& getPreprocessInfo() const {
        return _iePreprocessInfo;
    }

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

protected:
    std::unordered_map<std::string, InferenceEngine::PreProcessInfo> _iePreprocessInfo;
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

    // Destructor preserves unload order of implementation object and reference to library.
    // To preserve destruction order inside default generated assignment operator we store `_impl` before `_so`.
    // And use destructor to remove implementation object before reference to library explicitly.
    ~NetworkDescription() {
        _impl = {};
    }

    const std::string& getName() const {
        return _impl->getName();
    }
    const DataMap& getInputsInfo() const {
        return _impl->getInputsInfo();
    }
    const DataMap& getOutputsInfo() const {
        return _impl->getOutputsInfo();
    }
    const DataMap& getDeviceInputsInfo() const {
        return _impl->getDeviceInputsInfo();
    }
    const DataMap& getDeviceOutputsInfo() const {
        return _impl->getDeviceOutputsInfo();
    }
    const DataMap& getDeviceProfilingOutputsInfo() const {
        return _impl->getDeviceProfilingOutputsInfo();
    }
    const std::vector<OVRawNode>& getOVParameters() const {
        return _impl->getOVParameters();
    }
    const std::vector<OVRawNode>& getOVResults() const {
        return _impl->getOVResults();
    }
    const vpux::QuantizationParamMap& getQuantParamsInfo() const {
        return _impl->getQuantParamsInfo();
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
     * @param func a shared pointer to ngraph function representing the model
     * @param netName a reference to the string describing network name
     *        to be used for creating network description
     * @param inputsInfo a reference to map describing inputs of the network
     * @param outputsInfo a reference to map describing outputs of the network
     * @param config a reference to VPUXConfig containing plugin config options
     *        including config options related to compilation
     * @return a shared pointer on an object implementing INetworkDescription interface
     */
    virtual std::shared_ptr<INetworkDescription> compile(const std::shared_ptr<ngraph::Function>& func,
                                                         const std::string& netName,
                                                         const InferenceEngine::InputsDataMap& inputsInfo,
                                                         const InferenceEngine::OutputsDataMap& outputsInfo,
                                                         const Config& config) = 0;

    /**
     * @brief Returns information about supported layers of the network passed
     * @param network a const reference to CNNNetwork
     * @param config a reference to VPUXConfig containing plugin config options
     *        including config options related to compilation
     * @returns QueryNetworkResult structure with information about supported layers
     */
    virtual InferenceEngine::QueryNetworkResult query(const InferenceEngine::CNNNetwork& network,
                                                      const Config& config) = 0;

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

    // Destructor preserves unload order of implementation object and reference to library.
    // To preserve destruction order inside default generated assignment operator we store `_impl` before `_so`.
    // And use destructor to remove implementation object before reference to library explicitly.
    ~Compiler() {
        _impl = {};
    }

    std::shared_ptr<vpux::NetworkDescription> compile(const std::shared_ptr<ngraph::Function>& func,
                                                      const std::string& netName,
                                                      const InferenceEngine::InputsDataMap& inputsInfo,
                                                      const InferenceEngine::OutputsDataMap& outputsInfo,
                                                      const Config& config) {
        return std::make_shared<NetworkDescription>(_impl->compile(func, netName, inputsInfo, outputsInfo, config),
                                                    _impl);
    }

    InferenceEngine::QueryNetworkResult query(const InferenceEngine::CNNNetwork& network, const Config& config) {
        return _impl->query(network, config);
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
InferenceEngine::InputsDataMap dataMapIntoInputsDataMap(const vpux::DataMap& dataMap);
InferenceEngine::OutputsDataMap dataMapIntoOutputsDataMap(const vpux::DataMap& dataMap);
vpux::OVNodes ovRawNodesIntoOVNodes(const std::vector<vpux::OVRawNode>& rawNodes, const bool isResult);
}  // namespace helpers

}  // namespace vpux
