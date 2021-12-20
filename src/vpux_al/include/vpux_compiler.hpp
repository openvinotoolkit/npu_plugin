//
// Copyright 2020 Intel Corporation.
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

#include <cstddef>
#include <set>

#include <details/ie_so_loader.h>
#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include <details/ie_so_pointer.hpp>
#include <ie_icnn_network.hpp>
#include <ie_input_info.hpp>
#include <ie_remote_context.hpp>

#include "vpux/utils/IE/config.hpp"
#include "vpux/utils/core/preprocessing.hpp"
#include "vpux/utils/core/quant_params.hpp"

namespace vpux {

class ICompiler;
#ifdef OPENVINO_STATIC_LIBRARY
using CompilerPluginPtr = std::shared_ptr<ICompiler>;
#else
IE_SUPPRESS_DEPRECATED_START
using CompilerPluginPtr = InferenceEngine::details::SOPointer<ICompiler>;
IE_SUPPRESS_DEPRECATED_END
#endif

/**
 * @brief A helper map to represent descriptions for inputs and outputs
 * of a network
 */
using DataMap = std::map<std::string, InferenceEngine::DataPtr>;

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

    NetworkDescription(INetworkDescription::Ptr actual, const CompilerPluginPtr& plg = {});

    const std::string& getName() const {
        return _actual->getName();
    }
    const DataMap& getInputsInfo() const {
        return _actual->getInputsInfo();
    }
    const DataMap& getOutputsInfo() const {
        return _actual->getOutputsInfo();
    }
    const DataMap& getDeviceInputsInfo() const {
        return _actual->getDeviceInputsInfo();
    }
    const DataMap& getDeviceOutputsInfo() const {
        return _actual->getDeviceOutputsInfo();
    }
    const DataMap& getDeviceProfilingOutputsInfo() const {
        return _actual->getDeviceProfilingOutputsInfo();
    }
    const vpux::QuantizationParamMap& getQuantParamsInfo() const {
        return _actual->getQuantParamsInfo();
    }
    const std::vector<char>& getCompiledNetwork() const {
        return _actual->getCompiledNetwork();
    }
    const void* getNetworkModel() const {
        return _actual->getNetworkModel();
    }
    std::size_t getNetworkModelSize() const {
        return _actual->getNetworkModelSize();
    }

    ~NetworkDescription() = default;

private:
    INetworkDescription::Ptr _actual;

    CompilerPluginPtr _plg;
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
    Compiler(ICompiler::Ptr com): _actual(com){};
#else
    Compiler(std::string libpath): _actual(std::move(libpath)){};
#endif

    std::shared_ptr<vpux::NetworkDescription> compile(const std::shared_ptr<ngraph::Function>& func,
                                                      const std::string& netName,
                                                      const InferenceEngine::InputsDataMap& inputsInfo,
                                                      const InferenceEngine::OutputsDataMap& outputsInfo,
                                                      const Config& config) {
        return std::make_shared<NetworkDescription>(_actual->compile(func, netName, inputsInfo, outputsInfo, config),
                                                    _actual);
    }

    InferenceEngine::QueryNetworkResult query(const InferenceEngine::CNNNetwork& network, const Config& config) {
        return _actual->query(network, config);
    }

    std::shared_ptr<vpux::NetworkDescription> parse(const std::vector<char>& network, const Config& config) {
        return std::make_shared<NetworkDescription>(_actual->parse(network, config, ""), _actual);
    }

    std::shared_ptr<vpux::NetworkDescription> parse(const std::string& filename, const Config& config) {
        return std::make_shared<NetworkDescription>(_actual->parse(filename, config), _actual);
    }

    std::shared_ptr<vpux::NetworkDescription> parse(std::istream& stream, const Config& config,
                                                    const std::string& graphName) {
        return std::make_shared<NetworkDescription>(_actual->parse(stream, config, graphName), _actual);
    }

private:
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
