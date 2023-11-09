//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vcl_compiler.hpp"
#include "vcl_executable.hpp"
#include "vcl_query_network.hpp"

#include <cpp/ie_cnn_network.h>
#include <ie_core.hpp>
#include <openvino/openvino.hpp>

#include "vpux/al/config/common.hpp"
#include "vpux/al/config/compiler.hpp"
#include "vpux/al/config/runtime.hpp"
#include "vpux/al/opset/opset_version.hpp"
#include "vpux/vpux_plugin_config.hpp"
#include "vpux_compiler.hpp"
#include "vpux_private_config.hpp"

#define xstr(s) str(s)
#define str(s) #s

using namespace vpux;

/// Compiler version contains the info of code commit, compiler API version
static const char* COMPILER_VERSION =
        xstr(DRIVER_COMPILER_ID) "." xstr(VCL_COMPILER_VERSION_MAJOR) "." xstr(VCL_COMPILER_VERSION_MINOR);

namespace VPUXDriverCompiler {

VPUXCompilerL0::VPUXCompilerL0(vcl_compiler_desc_t desc, const std::map<std::string, std::string>& config,
                               VCLLogger* vclLogger)
        : _options(std::make_shared<OptionsDesc>()), _compilerDesc(desc), _logger(vclLogger) {
    /// Prepare default compilation configs
    registerCommonOptions(*_options);
    registerCompilerOptions(*_options);
    registerRunTimeOptions(*_options);

    Config parsedConfig(_options);
    parsedConfig.update(config, OptionMode::CompileTime);

    /// Create compiler instance with the default config
    _compiler = Compiler::create(parsedConfig);

    /// Update the compiler properties
    _compilerProp.id = COMPILER_VERSION;
    _compilerProp.version.major = VCL_COMPILER_VERSION_MAJOR;
    _compilerProp.version.minor = VCL_COMPILER_VERSION_MINOR;

    // If ov::get_available_opsets is upgraded to support new opset, this may not be supported by mlir compiler
    // Extract the latest int version from the opset string version, i.e., opset11 -> 11
    uint32_t largestVersion = vpux::extractOpsetVersion();
    _compilerProp.supportedOpsets = largestVersion;
}

std::pair<VPUXExecutableL0*, vcl_result_t> VPUXCompilerL0::importNetwork(BuildInfo& buildInfo) {
    InferenceEngine::CNNNetwork& cnnNet = buildInfo.cnnNet;
    std::shared_ptr<ov::Model> model = cnnNet.getFunction();
    VPUXExecutableL0* exe = nullptr;
    StopWatch stopWatch;
    if (buildInfo.enableProfiling) {
        /// Output time cost on vcl level
        stopWatch.start();
    }
    try {
        bool isNewAPI = false;
        bool isIRVersion11 = false;

        ov::RTMap& runtimeInfoMap = model->get_rt_info();
        const auto& isNewAPIMatch = runtimeInfoMap.find("is_new_api");
        if (isNewAPIMatch != runtimeInfoMap.end()) {
            isNewAPI = isNewAPIMatch->second.as<bool>();
        }

        const auto& irVersionMatch = runtimeInfoMap.find("version");
        if (irVersionMatch != runtimeInfoMap.end()) {
            const int64_t& irVersion = irVersionMatch->second.as<int64_t>();
            isIRVersion11 = (irVersion == 11);
        }

        if (!isNewAPI || !isIRVersion11) {
            /// Update input and output info
            auto inputs = cnnNet.getInputsInfo();
            auto outputs = cnnNet.getOutputsInfo();

            /// Update input precision with new value that parsed from user description
            for (const auto& item : buildInfo.inPrcsIE) {
                const auto& name = item.first;
                const auto input = inputs.find(name);
                if (input != inputs.end()) {
                    input->second->setPrecision(item.second);
                } else {
                    throw std::logic_error(name + " is not found in inputs to set precision!");
                }
            }

            /// Update input layout with new value that parsed from user description
            for (const auto& item : buildInfo.inLayoutsIE) {
                const auto& name = item.first;
                const auto input = inputs.find(name);
                if (input != inputs.end()) {
                    input->second->setLayout(item.second);
                } else {
                    throw std::logic_error(name + " is not found in inputs to set layout!");
                }
            }

            /// Update output precision with new value that parsed from user description
            for (const auto& item : buildInfo.outPrcsIE) {
                const auto& name = item.first;
                const auto output = outputs.find(name);
                if (output != outputs.end()) {
                    output->second->setPrecision(item.second);
                } else {
                    throw std::logic_error(name + " is not found in outputs to set precision!");
                }
            }

            /// Update output layout with new value that parsed from user description
            for (const auto& item : buildInfo.outLayoutsIE) {
                const auto& name = item.first;
                const auto output = outputs.find(name);
                if (output != outputs.end()) {
                    output->second->setLayout(item.second);
                } else {
                    throw std::logic_error(name + " is not found in outputs to set layout!");
                }
            }

            model->set_rt_info(inputs, "input_metadata");
            model->set_rt_info(outputs, "output_metadata");
        }

        /// Call compiler to compile the model and create blob
        /// Create executable with the result NetworkDescription, profiling option and logger
        exe = new VPUXExecutableL0(_compiler->compile(model, cnnNet.getName(), buildInfo.parsedConfig),
                                   buildInfo.enableProfiling, _logger);
    } catch (const std::exception& error) {
        _logger->outputError(formatv("{0}", error.what()));
        return std::pair<VPUXExecutableL0*, vcl_result_t>(nullptr, VCL_RESULT_ERROR_INVALID_ARGUMENT);
    } catch (...) {
        _logger->outputError("Internal exception! Can not compile!");
        return std::pair<VPUXExecutableL0*, vcl_result_t>(nullptr, VCL_RESULT_ERROR_INVALID_ARGUMENT);
    }

    if (buildInfo.enableProfiling) {
        stopWatch.stop();
        _logger->info("Compile net time: {0} ms", stopWatch.delta_ms());
    }

    return std::pair<VPUXExecutableL0*, vcl_result_t>(exe, VCL_RESULT_SUCCESS);
}

vcl_result_t VPUXCompilerL0::queryNetwork(const BuildInfo& buildInfo, VPUXQueryNetworkL0* pQueryNetwork) {
    _logger->info("Start to call query function from compiler to get supported layers!");
    ov::SupportedOpsMap queryNetworkResult;
    try {
        queryNetworkResult = _compiler->query(buildInfo.cnnNet.getFunction(), buildInfo.parsedConfig);
    } catch (const std::exception& error) {
        _logger->outputError(error.what());
        return VCL_RESULT_ERROR_UNKNOWN;
    } catch (...) {
        _logger->outputError("Failed to call query from compiler!");
        return VCL_RESULT_ERROR_UNKNOWN;
    }
    _logger->info("Successfully query supported layers!");

    /// Serialize the result to predefined format
    auto ret = pQueryNetwork->setQueryResult(queryNetworkResult);
    return ret;
}

}  // namespace VPUXDriverCompiler
