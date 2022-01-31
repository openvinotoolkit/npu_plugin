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

#include "vpux_compiler.hpp"

#include "vpux.hpp"
#include "vpux/al/config/common.hpp"
#include "vpux/al/config/compiler.hpp"
#include "vpux/utils/core/logger.hpp"

#include <file_reader.h>
#include <file_utils.h>
#include <openvino/util/shared_object.hpp>

#include <fstream>

#ifdef OPENVINO_STATIC_LIBRARY
#include "vpux/compiler/compiler.hpp"
#endif

vpux::NetworkDescription::NetworkDescription(INetworkDescription::Ptr impl, const std::shared_ptr<void>& so)
        : _impl(impl), _so(so) {
    if (_impl == nullptr) {
        IE_THROW() << "ExecutableNetwork wrapper was not initialized.";
    }
}

static std::string extractFileName(const std::string& fullPath) {
    const size_t lastSlashIndex = fullPath.find_last_of("/\\");
    return fullPath.substr(lastSlashIndex + 1);
}

std::shared_ptr<vpux::INetworkDescription> vpux::ICompiler::parse(const std::string& filename, const Config& config) {
    std::ifstream stream(filename, std::ios::binary);
    if (!stream.is_open()) {
        IE_THROW() << "Could not open file: " << filename;
    }
    const std::string graphName = extractFileName(filename);
    return parse(stream, config, graphName);
}

std::shared_ptr<vpux::INetworkDescription> vpux::ICompiler::parse(std::istream& stream, const Config& config,
                                                                  const std::string& graphName) {
    const size_t graphSize = vpu::KmbPlugin::utils::getFileSize(stream);
    if (graphSize == 0) {
        IE_THROW() << "Blob is empty";
    }
    std::vector<char> blob(graphSize);
    stream.read(blob.data(), graphSize);
    return parse(blob, config, graphName);
}

vpux::Compiler::Ptr vpux::Compiler::create(const Config& config) {
    vpux::Logger logger("CompilerCreate", config.get<LOG_LEVEL>());
#ifdef OPENVINO_STATIC_LIBRARY
    // use vpux compiler
    (void)(config);
    const auto mlir = std::make_shared<vpux::CompilerImpl>();
    return std::make_shared<Compiler>(mlir);
#else
    const auto compilerType = config.get<COMPILER_TYPE>();

    switch (compilerType) {
    case InferenceEngine::VPUXConfigParams::CompilerType::MCM: {
        logger.info("MCM compiler will be used.");
        return std::make_shared<Compiler>(getLibFilePath("vpux_mcm_frontend"));
    }
    case InferenceEngine::VPUXConfigParams::CompilerType::MLIR: {
        logger.info("MLIR compiler will be used.");
        return std::make_shared<Compiler>(getLibFilePath("vpux_mlir_compiler"));
    }
    case InferenceEngine::VPUXConfigParams::CompilerType::DRIVER: {
        logger.info("Driver compiler will be used.");
        return std::make_shared<Compiler>(getLibFilePath("vpux_driver_compiler_adapter"));
    }
    default:
        IE_THROW() << "Compiler type not found";
    }
#endif
}

#ifndef OPENVINO_STATIC_LIBRARY
vpux::Compiler::Compiler(const std::string& libpath) {
    using CreateFuncT = void (*)(std::shared_ptr<ICompiler>&);
    static constexpr auto CreateFuncName = "CreateVPUXCompiler";

    _so = ov::util::load_shared_object(libpath.c_str());

    const auto createFunc = reinterpret_cast<CreateFuncT>(ov::util::get_symbol(_so, CreateFuncName));
    createFunc(_impl);
}
#endif

InferenceEngine::InputsDataMap vpux::helpers::dataMapIntoInputsDataMap(const vpux::DataMap& dataMap) {
    InferenceEngine::InputsDataMap inputsDataMap = {};

    for (const auto& input : dataMap) {
        InferenceEngine::InputInfo info;
        info.setInputData(std::make_shared<InferenceEngine::Data>(*input.second));
        inputsDataMap.insert({input.first, std::make_shared<InferenceEngine::InputInfo>(info)});
    }

    return inputsDataMap;
}

InferenceEngine::OutputsDataMap vpux::helpers::dataMapIntoOutputsDataMap(const vpux::DataMap& dataMap) {
    InferenceEngine::OutputsDataMap outputsDataMap = {};

    for (const auto& output : dataMap) {
        outputsDataMap.insert({output.first, std::make_shared<InferenceEngine::Data>(*output.second)});
    }

    return outputsDataMap;
}

vpux::OVNodes vpux::helpers::ovRawNodesIntoOVNodes(const std::vector<vpux::OVRawNode>& rawNodes, const bool isResult) {
    vpux::OVNodes nodes;
    for (const auto& rawNode : rawNodes) {
        std::shared_ptr<ov::Node> parameter;
        parameter = std::make_shared<ov::op::v0::Parameter>(rawNode.type, rawNode.shape);
        if (isResult) {
            const auto fakeParameter = parameter;
            parameter = std::make_shared<ov::op::v0::Result>(parameter);
            fakeParameter->set_friendly_name(rawNode.inputName);
            parameter = parameter->copy_with_new_inputs({fakeParameter});
        }
        parameter->set_friendly_name(rawNode.friendlyName);
        parameter->output(0).get_tensor().set_names(rawNode.tensorNames);
        nodes.push_back(parameter);
    }
    return nodes;
}
