//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux_compiler.hpp"

#include "vpux/al/config/common.hpp"
#include "vpux/al/config/compiler.hpp"
#include "vpux/utils/IE/itt.hpp"
#include "vpux/utils/core/file_reader.hpp"
#include "vpux/utils/core/logger.hpp"

#include <file_utils.h>
#include <openvino/util/shared_object.hpp>

#include <fstream>

#ifdef OPENVINO_STATIC_LIBRARY

#ifdef ENABLE_MLIR_COMPILER
#include "vpux/compiler/compiler.hpp"
#endif

#ifdef ENABLE_DRIVER_COMPILER_ADAPTER
#include "vpux_driver_compiler_adapter.h"
using vpux::driverCompilerAdapter::LevelZeroCompilerAdapter;
#endif

#else
#include "vpux/utils/core/library_path.hpp"

#endif

vpux::NetworkDescription::NetworkDescription(INetworkDescription::Ptr impl, const std::shared_ptr<void>& so)
        : _impl(impl), _so(so) {
    if (_impl == nullptr) {
        OPENVINO_THROW("ExecutableNetwork wrapper was not initialized.");
    }
}

static std::string extractFileName(const std::string& fullPath) {
    const size_t lastSlashIndex = fullPath.find_last_of("/\\");
    return fullPath.substr(lastSlashIndex + 1);
}

std::shared_ptr<vpux::INetworkDescription> vpux::ICompiler::parse(const std::string& filename, const Config& config) {
    std::ifstream stream(filename, std::ios::binary);
    if (!stream.is_open()) {
        OPENVINO_THROW("Could not open file: ", filename);
    }
    const std::string graphName = extractFileName(filename);
    return parse(stream, config, graphName);
}

std::shared_ptr<vpux::INetworkDescription> vpux::ICompiler::parse(std::istream& stream, const Config& config,
                                                                  const std::string& graphName) {
    OV_ITT_TASK_CHAIN(ICOMPILER_PARSE, itt::domains::VPUXPlugin, "ICompiler::parse", "getFileSize");
    const size_t graphSize = vpux::getFileSize(stream);
    if (graphSize == 0) {
        OPENVINO_THROW("Blob is empty");
    }
    std::vector<char> blob(graphSize);
    OV_ITT_TASK_NEXT(ICOMPILER_PARSE, "read_blob");
    stream.read(blob.data(), graphSize);
    OV_ITT_TASK_NEXT(ICOMPILER_PARSE, "parse");
    return parse(blob, config, graphName);
}

vpux::Compiler::Ptr vpux::Compiler::create(const Config& config) {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "vpux::Compiler::create");
    vpux::Logger logger("CompilerCreate", config.get<LOG_LEVEL>());

    std::shared_ptr<Compiler> compiler;
    const auto compilerType = config.get<COMPILER_TYPE>();

    switch (compilerType) {
#ifdef ENABLE_MLIR_COMPILER
    case InferenceEngine::VPUXConfigParams::CompilerType::MLIR: {
        logger.info("MLIR compiler will be used.");

#ifndef OPENVINO_STATIC_LIBRARY
        compiler = std::make_shared<Compiler>(getLibFilePath("npu_mlir_compiler"));
#else
        const auto compilerInterface = std::make_shared<vpux::CompilerImpl>();
        compiler = std::make_shared<Compiler>(compilerInterface);
#endif

        break;
    }
#endif

#ifdef ENABLE_DRIVER_COMPILER_ADAPTER
    case InferenceEngine::VPUXConfigParams::CompilerType::DRIVER: {
        logger.info("Driver compiler will be used.");

#ifndef OPENVINO_STATIC_LIBRARY
        compiler = std::make_shared<Compiler>(getLibFilePath("npu_driver_compiler_adapter"));
#else
        const auto compilerInterface = std::make_shared<LevelZeroCompilerAdapter>();
        compiler = std::make_shared<Compiler>(compilerInterface);
#endif

        break;
    }
#endif

    default:
        OPENVINO_THROW("Compiler type not found");
    }

    return compiler;
}

#ifndef OPENVINO_STATIC_LIBRARY
vpux::Compiler::Compiler(const std::string& libpath) {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "vpux::Compiler::CreateVPUXCompiler");

    try {
        using CreateFuncT = void (*)(std::shared_ptr<ICompiler>&);
        static constexpr auto CreateFuncName = "CreateVPUXCompiler";

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        _so = ov::util::load_shared_object(ov::util::string_to_wstring(libpath).c_str());
#else
        _so = ov::util::load_shared_object(libpath.c_str());
#endif

        const auto createFunc = reinterpret_cast<CreateFuncT>(ov::util::get_symbol(_so, CreateFuncName));

        createFunc(_impl);
    } catch (const std::exception& ex) {
        OPENVINO_THROW("Got an error during compiler creation: ", ex.what());
    } catch (...) {
        OPENVINO_THROW("Got an unknown error during compiler creation");
    }
}
#endif

vpux::OVNodes vpux::helpers::nodeDescriptorsIntoNodes(const IONodeDescriptorMap& nodeDescriptors,
                                                      const std::vector<std::string>& names, const bool isResult) {
    vpux::OVNodes nodes;

    for (const std::string& name : names) {
        const IONodeDescriptor& nodeDescriptor = nodeDescriptors.at(name);

        std::shared_ptr<ov::Node> parameter;
        parameter = std::make_shared<ov::op::v0::Parameter>(nodeDescriptor.precision, nodeDescriptor.transposedShape);
        if (isResult) {
            const auto fakeParameter = parameter;
            parameter = std::make_shared<ov::op::v0::Result>(parameter);
            fakeParameter->set_friendly_name(nodeDescriptor.legacyName);
            parameter = parameter->copy_with_new_inputs({fakeParameter});
        }
        parameter->set_friendly_name(nodeDescriptor.currentNodeName);
        parameter->output(0).get_tensor().set_names(nodeDescriptor.outputTensorNames);
        nodes.push_back(parameter);
    }

    return nodes;
}
