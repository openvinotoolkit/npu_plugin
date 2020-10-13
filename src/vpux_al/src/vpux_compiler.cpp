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

#include <file_reader.h>
#include <file_utils.h>

#include <details/ie_exception.hpp>
#include <fstream>
#include <vpu/utils/io.hpp>
#include <vpux_compiler.hpp>

vpux::NetworkDescription::NetworkDescription(
    INetworkDescription::Ptr actual, InferenceEngine::details::SharedObjectLoader::Ptr plg)
    : _actual(actual), _plg(plg) {
    if (_actual == nullptr) {
        THROW_IE_EXCEPTION << "ExecutableNetwork wrapper was not initialized.";
    }
}

static std::string extractFileName(const std::string& fullPath) {
    const size_t lastSlashIndex = fullPath.find_last_of("/\\");
    return fullPath.substr(lastSlashIndex + 1);
}

std::shared_ptr<vpux::INetworkDescription> vpux::ICompiler::parse(
    const std::string& filename, const VPUXConfig& config) {
    std::ifstream stream(filename, std::ios::binary);
    if (!stream.is_open()) {
        THROW_IE_EXCEPTION << "Could not open file: " << filename;
    }
    const std::string graphName = extractFileName(filename);
    return parse(stream, config, graphName);
}

std::shared_ptr<vpux::INetworkDescription> vpux::ICompiler::parse(
    std::istream& stream, const VPUXConfig& config, const std::string& graphName) {
    const size_t graphSize = vpu::KmbPlugin::utils::getFileSize(stream);
    if (graphSize == 0) {
        THROW_IE_EXCEPTION << "Blob is empty";
    }
    auto blob = std::vector<char>(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
    return parse(blob, config, graphName);
}

vpux::Compiler::Ptr vpux::Compiler::create(vpux::CompilerType t) {
#ifdef ENABLE_EXPERIMENTAL_MLIR
    if (const auto env = std::getenv("IE_VPUX_USE_EXPERIMENTAL_COMPILER")) {
        if (std::stoi(env) != 0) {
            t = vpux::CompilerType::VPUXCompiler;
        }
    }
#endif

    switch (t) {
    case vpux::CompilerType::MCMCompiler: {
        return std::make_shared<Compiler>(getLibFilePath("frontend_mcm"));
    }
#ifdef ENABLE_EXPERIMENTAL_MLIR
    case vpux::CompilerType::VPUXCompiler: {
        return std::make_shared<Compiler>(getLibFilePath("vpux_compiler"));
    }
#endif
    default:
        THROW_IE_EXCEPTION << "Compiler type not found";
    }
    IE_ASSERT(false);
}

InferenceEngine::InputsDataMap vpux::helpers::dataMapIntoInputsDataMap(const vpux::DataMap& dataMap) {
    InferenceEngine::InputsDataMap inputsDataMap = {};

    for (const auto& input : dataMap) {
        InferenceEngine::InputInfo info;
        info.setInputData(input.second);
        inputsDataMap.insert({input.first, std::make_shared<InferenceEngine::InputInfo>(info)});
    }

    return inputsDataMap;
}

InferenceEngine::OutputsDataMap vpux::helpers::dataMapIntoOutputsDataMap(const vpux::DataMap& dataMap) {
    InferenceEngine::OutputsDataMap outputsDataMap = {};

    for (const auto& output : dataMap) {
        outputsDataMap.insert({output.first, output.second});
    }

    return outputsDataMap;
}
