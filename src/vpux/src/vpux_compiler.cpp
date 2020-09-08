
#include <file_reader.h>
#include <file_utils.h>

#include <details/ie_exception.hpp>
#include <fstream>
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
    auto root = InferenceEngine::getIELibraryPath();
    switch (t) {
    case vpux::CompilerType::MCMCompiler: {
#ifdef __unix__
        std::string lib_name = "/libfrontend_mcm.so";
#else
        std::string lib_name = "/frontend_mcm.dll";
#endif
        return std::make_shared<Compiler>(root + lib_name);
    }
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
