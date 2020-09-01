
#include <file_reader.h>
#include <file_utils.h>

#include <details/ie_exception.hpp>
#include <details/ie_so_pointer.hpp>
#include <fstream>
#include <vpux_compiler.hpp>

class Compiler : public vpux::ICompiler {
public:
    Compiler(std::string lib_path): _impl(std::move(lib_path)){};

    std::shared_ptr<vpux::NetworkDescription> compile(
        InferenceEngine::ICNNNetwork& network, const vpux::VPUXConfig& config) override {
        return _impl->compile(network, config);
    }

    std::shared_ptr<vpux::NetworkDescription> compile(const std::shared_ptr<ngraph::Function>& func,
        const std::string& netName, const InferenceEngine::InputsDataMap& inputsInfo,
        const InferenceEngine::OutputsDataMap& outputsInfo, const vpux::VPUXConfig& config) override {
        return _impl->compile(func, netName, inputsInfo, outputsInfo, config);
    }

    std::shared_ptr<vpux::NetworkDescription> parse(
        const std::vector<char>& network, const vpux::VPUXConfig& config, const std::string& graphName) override {
        return _impl->parse(network, config, graphName);
    }

    std::set<std::string> getSupportedLayers(InferenceEngine::ICNNNetwork& network) override {
        return _impl->getSupportedLayers(network);
    }

    std::unordered_set<std::string> getSupportedOptions() override { return _impl->getSupportedOptions(); };

private:
    using ICompilerPtr = InferenceEngine::details::SOPointer<vpux::ICompiler>;
    ICompilerPtr _impl;
};

static std::string extractFileName(const std::string& fullPath) {
    const size_t lastSlashIndex = fullPath.find_last_of("/\\");
    return fullPath.substr(lastSlashIndex + 1);
}

std::shared_ptr<vpux::NetworkDescription> vpux::ICompiler::parse(
    const std::string& filename, const VPUXConfig& config) {
    std::ifstream stream(filename, std::ios::binary);
    if (!stream.is_open()) {
        THROW_IE_EXCEPTION << "Could not open file: " << filename;
    }
    const std::string graphName = extractFileName(filename);
    return parse(stream, config, graphName);
}

std::shared_ptr<vpux::NetworkDescription> vpux::ICompiler::parse(
    std::istream& stream, const VPUXConfig& config, const std::string& graphName) {
    const size_t graphSize = vpu::KmbPlugin::utils::getFileSize(stream);
    if (graphSize == 0) {
        THROW_IE_EXCEPTION << "Blob is empty";
    }
    auto blob = std::vector<char>(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
    return parse(blob, config, graphName);
}

std::shared_ptr<vpux::ICompiler> vpux::ICompiler::create(CompilerType t) {
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
